"""Playwright session management and compact observations of the storefront."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import Browser, BrowserContext, Error as PlaywrightError
from playwright.sync_api import Page, Playwright, Response, Route, sync_playwright


ALLOWED_ORIGIN = "https://furniture-mart-ps3p.onrender.com"
AGENT_SESSIONS_DIR = Path(__file__).resolve().parent.parent / "agent_data" / "sessions"
MAX_VISIBLE_TEXT_CHARS = 1800
MAX_OBSERVATION_ITEMS = 40
logger = logging.getLogger(__name__)

_OBSERVATION_SCRIPT = """() => {
    const visible = (element) => element.getClientRects().length > 0;
    const text = (element) => element ? (element.innerText || element.textContent || '').replace(/\\s+/g, ' ').trim() : '';
    const limited = (selector, mapper, limit = 40) =>
        Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, limit).map(mapper);
    return {
        headings: limited('h1, h2, h3, h4', (element) => text(element), 20),
        links: limited('a', (element) => ({
            label: text(element) || element.getAttribute('aria-label') || '',
            url: element.href
        })).filter((item) => item.label || item.url),
        buttons: limited('button, input[type=submit], input[type=button]', (element) => ({
            label: text(element) || element.getAttribute('aria-label') || element.value || '',
            type: element.getAttribute('type') || 'button'
        })).filter((item) => item.label),
        inputs: limited('input:not([type=hidden]):not([type=submit]):not([type=button]), textarea, select', (element) => ({
            name: element.name || '',
            type: element.tagName.toLowerCase() === 'textarea' ? 'textarea' : (element.type || element.tagName.toLowerCase()),
            label: element.labels && element.labels.length ? text(element.labels[0]) : (element.getAttribute('aria-label') || ''),
            placeholder: element.placeholder || ''
        })),
        products: limited('a.product-card', (element) => {
            const priceElement = element.querySelector('.price');
            const sellingPriceElement = priceElement ? priceElement.cloneNode(true) : null;
            if (sellingPriceElement) sellingPriceElement.querySelectorAll('.old').forEach((oldPrice) => oldPrice.remove());
            return {
                name: text(element.querySelector('h3')),
                price: text(sellingPriceElement),
                original_price: text(priceElement && priceElement.querySelector('.old')),
                url: element.href
            };
        }, 20).filter((item) => item.name)
    };
}"""


class BrowserSessionError(RuntimeError):
    """Raised when a browser session cannot be started or used safely."""


class BrowserSession:
    """Manage a Chromium session restricted to the Furniture Mart site."""

    def __init__(
        self,
        website_url: str,
        headless: bool = False,
        screenshots: bool = False,
        startup_timeout_seconds: float = 180,
        startup_retry_interval_seconds: float = 5,
    ) -> None:
        self.website_url = website_url
        self.headless = headless
        self.screenshots = screenshots
        self.startup_timeout_seconds = startup_timeout_seconds
        self.startup_retry_interval_seconds = startup_retry_interval_seconds
        self._validate_url(website_url)
        if startup_timeout_seconds <= 0 or startup_retry_interval_seconds <= 0:
            raise ValueError("Startup timeout and retry interval must both be greater than zero.")
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    @staticmethod
    def _is_allowed_url(url: str) -> bool:
        parsed = urlsplit(url)
        return parsed.scheme == "https" and parsed.netloc.lower() == "furniture-mart-ps3p.onrender.com"

    @classmethod
    def _validate_url(cls, url: str) -> None:
        if not cls._is_allowed_url(url):
            raise ValueError(f"Browser navigation is restricted to {ALLOWED_ORIGIN}/")

    @staticmethod
    def _is_render_loading(status: int | None, title: str, visible_text: str) -> bool:
        content = f"{title} {visible_text}".casefold()
        return (
            status == 503
            or "application loading" in content
            or "service waking up" in content
            or "incoming http request detected" in content
        )

    def start(self) -> BrowserSession:
        """Launch Chromium and create a page without navigating yet."""
        if self.page is not None:
            return self
        try:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=self.headless)
            self._context = self._browser.new_context()
            self._context.route("**/*", self._guard_navigation)
            self.page = self._context.new_page()
            self.page.on(
                "request",
                lambda request: (
                    print(f"GA4 REQUEST: {request.url}")
                    if "google-analytics.com" in request.url
                    else None
                )
            )
            def log_ga4_response(response):
                if "google-analytics.com" in response.url:
                    print(f"GA4 RESPONSE: {response.status} {response.url}")
            self.page.on("response", log_ga4_response)
        except (PlaywrightError, OSError) as exc:
            logger.exception("browser_error during startup")
            self.close()
            raise BrowserSessionError(f"Could not start the Furniture Mart browser session: {exc}") from exc
        logger.info("browser_started headless=%s", self.headless)
        return self

    def _guard_navigation(self, route: Route) -> None:
        request = route.request
        try:
            is_top_level_navigation = request.is_navigation_request() and request.frame.parent_frame is None
        except PlaywrightError:
            is_top_level_navigation = False

        if is_top_level_navigation and not self._is_allowed_url(request.url):
            logger.warning("browser_error blocked external navigation url=%s", request.url)
            route.abort("blockedbyclient")
            return
        route.continue_()

    def navigate(self, url: str, timeout_ms: int = 30000) -> Response | None:
        """Navigate the managed page, rejecting destinations outside the allowlist."""
        self._validate_url(url)
        if self.page is None:
            raise BrowserSessionError("Start the browser session before navigating.")
        logger.info("navigation_started url=%s", url)
        try:
            response = self.page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except PlaywrightError as exc:
            logger.exception("browser_error during navigation url=%s", url)
            raise BrowserSessionError(f"Could not navigate to {url}: {exc}") from exc
        logger.info("navigation_completed url=%s status=%s", url, response.status if response else "unknown")
        return response

    def open_homepage(self) -> Page:
        """Open the configured homepage, retrying while Render is waking up."""
        if self.page is None:
            raise BrowserSessionError("Start the browser session before opening the homepage.")

        deadline = time.monotonic() + self.startup_timeout_seconds
        last_status: int | None = None
        last_title = ""
        last_error = ""

        while True:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                response = self.navigate(self.website_url, timeout_ms=max(1000, int(remaining * 1000)))
                last_status = response.status if response else None
                last_title = self.get_page_title()
                visible_text = self.get_visible_text()
                if self._is_render_loading(last_status, last_title, visible_text):
                    logger.warning(
                        "render_cold_start_detected status=%s title=%s",
                        last_status,
                        last_title,
                    )
                    last_error = "Render returned its application-loading page"
                elif last_status is not None and last_status >= 400:
                    raise BrowserSessionError(
                        f"Furniture Mart returned HTTP {last_status}; title={last_title!r}."
                    )
                else:
                    logger.info("navigation_completed homepage_ready url=%s", self.get_current_url())
                    return self.page
            except BrowserSessionError as exc:
                if last_status is not None and last_status >= 400 and last_status != 503:
                    logger.error("browser_error homepage unavailable: %s", exc)
                    raise
                last_error = str(exc)
                logger.warning("browser_error temporary homepage navigation failure: %s", exc)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            wait_ms = max(1, int(min(self.startup_retry_interval_seconds, remaining) * 1000))
            try:
                self.page.wait_for_timeout(wait_ms)
            except PlaywrightError as exc:
                logger.exception("browser_error while waiting for Render startup")
                raise BrowserSessionError(f"Could not wait for the Furniture Mart homepage: {exc}") from exc

        detail = f"last HTTP status={last_status}, title={last_title!r}"
        if last_error:
            detail += f", detail={last_error}"
        error = BrowserSessionError(
            f"Furniture Mart did not become available within {self.startup_timeout_seconds:g} seconds "
            f"({detail})."
        )
        logger.error("browser_error %s", error)
        raise error

    def get_current_url(self) -> str:
        """Return the current page URL."""
        if self.page is None:
            raise BrowserSessionError("Start the browser session before reading the current URL.")
        return self.page.url

    def get_page_title(self) -> str:
        """Return the current page title."""
        if self.page is None:
            raise BrowserSessionError("Start the browser session before reading the page title.")
        try:
            return self.page.title()
        except PlaywrightError as exc:
            raise BrowserSessionError(f"Could not read the page title: {exc}") from exc

    def get_visible_text(self, max_chars: int = MAX_VISIBLE_TEXT_CHARS) -> str:
        """Return compact visible body text, capped to avoid oversized observations."""
        if self.page is None:
            raise BrowserSessionError("Start the browser session before reading visible text.")
        try:
            text = self.page.locator("body").inner_text(timeout=5000)
        except PlaywrightError as exc:
            raise BrowserSessionError(f"Could not read visible page text: {exc}") from exc
        compact = " ".join(text.split())
        return compact[:max(0, max_chars)]

    def observe_page(self) -> dict[str, Any]:
        """Return a compact semantic summary; never return the document HTML."""
        if self.page is None:
            raise BrowserSessionError("Start the browser session before observing the page.")
        try:
            elements: Any = self.page.evaluate(_OBSERVATION_SCRIPT)
        except PlaywrightError as exc:
            logger.exception("browser_error while creating page observation")
            raise BrowserSessionError(f"Could not observe the current page: {exc}") from exc
        if not isinstance(elements, dict):
            raise BrowserSessionError("The page observation script returned an invalid result.")

        observation = {
            "url": self.get_current_url(),
            "title": self.get_page_title(),
            "headings": elements.get("headings", [])[:20],
            "links": elements.get("links", [])[:MAX_OBSERVATION_ITEMS],
            "buttons": elements.get("buttons", [])[:MAX_OBSERVATION_ITEMS],
            "inputs": elements.get("inputs", [])[:MAX_OBSERVATION_ITEMS],
            "visible_text": self.get_visible_text(),
            "products": elements.get("products", [])[:20],
        }
        logger.info(
            "observation_created headings=%s links=%s products=%s",
            len(observation["headings"]),
            len(observation["links"]),
            len(observation["products"]),
        )
        return observation

    def screenshot(self, filename: str | None = None) -> Path | None:
        """Save a screenshot under agent_data/sessions only when enabled."""
        if not self.screenshots:
            return None
        if self.page is None:
            raise BrowserSessionError("Start the browser session before taking a screenshot.")
        name = Path(filename or f"browser-{datetime.now():%Y%m%d-%H%M%S}.png").name
        if not name.lower().endswith(".png"):
            name += ".png"
        screenshot_path = AGENT_SESSIONS_DIR / name
        try:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(screenshot_path), full_page=True)
        except (OSError, PlaywrightError) as exc:
            logger.exception("browser_error while saving screenshot")
            raise BrowserSessionError(f"Could not save screenshot to {screenshot_path}: {exc}") from exc
        return screenshot_path

    def close(self) -> None:
        """Close browser resources safely; repeated calls are harmless."""
        had_session = any((self.page, self._context, self._browser, self._playwright))
        for resource, close_method in (
            (self.page, "close"),
            (self._context, "close"),
            (self._browser, "close"),
            (self._playwright, "stop"),
        ):
            if resource is not None:
                try:
                    getattr(resource, close_method)()
                except PlaywrightError:
                    logger.exception("browser_error while cleaning up browser resources")
        self._context = None
        self._browser = None
        self._playwright = None
        self.page = None
        if had_session:
            logger.info("browser_closed")

    def __enter__(self) -> BrowserSession:
        return self.start()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()