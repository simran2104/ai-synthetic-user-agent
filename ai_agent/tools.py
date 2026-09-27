"""Validated, bounded browser actions exposed to the LLM."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import Error as PlaywrightError

from .browser import ALLOWED_ORIGIN, BrowserSession, BrowserSessionError


logger = logging.getLogger(__name__)
FINISH_REASONS = (
    "goal_completed",
    "no_suitable_product",
    "maximum_actions_reached",
    "agent_error",
    "user_journey_complete",
)
BLOCKED_PATHS = (
    "/cart/add/",
    "/cart/update/",
    "/cart/remove/",
    "/wishlist/toggle/",
    "/enquiry/",
    "/contact/",
    "/register/",
    "/account/",
    "/accounts/",
    "/api/",
)


class ActionEffect(str, Enum):
    """Whether a browser tool can change website state."""

    READ_ONLY = "read_only"
    STATE_CHANGING = "state_changing"


@dataclass(frozen=True)
class ToolResult:
    """JSON-serializable outcome from one browser tool."""

    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    detail: str | None = None
    url: str | None = None
    finished: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.error:
            result["error"] = self.error
        if self.detail:
            result["detail"] = self.detail
        if self.url:
            result["url"] = self.url
        if self.data:
            result["data"] = self.data
        if self.finished:
            result["finished"] = True
        return result


ACTION_NAMES = (
    "open_link",
    "search",
    "view_product",
    "go_back",
    "add_to_cart",
    "view_cart",
    "finish",
)
TARGET_ACTIONS = frozenset({"open_link", "search", "view_product", "add_to_cart"})
SESSION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "session_plan",
        "description": "Return the complete bounded action plan for this synthetic visitor session.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": list(ACTION_NAMES)},
                            "target": {
                                "type": ["string", "null"],
                                "minLength": 1,
                                "maxLength": 180,
                                "description": (
                                    "For open_link use an observed link label or exact observed internal URL. "
                                    "For view_product use an observed product name or exact observed internal product URL. "
                                    "Never invent targets or use external URLs."
                                ),
                            },
                            "reason": {"type": "string", "maxLength": 160},
                        },
                        "required": ["action"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["actions"],
            "additionalProperties": False,
        },
    },
}
TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (SESSION_PLAN_SCHEMA,)
ACTION_ARGUMENT_SCHEMAS: dict[str, dict[str, Any]] = {
    "open_link": {"type": "object", "properties": {"link_text": {"type": "string", "minLength": 1, "maxLength": 160}}, "required": ["link_text"]},
    "search": {"type": "object", "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 160}}, "required": ["query"]},
    "view_product": {"type": "object", "properties": {"product_name": {"type": "string", "minLength": 1, "maxLength": 180}}, "required": ["product_name"]},
    "go_back": {"type": "object", "properties": {}, "required": []},
    "add_to_cart": {"type": "object", "properties": {"product_name": {"type": "string", "minLength": 1, "maxLength": 180}}, "required": ["product_name"]},
    "view_cart": {"type": "object", "properties": {}, "required": []},
    "finish": {"type": "object", "properties": {"reason": {"type": "string", "enum": list(FINISH_REASONS)}}, "required": ["reason"]},
}


class ToolArgumentError(ValueError):
    """Raised for arguments that do not match the declared tool schema."""


class BrowserToolDispatcher:
    """Validate and dispatch a fixed set of browser actions."""

    def __init__(self, browser: BrowserSession, allow_cart_actions: bool = False) -> None:
        self.browser = browser
        self.allow_cart_actions = allow_cart_actions
        self._observed_products: dict[str, dict[str, Any]] = {}
        self._schemas = ACTION_ARGUMENT_SCHEMAS
        self._handlers = {
            "observe_page": self._observe_page,
            "open_link": self._open_link,
            "search": self._search,
            "view_product": self._view_product,
            "go_back": self._go_back,
            "add_to_cart": self._add_to_cart,
            "view_cart": self._view_cart,
            "finish": self._finish,
        }

    def available_tool_schemas(self) -> list[dict[str, Any]]:
        """Return exactly one native tool schema to the language model."""
        return [SESSION_PLAN_SCHEMA]

    def allowed_actions(self) -> list[str]:
        """Return actions permitted by current session configuration."""
        return [
            action
            for action in ACTION_NAMES
            if action != "add_to_cart" or self.allow_cart_actions
        ]

    def remember_observation(self, observation: dict[str, Any]) -> None:
        """Cache products the browser actually observed for later detail actions."""
        self._remember_products(observation)

    @staticmethod
    def _validate_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ToolArgumentError("Tool arguments must be a JSON object.")
        function_schema = schema.get("function", {}).get("parameters", schema)
        properties = function_schema["properties"]
        required = function_schema["required"]
        unexpected = set(arguments) - set(properties)
        missing = set(required) - set(arguments)
        if unexpected:
            raise ToolArgumentError(f"Unexpected argument(s): {', '.join(sorted(unexpected))}.")
        if missing:
            raise ToolArgumentError(f"Missing required argument(s): {', '.join(sorted(missing))}.")
        for name, value in arguments.items():
            definition = properties[name]
            if definition["type"] == "string":
                if not isinstance(value, str):
                    raise ToolArgumentError(f"Argument {name!r} must be a string.")
                if len(value.strip()) < definition.get("minLength", 0):
                    raise ToolArgumentError(f"Argument {name!r} cannot be empty.")
                if len(value) > definition.get("maxLength", 10000):
                    raise ToolArgumentError(f"Argument {name!r} is too long.")
                if "enum" in definition and value not in definition["enum"]:
                    raise ToolArgumentError(f"Argument {name!r} is not an allowed value.")
        return arguments

    def dispatch(self, name: str, arguments: Any) -> ToolResult:
        """Validate and execute exactly one registered browser action."""
        logger.info("tool_action name=%s arguments=%s", name, arguments)
        if name == "add_to_cart" and not self.allow_cart_actions:
            return ToolResult(success=False, error="cart_actions_disabled", detail="Cart actions are disabled for this session.")
        schema = self._schemas.get(name)
        if schema is None:
            return ToolResult(success=False, error="unknown_tool", detail=f"Tool {name!r} is not available.")
        try:
            validated_arguments = self._validate_arguments(schema, arguments)
            result = self._handlers[name](validated_arguments)
        except ToolArgumentError as exc:
            result = ToolResult(success=False, error="invalid_arguments", detail=str(exc))
        except BrowserSessionError as exc:
            result = ToolResult(success=False, error="browser_error", detail=str(exc))
        except PlaywrightError as exc:
            result = ToolResult(success=False, error="browser_error", detail=str(exc))
        except Exception as exc:
            logger.exception("tool_execution_error name=%s", name)
            result = ToolResult(success=False, error="tool_execution_error", detail=str(exc))
        logger.info("tool_result name=%s success=%s error=%s", name, result.success, result.error)
        return result

    @staticmethod
    def validate_next_action(
        arguments: Any,
        allowed_actions: set[str] | frozenset[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Validate one plan item and convert it to an existing browser action."""
        if not isinstance(arguments, dict):
            raise ToolArgumentError("Each plan action must be an object.")
        unknown = set(arguments) - {"action", "target", "reason"}
        if unknown:
            raise ToolArgumentError(f"Unexpected plan field(s): {', '.join(sorted(unknown))}.")
        action = arguments.get("action")
        if not isinstance(action, str):
            raise ToolArgumentError("Each plan action requires a string 'action'.")
        if action not in ACTION_NAMES:
            raise ToolArgumentError(f"Unknown action {action!r}.")
        if allowed_actions is not None and action not in allowed_actions:
            raise ToolArgumentError(f"Action {action!r} is not allowed in this session.")

        target = arguments.get("target")
        if action in TARGET_ACTIONS:
            if not isinstance(target, str) or not target.strip():
                raise ToolArgumentError(f"Action {action!r} requires a non-empty target.")
            if len(target) > 180:
                raise ToolArgumentError(f"Action {action!r} target is too long.")
        elif target is not None:
            raise ToolArgumentError(f"Action {action!r} does not accept a target.")

        reason = arguments.get("reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 160):
            raise ToolArgumentError("Action reason must be a string of at most 160 characters.")
        if action == "finish":
            reason = reason or "goal_completed"
            if not isinstance(reason, str) or not reason.strip():
                raise ToolArgumentError(
                    "Finish reason must be a non-empty string."
                )
            action_arguments = {"reason": reason}
        elif action == "open_link":
            action_arguments = {"link_text": target}
        elif action == "search":
            action_arguments = {"query": target}
        elif action in {"view_product", "add_to_cart"}:
            action_arguments = {"product_name": target}
        else:
            action_arguments = {}
        return action, action_arguments

    @staticmethod
    def validate_session_plan(
        payload: Any,
        allowed_actions: set[str] | frozenset[str],
        max_actions: int,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Validate plan structure, permissions, bounds, and finish placement."""
        if not isinstance(payload, dict) or set(payload) != {"actions"}:
            raise ToolArgumentError("A session plan must contain only an 'actions' array.")
        actions = payload["actions"]
        if not isinstance(actions, list) or not actions:
            raise ToolArgumentError("A session plan must contain at least one action.")
        if len(actions) > max_actions:
            raise ToolArgumentError(f"The plan has {len(actions)} actions; the session limit is {max_actions}.")

        validated: list[tuple[str, dict[str, Any]]] = []
        for index, item in enumerate(actions):
            action, action_arguments = BrowserToolDispatcher.validate_next_action(item, allowed_actions)
            if action == "finish" and index != len(actions) - 1:
                raise ToolArgumentError("finish must be the final plan action.")
            if action in {"open_link", "view_product"}:
                target = item.get("target", "").strip()
                parsed = urlsplit(target)
                if (parsed.scheme or parsed.netloc) and not BrowserSession._is_allowed_url(target):
                    raise ToolArgumentError(f"{action} URL at action {index + 1} is outside the allowed Furniture Mart origin.")
                if action == "view_product" and (parsed.scheme or parsed.netloc) and not url_path(target).startswith("/product/"):
                    raise ToolArgumentError(f"view_product URL at action {index + 1} is not a product detail URL.")
            validated.append((action, action_arguments))

        if validated[-1][0] != "finish":
            raise ToolArgumentError("The complete session plan must end with finish.")
        return validated

    def dispatch_next_action(self, arguments: Any) -> ToolResult:
        """Validate a next_action payload, enforce permissions, and dispatch it."""
        logger.info("next_action_requested arguments=%s", arguments)
        if (
            isinstance(arguments, dict)
            and arguments.get("action") == "add_to_cart"
            and not self.allow_cart_actions
        ):
            return ToolResult(
                success=False,
                error="cart_actions_disabled",
                detail="Cart actions are disabled for this session.",
            )
        try:
            action, action_arguments = self.validate_next_action(arguments, set(self.allowed_actions()))
        except ToolArgumentError as exc:
            return ToolResult(success=False, error="invalid_action", detail=str(exc))
        return self.dispatch(action, action_arguments)

    def _observe_page(self, arguments: dict[str, Any]) -> ToolResult:
        observation = self.browser.observe_page()
        self._remember_products(observation)
        return ToolResult(success=True, data={"observation": observation}, url=observation["url"])

    def _open_link(self, arguments: dict[str, Any]) -> ToolResult:
        requested_target = arguments["link_text"].strip()
        observation = self.browser.observe_page()
        self._remember_products(observation)
        parsed_target = urlsplit(requested_target)
        target_is_url = bool(parsed_target.scheme or parsed_target.netloc)
        if target_is_url and not BrowserSession._is_allowed_url(requested_target):
            return ToolResult(success=False, error="invalid_url", detail="The requested URL is outside the allowed website.")
        if target_is_url:
            matches = [link for link in observation["links"] if link.get("url") == requested_target]
        else:
            requested_label = " ".join(requested_target.split()).casefold()
            matches = [
                link
                for link in observation["links"]
                if " ".join(link.get("label", "").split()).casefold() == requested_label
            ]
        targets = {link.get("url") for link in matches if link.get("url")}
        if not targets:
            return ToolResult(success=False, error="link_not_found", detail="No observed link matches that label.")
        if len(targets) != 1:
            return ToolResult(success=False, error="ambiguous_link", detail="That label points to multiple observed destinations.")
        target = targets.pop()
        if not BrowserSession._is_allowed_url(target):
            return ToolResult(success=False, error="invalid_url", detail="The observed link is outside the allowed website.")
        path = target.removeprefix(ALLOWED_ORIGIN)
        if any(path.startswith(blocked_path) for blocked_path in BLOCKED_PATHS):
            return ToolResult(success=False, error="restricted_destination", detail="This destination is not permitted in the current test.")
        self.browser.navigate(target)
        updated = self.browser.observe_page()
        return ToolResult(success=True, data={"observation": updated}, url=updated["url"])

    def _search(self, arguments: dict[str, Any]) -> ToolResult:
        if self.browser.page is None:
            raise BrowserSessionError("There is no active browser page.")
        search_input = self.browser.page.locator("form.search input[name='q']")
        if search_input.count() == 0:
            return ToolResult(success=False, error="search_control_missing", detail="The visible site search field was not found.")
        search_input.fill(arguments["query"], timeout=5000)
        search_input.press("Enter", timeout=10000)
        self.browser.page.wait_for_load_state("domcontentloaded", timeout=15000)
        if not BrowserSession._is_allowed_url(self.browser.get_current_url()):
            return ToolResult(success=False, error="invalid_url", detail="Search navigated outside the allowed website.")
        observation = self.browser.observe_page()
        self._remember_products(observation)
        return ToolResult(success=True, data={"observation": observation}, url=observation["url"])

    def _remember_products(self, observation: dict[str, Any]) -> None:
        for product in observation.get("products", []):
            name = product.get("name", "").strip().casefold()
            if name:
                self._observed_products[name] = product

    def _resolve_observed_product(self, product_name_or_url: str) -> dict[str, Any] | None:
        observation = self.browser.observe_page()
        self._remember_products(observation)
        target = product_name_or_url.strip()
        if BrowserSession._is_allowed_url(target):
            return next(
                (product for product in self._observed_products.values() if product.get("url") == target),
                None,
            )
        parsed_target = urlsplit(target)
        if parsed_target.scheme or parsed_target.netloc:
            return None
        return self._observed_products.get(target.casefold())

    def _view_product(self, arguments: dict[str, Any]) -> ToolResult:
        target = arguments["product_name"].strip()
        product = self._resolve_observed_product(target)
        if product is None:
            parsed_target = urlsplit(target)
            if parsed_target.scheme or parsed_target.netloc:
                error = "invalid_url" if not BrowserSession._is_allowed_url(target) else "product_not_found"
                detail = "The URL is not an observed internal product." if error == "product_not_found" else "The requested URL is outside the allowed website."
                return ToolResult(success=False, error=error, detail=detail)
            return ToolResult(success=False, error="product_not_found", detail="No matching product is present in the current page observation.")
        product_url = product.get("url", "")
        if not BrowserSession._is_allowed_url(product_url) or not url_path(product_url).startswith("/product/"):
            return ToolResult(success=False, error="invalid_url", detail="The observed product destination is not an allowed product page.")
        self.browser.navigate(product_url)
        observation = self.browser.observe_page()
        self._remember_products(observation)
        return ToolResult(
            success=True,
            data={
                "product": {
                    "name": product["name"],
                    "price": product.get("price", ""),
                    "original_price": product.get("original_price", ""),
                    "url": observation["url"],
                    "visible_information": observation["visible_text"],
                },
                "observation": observation,
            },
            url=observation["url"],
        )

    def _go_back(self, arguments: dict[str, Any]) -> ToolResult:
        if self.browser.page is None:
            raise BrowserSessionError("There is no active browser page.")
        previous_url = self.browser.get_current_url()
        self.browser.page.go_back(wait_until="domcontentloaded", timeout=15000)
        current_url = self.browser.get_current_url()
        if current_url == previous_url:
            return ToolResult(success=False, error="no_history", detail="There is no earlier page in this browser session.", url=current_url)
        if not BrowserSession._is_allowed_url(current_url):
            return ToolResult(success=False, error="invalid_url", detail="Back navigation left the allowed website.")
        observation = self.browser.observe_page()
        self._remember_products(observation)
        return ToolResult(success=True, data={"observation": observation}, url=current_url)

    def _add_to_cart(self, arguments: dict[str, Any]) -> ToolResult:
        product = self._resolve_observed_product(arguments["product_name"])
        if product is None:
            return ToolResult(success=False, error="product_not_found", detail="No matching product is present in the current page observation.")
        product_url = product.get("url", "")
        if not BrowserSession._is_allowed_url(product_url) or not url_path(product_url).startswith("/product/"):
            return ToolResult(success=False, error="invalid_url", detail="The observed product destination is not an allowed product page.")
        self.browser.navigate(product_url)
        if self.browser.page is None:
            raise BrowserSessionError("There is no active browser page.")
        add_button = self.browser.page.get_by_role("button", name="Add to shortlist cart", exact=True)
        if add_button.count() == 0:
            return ToolResult(success=False, error="add_to_cart_control_missing", detail="The product page has no observed add-to-cart control.")
        add_button.click(timeout=10000)
        self.browser.page.wait_for_load_state("domcontentloaded", timeout=15000)
        cart_result = self._load_cart()
        found = any(item["name"].casefold() == product["name"].casefold() for item in cart_result["products"])
        if not found:
            return ToolResult(
                success=False,
                error="cart_add_unverified",
                detail="The product was not visible in the cart after submitting the site form; website state may have changed.",
                data={"state_may_have_changed": True, "cart": cart_result},
                url=self.browser.get_current_url(),
            )
        return ToolResult(success=True, data={"cart": cart_result}, url=self.browser.get_current_url())

    def _load_cart(self) -> dict[str, Any]:
        self.browser.navigate(f"{ALLOWED_ORIGIN}/cart/")
        if self.browser.page is None:
            raise BrowserSessionError("There is no active browser page.")
        rows = self.browser.page.locator(".cart-row")
        products: list[dict[str, Any]] = []
        for index in range(rows.count()):
            row = rows.nth(index)
            name = row.locator("h3").inner_text().strip()
            details = row.locator("p").inner_text().strip()
            quantity_value = row.locator("input[name='quantity']").input_value()
            price_matches = re.findall(r"₹\s*[\d,]+(?:\.\d{1,2})?", details)
            products.append({
                "name": name,
                "quantity": int(quantity_value) if quantity_value.isdigit() else quantity_value,
                "price": price_matches[-1].replace(" ", "") if price_matches else "",
            })
        total_element = self.browser.page.locator(".total-box h2")
        estimated_total = total_element.inner_text().strip() if total_element.count() else None
        messages = [
            message.strip()
            for message in self.browser.page.locator(".messages div").all_inner_texts()
            if message.strip()
        ]
        return {
            "products": products,
            "estimated_total": estimated_total,
            "messages": messages,
            "visible_text": self.browser.get_visible_text(),
        }

    def _view_cart(self, arguments: dict[str, Any]) -> ToolResult:
        cart = self._load_cart()
        return ToolResult(success=True, data={"cart": cart}, url=self.browser.get_current_url())

    @staticmethod
    def _finish(arguments: dict[str, Any]) -> ToolResult:
        reason = arguments["reason"]
        logger.info("agent_finish reason=%s", reason)
        return ToolResult(success=True, data={"reason": reason}, finished=True)


def url_path(url: str) -> str:
    """Extract the path from an already validated internal URL."""
    from urllib.parse import urlsplit

    return urlsplit(url).path


