from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from ai_agent.browser import BrowserSession, BrowserSessionError, _OBSERVATION_SCRIPT


WEBSITE_URL = "https://furniture-mart-ps3p.onrender.com/"


class BrowserSessionTests(unittest.TestCase):
    def make_session(self, **kwargs: object) -> BrowserSession:
        session = BrowserSession(WEBSITE_URL, **kwargs)
        session.page = Mock()
        session.page.url = WEBSITE_URL
        return session

    def test_rejects_external_and_non_https_urls(self) -> None:
        with self.assertRaises(ValueError):
            BrowserSession("https://example.com/")
        with self.assertRaises(ValueError):
            BrowserSession("http://furniture-mart-ps3p.onrender.com/")

    def test_recognizes_render_cold_start_variants(self) -> None:
        self.assertTrue(BrowserSession._is_render_loading(503, "", ""))
        self.assertTrue(BrowserSession._is_render_loading(200, "Render - Application loading", ""))
        self.assertTrue(BrowserSession._is_render_loading(None, "", "SERVICE WAKING UP"))
        self.assertFalse(BrowserSession._is_render_loading(200, "Sarvotam Furniture", "Browse furniture"))

    def test_homepage_retries_after_render_loading_page(self) -> None:
        session = self.make_session(startup_timeout_seconds=1, startup_retry_interval_seconds=0.01)
        session.navigate = Mock(side_effect=[Mock(status=503), Mock(status=200)])
        session.get_page_title = Mock(side_effect=["Render - Application loading", "Sarvotam Furniture"])
        session.get_visible_text = Mock(side_effect=["SERVICE WAKING UP", "Browse furniture"])

        page = session.open_homepage()

        self.assertIs(page, session.page)
        self.assertEqual(session.navigate.call_count, 2)
        session.page.wait_for_timeout.assert_called_once()

    def test_homepage_timeout_reports_last_http_status(self) -> None:
        session = self.make_session(startup_timeout_seconds=1, startup_retry_interval_seconds=0.01)
        session.navigate = Mock(return_value=Mock(status=503))
        session.get_page_title = Mock(return_value="Render - Application loading")
        session.get_visible_text = Mock(return_value="SERVICE WAKING UP")

        with patch("ai_agent.browser.time.monotonic", side_effect=[0.0, 0.0, 1.1]):
            with self.assertRaisesRegex(BrowserSessionError, "last HTTP status=503"):
                session.open_homepage()
        self.assertEqual(session.navigate.call_count, 1)

    def test_non_503_http_error_fails_without_retry(self) -> None:
        session = self.make_session(startup_timeout_seconds=1, startup_retry_interval_seconds=0.01)
        session.navigate = Mock(return_value=Mock(status=500))
        session.get_page_title = Mock(return_value="Internal Server Error")
        session.get_visible_text = Mock(return_value="Server error")

        with self.assertRaisesRegex(BrowserSessionError, "HTTP 500"):
            session.open_homepage()
        self.assertEqual(session.navigate.call_count, 1)

    def test_observation_is_compact_and_includes_product_cards(self) -> None:
        session = self.make_session()
        session.page.title.return_value = "Furniture Mart"
        session.page.evaluate.return_value = {
            "headings": ["Furniture for every corner"],
            "links": [{"label": "Products", "url": f"{WEBSITE_URL}products/"}],
            "buttons": [{"label": "Search", "type": "submit"}],
            "inputs": [{"name": "q", "type": "text", "label": "", "placeholder": "Search furniture..."}],
            "products": [{
                "name": "Dining Table",
                "price": "₹12,000",
                "original_price": "₹15,000",
                "url": f"{WEBSITE_URL}product/dining-table/",
            }],
        }
        session.page.locator.return_value.inner_text.return_value = "  Browse   furniture  "

        observation = session.observe_page()

        self.assertEqual(observation["url"], WEBSITE_URL)
        self.assertEqual(observation["title"], "Furniture Mart")
        self.assertEqual(observation["visible_text"], "Browse furniture")
        self.assertEqual(observation["products"][0]["name"], "Dining Table")
        self.assertEqual(observation["products"][0]["original_price"], "₹15,000")
        self.assertNotIn("html", observation)
        self.assertIn("a.product-card", _OBSERVATION_SCRIPT)
        self.assertIn(".price", _OBSERVATION_SCRIPT)
        self.assertIn(".old", _OBSERVATION_SCRIPT)

    def test_screenshot_is_disabled_by_default(self) -> None:
        session = BrowserSession(WEBSITE_URL)
        self.assertIsNone(session.screenshot())


if __name__ == "__main__":
    unittest.main()