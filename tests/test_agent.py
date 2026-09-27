from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ai_agent.agent import Agent
from ai_agent.browser import BrowserSession
from ai_agent.scenarios import load_personas
from ai_agent.tools import ACTION_NAMES, BrowserToolDispatcher, NEXT_ACTION_SCHEMA, ToolResult


WEBSITE_URL = "https://furniture-mart-ps3p.onrender.com/"
OBSERVATION = {
    "url": WEBSITE_URL,
    "title": "Furniture Mart",
    "headings": ["Furniture"],
    "links": [{"label": "Shop", "url": f"{WEBSITE_URL}products/"}],
    "buttons": [],
    "inputs": [],
    "visible_text": "Browse furniture",
    "products": [{
        "name": "Studio Transitional Dining Set",
        "price": "₹2355",
        "original_price": "",
        "url": f"{WEBSITE_URL}product/studio-transitional-dining-set-16/",
    }],
}


def tool_response(name: str, arguments: dict[str, object]) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(message=SimpleNamespace(tool_calls=[SimpleNamespace(function=function)]))


class FakeLLM:
    def __init__(self, decisions: list[dict[str, object]]) -> None:
        self.decisions = list(decisions)
        self.calls = 0
        self.supplied_schemas: list[list[dict[str, object]]] = []
        self.messages: list[list[object]] = []

    def decide(self, messages: object, tool_schemas: object) -> SimpleNamespace:
        self.calls += 1
        self.supplied_schemas.append(tool_schemas)
        self.messages.append(messages)
        arguments = self.decisions.pop(0)
        return SimpleNamespace(
            name="next_action",
            arguments=arguments,
            assistant_message={"role": "assistant", "tool_calls": [{"function": {"name": "next_action", "arguments": arguments}}]},
        )


class BrowserToolDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = Mock(spec=BrowserSession)
        self.browser.observe_page.return_value = OBSERVATION
        self.browser.get_current_url.return_value = WEBSITE_URL
        self.dispatcher = BrowserToolDispatcher(self.browser)

    def test_exposes_one_next_action_schema(self) -> None:
        schemas = self.dispatcher.available_tool_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "next_action")
        self.assertEqual(set(schemas[0]["function"]["parameters"]["required"]), {"action"})
        self.assertEqual(set(schemas[0]["function"]["parameters"]["properties"]["action"]["enum"]), set(ACTION_NAMES))

    def test_accepts_each_supported_action(self) -> None:
        sample_arguments = {
            "open_link": {"action": "open_link", "target": "Shop"},
            "search": {"action": "search", "target": "table"},
            "view_product": {"action": "view_product", "target": "Dining Table"},
            "go_back": {"action": "go_back"},
            "add_to_cart": {"action": "add_to_cart", "target": "Dining Table"},
            "view_cart": {"action": "view_cart"},
            "finish": {"action": "finish", "reason": "goal_completed"},
        }
        for action in ACTION_NAMES:
            with self.subTest(action=action):
                allowed = set(ACTION_NAMES)
                normalized_action, _ = BrowserToolDispatcher.validate_next_action(sample_arguments[action], allowed)
                self.assertEqual(normalized_action, action)

    def test_rejects_invalid_action_missing_action_and_invalid_targets(self) -> None:
        invalid_payloads = (
            {"action": "run_code"},
            {"target": "Shop"},
            {"action": "search", "target": "   "},
            {"action": "finish", "target": "https://example.com"},
            {"action": "open_link", "target": "x" * 181},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    BrowserToolDispatcher.validate_next_action(payload, set(ACTION_NAMES))

    def test_rejects_unexpected_arguments(self) -> None:
        result = self.dispatcher.dispatch_next_action({"action": "open_link", "target": "Shop", "url": "https://example.com"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid_action")
        self.browser.navigate.assert_not_called()

    def test_rejects_unobserved_external_link(self) -> None:
        self.browser.observe_page.return_value = {
            **OBSERVATION,
            "links": [{"label": "Outside", "url": "https://example.com/"}],
        }
        result = self.dispatcher.dispatch("open_link", {"link_text": "Outside"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid_url")
        self.browser.navigate.assert_not_called()

    def test_dispatches_observed_internal_link(self) -> None:
        destination = f"{WEBSITE_URL}products/"
        self.browser.observe_page.side_effect = [OBSERVATION, {**OBSERVATION, "url": destination}]
        result = self.dispatcher.dispatch("open_link", {"link_text": "Shop"})
        self.assertTrue(result.success)
        self.browser.navigate.assert_called_once_with(destination)

    def test_open_link_resolves_observed_internal_url(self) -> None:
        destination = f"{WEBSITE_URL}products/"
        self.browser.observe_page.side_effect = [OBSERVATION, {**OBSERVATION, "url": destination}]
        result = self.dispatcher.dispatch_next_action({"action": "open_link", "target": destination})
        self.assertTrue(result.success)
        self.browser.navigate.assert_called_once_with(destination)

    def test_open_link_rejects_external_url_target(self) -> None:
        result = self.dispatcher.dispatch("open_link", {"link_text": "https://example.com/product/1/"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid_url")
        self.browser.navigate.assert_not_called()

    def test_view_product_resolves_observed_internal_url(self) -> None:
        product_url = OBSERVATION["products"][0]["url"]
        self.browser.observe_page.side_effect = [OBSERVATION, {**OBSERVATION, "url": product_url, "products": []}]
        result = self.dispatcher.dispatch_next_action({"action": "view_product", "target": product_url})
        self.assertTrue(result.success)
        self.browser.navigate.assert_called_once_with(product_url)
        self.assertEqual(result.data["product"]["name"], "Studio Transitional Dining Set")

    def test_view_product_rejects_external_url_target(self) -> None:
        result = self.dispatcher.dispatch("view_product", {"product_name": "https://example.com/product/1/"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "invalid_url")
        self.browser.navigate.assert_not_called()

    def test_cart_action_is_explicitly_disabled(self) -> None:
        result = self.dispatcher.dispatch_next_action({"action": "add_to_cart", "target": "Dining Table"})
        self.assertFalse(result.success)
        self.assertEqual(result.error, "cart_actions_disabled")

    def test_dispatches_next_action_through_existing_handlers(self) -> None:
        self.dispatcher.dispatch = Mock(return_value=ToolResult(success=True))
        actions = [
            ({"action": "open_link", "target": "Shop"}, "open_link", {"link_text": "Shop"}),
            ({"action": "search", "target": "table"}, "search", {"query": "table"}),
            ({"action": "view_product", "target": "Dining Table"}, "view_product", {"product_name": "Dining Table"}),
            ({"action": "go_back"}, "go_back", {}),
            ({"action": "view_cart"}, "view_cart", {}),
            ({"action": "finish"}, "finish", {"reason": "goal_completed"}),
        ]
        for payload, expected_name, expected_arguments in actions:
            with self.subTest(action=expected_name):
                self.dispatcher.dispatch.reset_mock()
                self.dispatcher.dispatch_next_action(payload)
                self.dispatcher.dispatch.assert_called_once_with(expected_name, expected_arguments)

    def test_finish_returns_recorded_reason(self) -> None:
        result = self.dispatcher.dispatch("finish", {"reason": "goal_completed"})
        self.assertTrue(result.success)
        self.assertTrue(result.finished)
        self.assertEqual(result.data["reason"], "goal_completed")


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = Mock(spec=BrowserSession)
        self.browser.observe_page.return_value = OBSERVATION
        self.persona = load_personas()["casual_browser"]

    def make_agent(self, llm: FakeLLM, **kwargs: object) -> Agent:
        return Agent(llm, self.browser, self.persona, debug=False, **kwargs)

    def test_finishes_when_model_calls_finish(self) -> None:
        llm = FakeLLM([{"action": "finish", "reason": "goal_completed"}])
        result = self.make_agent(llm).run()
        self.assertEqual(result.termination_reason, "goal_completed")
        self.assertEqual(result.actions, 1)
        self.assertEqual(len(llm.supplied_schemas[0]), 1)
        self.assertEqual(llm.supplied_schemas[0][0]["function"]["name"], "next_action")
        prompt_data = __import__("json").loads(llm.messages[0][1]["content"])
        self.assertIn("current_observation", prompt_data)
        self.assertIn("allowed_actions", prompt_data)
        self.assertNotIn("add_to_cart", prompt_data["allowed_actions"])

    def test_stops_at_maximum_action_limit(self) -> None:
        self.browser.get_current_url.return_value = WEBSITE_URL
        self.browser.page = Mock()
        llm = FakeLLM([{"action": "go_back"}, {"action": "go_back"}])
        result = self.make_agent(llm, max_actions=1).run()
        self.assertEqual(result.termination_reason, "maximum_actions_reached")
        self.assertEqual(result.actions, 1)
        self.assertEqual(llm.calls, 1)

    def test_stops_repeated_action_without_progress(self) -> None:
        self.browser.get_current_url.return_value = WEBSITE_URL
        self.browser.page = Mock()
        self.browser.page.go_back.return_value = None
        llm = FakeLLM([{"action": "go_back"}, {"action": "go_back"}])
        result = self.make_agent(llm, max_actions=10, max_repeated_action_repetitions=2).run()
        self.assertEqual(result.termination_reason, "repeated_action_limit")
        self.assertEqual(result.actions, 2)

    def test_cart_remains_disabled_even_if_global_setting_is_true_for_read_only_persona(self) -> None:
        llm = FakeLLM([
            {"action": "add_to_cart", "target": "A product"},
            {"action": "finish", "reason": "agent_error"},
        ])
        result = self.make_agent(llm, allow_cart_actions=True).run()
        self.assertEqual(result.tool_history[0]["result"]["error"], "cart_actions_disabled")


if __name__ == "__main__":
    unittest.main()