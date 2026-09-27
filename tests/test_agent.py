from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from ai_agent.agent import Agent
from ai_agent.browser import BrowserSession
from ai_agent.scenarios import load_personas
from ai_agent.tools import ACTION_NAMES, BrowserToolDispatcher, SESSION_PLAN_SCHEMA, ToolResult


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
    def __init__(self, plans: list[dict[str, object]]) -> None:
        self.plans = list(plans)
        self.calls = 0
        self.supplied_schemas: list[list[dict[str, object]]] = []
        self.messages: list[list[object]] = []

    def plan_session(self, messages: object, tool_schemas: object) -> SimpleNamespace:
        self.calls += 1
        self.supplied_schemas.append(tool_schemas)
        self.messages.append(messages)

        arguments = self.plans.pop(0)

        return SimpleNamespace(
            name="session_plan",
            arguments=arguments,
            assistant_message={
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "session_plan",
                            "arguments": arguments,
                        }
                    }
                ],
            },
        )

class BrowserToolDispatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = Mock(spec=BrowserSession)
        self.browser.observe_page.return_value = OBSERVATION
        self.browser.get_current_url.return_value = WEBSITE_URL
        self.dispatcher = BrowserToolDispatcher(self.browser)

    def test_exposes_one_session_plan_schema(self) -> None:
        schemas = self.dispatcher.available_tool_schemas()

        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "session_plan")

        parameters = schemas[0]["function"]["parameters"]

        self.assertEqual(
            set(parameters["required"]),
            {"actions"},
        )

        self.assertIn("actions", parameters["properties"])
        self.assertEqual(
            set(parameters["properties"]["actions"]["items"]["properties"]["action"]["enum"]),
            set(ACTION_NAMES),
    )

    def test_rejects_plan_without_finish(self) -> None:
        plan = {
            "actions": [
                {
                    "action": "view_product",
                    "target": "Studio Transitional Dining Set",
                }
            ]
        }

        with self.assertRaises(ValueError):
            BrowserToolDispatcher.validate_session_plan(
                plan,
                set(ACTION_NAMES),
                max_actions=10,
        )
            
    def test_rejects_action_after_finish(self) -> None:
        plan = {
            "actions": [
                {
                    "action": "finish",
                    "reason": "goal_completed",
                },
                {
                    "action": "go_back",
                },
            ]
        }

        with self.assertRaises(ValueError):
            BrowserToolDispatcher.validate_session_plan(
                plan,
                set(ACTION_NAMES),
                max_actions=10,
        )
            
    def test_rejects_plan_exceeding_max_actions(self) -> None:
        plan = {
            "actions": [
                {"action": "go_back"},
                {"action": "go_back"},
                {"action": "finish", "reason": "goal_completed"},
            ]
        }

        with self.assertRaises(ValueError):
            BrowserToolDispatcher.validate_session_plan(
                plan,
                set(ACTION_NAMES),
                max_actions=2,
        )

    def test_validates_complete_session_plan(self) -> None:
        plan = {
            "actions": [
                {
                    "action": "open_link",
                    "target": "Shop",
                    "reason": "Browse products",
                },
                {
                    "action": "view_product",
                    "target": "Studio Transitional Dining Set",
                    "reason": "Inspect a product",
                },
                {
                    "action": "go_back",
                    "target": None,
                    "reason": "Return to products",
                },
                {
                    "action": "finish",
                    "target": None,
                    "reason": "goal_completed",
                },
            ]
        }

        validated = BrowserToolDispatcher.validate_session_plan(
            plan,
            set(ACTION_NAMES),
            max_actions=10,
        )

        self.assertEqual(len(validated), 4)
        self.assertEqual(validated[-1][0], "finish")
    
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


# tests/test_agent.py

class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.browser = Mock(spec=BrowserSession)
        self.browser.page = Mock()
        self.browser.observe_page.return_value = OBSERVATION

        self.browser.navigate.return_value = {
            "url": "/product/test",
            "title": "Test Product",
        }

        self.persona = load_personas()["casual_browser"]

    def make_agent(self, llm: FakeLLM, **kwargs: object) -> Agent:
        return Agent(llm, self.browser, self.persona, debug=False, **kwargs)

    # tests/test_agent.py

    def test_executes_complete_plan_with_exactly_one_gemini_call(self) -> None:
        llm = FakeLLM([
            {
                "actions": [
                    {
                        "action": "view_product",
                        "target": "Test Product",
                        "reason": "Inspect product",
                    },
                    {
                        "action": "go_back",
                        "reason": "Return to previous page",
                    },
                    {
                        "action": "finish",
                        "reason": "Goal completed",
                    },
                ]
            }
        ])

        self.browser.navigate.return_value = {
            "url": "/product/test",
            "title": "Test Product",
        }

        result = self.make_agent(llm).run()

        self.assertEqual(llm.calls, 1)
    
    def test_rejects_plan_exceeding_maximum_action_limit(self) -> None:
        llm = FakeLLM([
            {
                "actions": [
                    {"action": "go_back"},
                    {"action": "go_back"},
                    {"action": "finish", "reason": "goal_completed"},
                ]
            }
        ])

        result = self.make_agent(llm, max_actions=2).run()

        self.assertEqual(result.termination_reason, "agent_error")
        self.assertEqual(result.actions, 0)

        # Gemini still called exactly once.
        self.assertEqual(llm.calls, 1)
        
    def test_action_failure_does_not_trigger_second_gemini_call(self) -> None:
        llm = FakeLLM([
            {
                "actions": [
                    {
                        "action": "view_product",
                        "target": "Studio Transitional Dining Set",
                        "reason": "Inspect product",
                    },
                    {
                        "action": "finish",
                        "reason": "goal_completed",
                    },
                ]
            }
        ])

        self.browser.navigate.side_effect = RuntimeError("browser failure")

        result = self.make_agent(llm).run()

        self.assertEqual(result.termination_reason, "agent_error")

        # Critical requirement:
        self.assertEqual(llm.calls, 1)

    def test_cart_remains_disabled_even_if_global_setting_is_true_for_read_only_persona(self) -> None:
        llm = FakeLLM([
            {
                "actions": [
                    {
                        "action": "add_to_cart",
                        "target": "A product",
                        "reason": "Test cart permission",
                    },
                    {
                        "action": "finish",
                        "reason": "agent_error",
                    },
                ]
            }
        ])
        result = self.make_agent(llm, allow_cart_actions=True).run()
        self.assertEqual(llm.calls, 1)
        self.assertEqual(result.termination_reason, "agent_error")
        self.assertEqual(result.tool_errors[0], "invalid_session_plan")
        self.assertEqual(result.tool_history, ())


if __name__ == "__main__":
    unittest.main()