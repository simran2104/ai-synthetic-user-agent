from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_agent.llm import (
    GeminiConfigurationError,
    GeminiLLM,
    LLMProtocolError,
)
from ai_agent.tools import SESSION_PLAN_SCHEMA
from main import load_settings, run_gemini_test


def function_call(
    name: str = "session_plan",
    arguments: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=(
            arguments
            if arguments is not None
            else {
                "actions": [
                    {
                        "action": "finish",
                        "reason": "goal_completed",
                    }
                ]
            }
        ),
    )


class GeminiProviderTests(unittest.TestCase):
    def make_client(self, response: object | None = None) -> Mock:
        client = Mock()
        client.interactions.create.return_value = response or SimpleNamespace(
            steps=[function_call()],
            output_text="",
            status="completed",
        )
        return client

    def test_missing_api_key_fails_clearly(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GeminiConfigurationError, "GEMINI_API_KEY is not set"):
                GeminiLLM("gemini-3.5-flash-lite", client=self.make_client())

    def test_initializes_with_environment_key_without_logging_or_echoing_it(self) -> None:
        secret = "test-only-not-a-real-key"
        with patch.dict(os.environ, {"GEMINI_API_KEY": secret}):
            provider = GeminiLLM("gemini-3.5-flash-lite", timeout_seconds=12, client=self.make_client())
        self.assertEqual(provider.model, "gemini-3.5-flash-lite")
        self.assertEqual(provider.provider, "gemini")
        self.assertEqual(provider.timeout_seconds, 12)
        self.assertNotIn(secret, repr(provider.__dict__))

    def test_parses_native_session_plan_function_call(self) -> None:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}):
            provider = GeminiLLM(
                "gemini-3.5-flash-lite",
                client=self.make_client(),
            )

            decision = provider.plan_session(
                [
                    {"role": "system", "content": "Return one complete plan."},
                    {"role": "user", "content": "Finish."},
                ],
                [SESSION_PLAN_SCHEMA],
            )

        self.assertEqual(decision.name, "session_plan")
        self.assertEqual(
            decision.arguments,
            {
                "actions": [
                    {
                        "action": "finish",
                        "reason": "goal_completed",
                    }
                ]
            },
    )

    def test_supplies_exactly_one_forced_session_plan_function(self) -> None:
        client = self.make_client()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}):
            provider = GeminiLLM(
                "gemini-3.5-flash-lite",
                client=client,
            )

            provider.plan_session(
                [{"role": "user", "content": "Finish."}],
                [SESSION_PLAN_SCHEMA],
            )

        kwargs = client.interactions.create.call_args.kwargs

        self.assertEqual(len(kwargs["tools"]), 1)
        self.assertEqual(
            kwargs["tools"][0]["name"],
            "session_plan",
        )

        self.assertEqual(
            kwargs["generation_config"]["tool_choice"]["allowed_tools"]["tools"],
            ["session_plan"],
        )

        self.assertFalse(kwargs["store"])

    def test_rejects_multiple_tools_before_request(self) -> None:
        client = self.make_client()

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}):
            provider = GeminiLLM(
                "gemini-3.5-flash-lite",
                client=client,
            )

            with self.assertRaises(ValueError):
                provider.plan_session(
                    [],
                    [SESSION_PLAN_SCHEMA, SESSION_PLAN_SCHEMA],
                )

        client.interactions.create.assert_not_called()

    def test_rejects_missing_or_multiple_native_function_calls(self) -> None:
        with self.assertRaises(LLMProtocolError):
            GeminiLLM._parse_function_call(SimpleNamespace(steps=[]))
        with self.assertRaises(LLMProtocolError):
            GeminiLLM._parse_function_call(SimpleNamespace(steps=[function_call(), function_call()]))

    def test_rejects_other_function_names(self) -> None:
        with self.assertRaisesRegex(
            LLMProtocolError,
            "expected 'session_plan'",
        ):
            GeminiLLM._parse_function_call(
                SimpleNamespace(
                    steps=[function_call("open_link")]
                )
            )

    def test_settings_define_gemini_without_provider_fallback(self) -> None:
        settings = load_settings()
        self.assertEqual(settings["gemini_model"], "gemini-3.5-flash-lite")

    def test_gemini_smoke_test_does_not_create_a_browser(self) -> None:
        settings = load_settings()
        fake_provider = Mock()
        fake_provider.plan_session.return_value = SimpleNamespace(
            name="session_plan",
            arguments={
                "actions": [
                    {
                        "action": "finish",
                        "reason": "goal_completed",
                    }
                ]
            },
        )
        with patch("main.GeminiLLM", return_value=fake_provider), patch("main.BrowserSession") as browser_type:
            result = run_gemini_test(settings)
        self.assertEqual(result, 0)
        fake_provider.plan_session.assert_called_once()
        browser_type.assert_not_called()


if __name__ == "__main__":
    unittest.main()