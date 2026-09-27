from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_agent.llm import GeminiLLM, LLMCallDiagnostics
from diagnose_llm import ProbeResult
from ai_agent.tools import SESSION_PLAN_SCHEMA


def function_call() -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name="session_plan",
        arguments={
            "actions": [
                {
                    "action": "finish",
                    "reason": "goal_completed",
                }
            ]
        }
    )


class GeminiDiagnosticInstrumentationTests(unittest.TestCase):
    def test_records_gemini_request_metrics(self) -> None:
        client = Mock()
        client.interactions.create.return_value = SimpleNamespace(
            steps=[function_call()],
            output_text="",
            status="requires_action",
        )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}):
            llm = GeminiLLM("gemini-3.8-flash", client=client)
            diagnostics, response, decision, error = llm.diagnose_tool_call(
                "test-gemini",
                [{"role": "user", "content": "Finish."}],
                [{
                    "type": "function",
                    "function": {
                        "name": "session_plan",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "action": {
                                                "type": "string",
                                                "enum": ["finish"],
                                            },
                                            "reason": {
                                                "type": "string",
                                            },
                                        },
                                        "required": ["action"],
                                    },
                                }
                            },
                            "required": ["actions"],
                        },
                    }
                }],
            )
        self.assertTrue(diagnostics.http_success)
        self.assertEqual(diagnostics.provider, "gemini")
        self.assertEqual(diagnostics.model, "gemini-3.8-flash")
        self.assertEqual(diagnostics.tool_call_count, 1)
        self.assertEqual(diagnostics.parsed_tool_names, ("session_plan",))
        self.assertEqual(
            decision.arguments["actions"][0]["action"],
            "finish",
        )
        self.assertIsNotNone(response)
        self.assertIsNone(error)

    def test_request_error_records_failure_metadata(self) -> None:
        client = Mock()

        client.interactions.create.side_effect = TimeoutError("diagnostic timeout")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-placeholder"}):
            llm = GeminiLLM("gemini-3.8-flash", client=client)

            diagnostics, response, decision, error = llm.diagnose_tool_call(
                "test-timeout",
                [{"role": "user", "content": "Finish."}],
                [SESSION_PLAN_SCHEMA],
            )

        self.assertFalse(diagnostics.http_success)
        self.assertEqual(diagnostics.error_type, "TimeoutError")
        self.assertIsNone(response)
        self.assertIsNone(decision)
        self.assertIn("TimeoutError", error)


class DiagnosticProbeResultTests(unittest.TestCase):
    @staticmethod
    def diagnostic(tool_calls: int, names: tuple[str, ...] = ()) -> LLMCallDiagnostics:
        return LLMCallDiagnostics(
            model="gemini-3.5-flash-lite",
            message_count=2,
            message_characters=100,
            tool_count=1,
            tool_schema_characters=80,
            request_started_at="start",
            request_ended_at="end",
            duration_seconds=1.25,
            http_success=True,
            finish_reason="requires_action",
            response_content_characters=0,
            tool_call_count=tool_calls,
            parsed_tool_names=names,
            provider="gemini",
        )

    def test_pass_requires_one_schema_valid_tool_call(self) -> None:
        result = ProbeResult(
            self.diagnostic(1, ("next_action",)),
            "finish",
            {"action": "finish", "reason": "goal_completed"},
            True,
            None,
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.selected_action, "finish")

    def test_no_tool_call_is_a_failure_even_when_request_succeeds(self) -> None:
        result = ProbeResult(self.diagnostic(0), None, None, False, "no_tool_call")
        self.assertFalse(result.passed)
        self.assertEqual(result.error, "no_tool_call")

    def test_invalid_tool_arguments_are_a_failure(self) -> None:
        result = ProbeResult(
            self.diagnostic(1, ("next_action",)),
            "finish",
            {"action": "finish", "reason": "made_up"},
            False,
            "invalid_tool_call: reason is not allowed",
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.error.startswith("invalid_tool_call:"))


if __name__ == "__main__":
    unittest.main()