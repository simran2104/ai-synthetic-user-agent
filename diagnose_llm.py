"""Probe Gemini with one next_action tool and a real homepage observation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from ai_agent.browser import BrowserSession, BrowserSessionError
from ai_agent.llm import GeminiLLM, LLMCallDiagnostics
from ai_agent.prompts import SYSTEM_PROMPT
from ai_agent.scenarios import load_personas
from ai_agent.tools import ACTION_NAMES, BrowserToolDispatcher, NEXT_ACTION_SCHEMA, ToolArgumentError
from main import load_settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeResult:
    diagnostics: LLMCallDiagnostics
    selected_action: str | None
    arguments: dict[str, Any] | None
    schema_valid: bool
    error: str | None

    @property
    def passed(self) -> bool:
        return (
            self.diagnostics.http_success
            and self.diagnostics.tool_count == 1
            and self.diagnostics.tool_call_count == 1
            and self.schema_valid
        )


def run_homepage_probe() -> ProbeResult:
    """Observe the live site and request exactly one validated next_action call."""
    settings = load_settings()
    persona = load_personas()["casual_browser"]
    allow_cart_actions = (
        settings["allow_cart_actions"]
        and "cart" in persona.allowed_state_changing_actions
    )

    try:
        with BrowserSession(
            website_url=settings["website_url"],
            headless=settings["headless"],
            screenshots=False,
            startup_timeout_seconds=settings["startup_timeout_seconds"],
            startup_retry_interval_seconds=settings["startup_retry_interval_seconds"],
        ) as browser:
            browser.open_homepage()
            observation = browser.observe_page()
    except (BrowserSessionError, ValueError) as exc:
        diagnostics = LLMCallDiagnostics(
            model=settings["gemini_model"],
            message_count=0,
            message_characters=0,
            tool_count=1,
            tool_schema_characters=len(json.dumps(NEXT_ACTION_SCHEMA)),
            request_started_at="",
            request_ended_at="",
            duration_seconds=0,
            http_success=False,
            finish_reason=None,
            response_content_characters=0,
            tool_call_count=0,
            parsed_tool_names=(),
            error_type=type(exc).__name__,
            error=str(exc),
            provider="gemini",
        )
        return ProbeResult(diagnostics, None, None, False, str(exc))

    allowed_actions = [
        action for action in ACTION_NAMES
        if action != "add_to_cart" or allow_cart_actions
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({
            "persona": persona.name,
            "goal": persona.goal,
            "current_observation": observation,
            "allowed_actions": allowed_actions,
        }, ensure_ascii=False)},
    ]
    llm = GeminiLLM(model=settings["gemini_model"], timeout_seconds=settings["gemini_timeout_seconds"])
    diagnostics, _response, decision, error = llm.diagnose_tool_call(
        "single-next-action-homepage",
        messages,
        [NEXT_ACTION_SCHEMA],
    )

    selected_action: str | None = None
    schema_valid = False
    if decision is not None and decision.name == "next_action":
        try:
            selected_action, _action_arguments = BrowserToolDispatcher.validate_next_action(
                decision.arguments,
                set(allowed_actions),
            )
            schema_valid = True
        except ToolArgumentError as exc:
            error = str(exc)
    elif decision is not None:
        error = f"Expected next_action, received {decision.name!r}."

    return ProbeResult(
        diagnostics,
        selected_action,
        decision.arguments if decision is not None else None,
        schema_valid,
        error,
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        result = run_homepage_probe()
    except Exception as exc:
        logger.exception("homepage_probe_failed")
        print(f"Probe: FAIL; {type(exc).__name__}: {exc}")
        return 1

    diagnostics = result.diagnostics
    print(
        "Probe: {status}; provider={provider}; model={model}; tools={tools}; context_chars={context}; "
        "schema_chars={schema_chars}; time={duration:.2f}s; tokens={tokens}; "
        "done_reason={done_reason}; tool_calls={calls}; action={action}; "
        "schema_valid={valid}; raw={raw}".format(
            status="PASS" if result.passed else "FAIL",
            provider=diagnostics.provider,
            model=diagnostics.model,
            tools=diagnostics.tool_count,
            context=diagnostics.message_characters,
            schema_chars=diagnostics.tool_schema_characters,
            duration=diagnostics.duration_seconds,
            tokens=diagnostics.generated_tokens if diagnostics.generated_tokens is not None else "unknown",
            done_reason=diagnostics.finish_reason,
            calls=diagnostics.tool_call_count,
            action=result.selected_action or "none",
            valid=result.schema_valid,
            raw=diagnostics.raw_response_path or "not saved",
        )
    )
    if result.arguments:
        print(f"Arguments: {json.dumps(result.arguments, ensure_ascii=False)}")
    if result.error:
        print(f"Error: {result.error}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())