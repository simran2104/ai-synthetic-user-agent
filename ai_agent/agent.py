"""Controlled LLM-to-browser tool loop; the model cannot execute code."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .browser import BrowserSession, BrowserSessionError
from .llm import GeminiLLM, LLMError
from .prompts import SYSTEM_PROMPT
from .scenarios import Persona
from .tools import BrowserToolDispatcher, SESSION_PLAN_SCHEMA, ToolArgumentError


logger = logging.getLogger(__name__)
MAX_REPEATED_ACTIONS = 3


@dataclass(frozen=True)
class AgentRunResult:
    termination_reason: str
    actions: int
    detail: str = ""
    tool_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    llm_requests: int = 0
    duration_seconds: float = 0.0
    tool_errors: tuple[str, ...] = field(default_factory=tuple)


def compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep the model context to the most useful observed browser content."""
    return {
        "url": observation.get("url", ""),
        "title": observation.get("title", ""),
        "headings": observation.get("headings", [])[:12],
        "links": observation.get("links", [])[:18],
        "buttons": observation.get("buttons", [])[:16],
        "inputs": observation.get("inputs", [])[:12],
        "visible_text": observation.get("visible_text", "")[:900],
        "products": observation.get("products", [])[:10],
    }


class Agent:
    """Run bounded, schema-constrained browser actions for one persona."""

    def __init__(
        self,
        llm: GeminiLLM,
        browser: BrowserSession,
        persona: Persona,
        max_actions: int = 30,
        max_session_seconds: float = 300,
        max_repeated_action_repetitions: int = MAX_REPEATED_ACTIONS,
        allow_cart_actions: bool = False,
        debug: bool = True,
    ) -> None:
        if max_actions < 1 or max_session_seconds <= 0 or max_repeated_action_repetitions < 1:
            raise ValueError("Agent limits must be positive.")
        self.llm = llm
        self.browser = browser
        self.persona = persona
        self.max_actions = max_actions
        self.max_session_seconds = max_session_seconds
        self.max_repeated_action_repetitions = max_repeated_action_repetitions
        self.allow_cart_actions = allow_cart_actions and "cart" in persona.allowed_state_changing_actions
        self.debug = debug
        self.dispatcher = BrowserToolDispatcher(browser, allow_cart_actions=self.allow_cart_actions)

    @staticmethod
    def _state_fingerprint(observation: dict[str, Any]) -> str:
        state = {
            "url": observation.get("url"),
            "title": observation.get("title"),
            "headings": observation.get("headings"),
            "products": observation.get("products"),
            "visible_text": observation.get("visible_text"),
        }
        return json.dumps(state, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _action_fingerprint(name: str, arguments: dict[str, Any]) -> str:
        return json.dumps({"name": name, "arguments": arguments}, sort_keys=True, ensure_ascii=False)

    def _print(self, message: str) -> None:
        if self.debug:
            print(message)

    def run(self) -> AgentRunResult:
        """Request one complete Gemini plan, validate it, then execute locally."""
        history: list[dict[str, Any]] = []
        action_count = 0
        llm_requests = 0
        started_at = time.monotonic()
        repeat_counts: dict[tuple[str, str], int] = {}
        tool_errors: list[str] = []

        def result(reason: str, detail: str = "") -> AgentRunResult:
            duration = time.monotonic() - started_at
            logger.info(
                "agent_session_finished persona=%s gemini_requests=%s actions=%s duration_seconds=%.3f result=%s tool_errors=%s",
                self.persona.name,
                llm_requests,
                action_count,
                duration,
                reason,
                len(tool_errors),
            )
            return AgentRunResult(
                termination_reason=reason,
                actions=action_count,
                detail=detail,
                tool_history=tuple(history),
                llm_requests=llm_requests,
                duration_seconds=duration,
                tool_errors=tuple(tool_errors),
            )

        try:
            current_observation = self.browser.observe_page()
            self.dispatcher.remember_observation(current_observation)
        except (BrowserSessionError, Exception) as exc:
            logger.exception("agent_error while observing initial page")
            tool_errors.append(f"initial_observation: {type(exc).__name__}")
            return result("agent_error", f"Initial observation failed: {exc}")

        self._print(
            f"Session started\nPersona: {self.persona.name}\nGoal: {self.persona.goal}\n"
            f"Current URL: {current_observation['url']}"
        )
        website_capabilities = [
            "Home, product listings, categories, search, filters, sorting, pagination, and product details",
            "Cart, wishlist, login, account, enquiry, and contact functionality only where visible and allowed",
            "No verified online checkout, payment, or order workflow",
        ]
        decision_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({
                "persona": {"name": self.persona.name, "description": self.persona.description},
                "goal": self.persona.goal,
                "website_capabilities": website_capabilities,
                "current_observation": compact_observation(current_observation),
                "allowed_actions": self.dispatcher.allowed_actions(),
                "plan_requirements": {
                    "max_actions": self.max_actions,
                    "end_with": "finish",
                    "exactly_one_complete_plan": True,
                    "do_not_plan_actions_after_finish": True,
                },
            }, ensure_ascii=False)},
        ]
        if time.monotonic() - started_at >= self.max_session_seconds:
            return result("session_timeout", "Maximum session duration reached before planning.")

        llm_requests += 1
        self._print("Gemini request count: 1")
        try:
            plan_decision = self.llm.plan_session(decision_messages, [SESSION_PLAN_SCHEMA])
        except LLMError as exc:
            logger.exception("agent_error while requesting session plan")
            tool_errors.append(f"gemini_plan:{type(exc).__name__}")
            return result("agent_error", str(exc))

        if plan_decision.name != "session_plan":
            tool_errors.append("invalid_plan_function")
            return result("agent_error", f"Expected session_plan, received {plan_decision.name!r}.")
        try:
            plan = self.dispatcher.validate_session_plan(
                plan_decision.arguments,
                set(self.dispatcher.allowed_actions()),
                self.max_actions,
            )
        except ToolArgumentError as exc:
            logger.error("session_plan_invalid error=%s", exc)
            tool_errors.append("invalid_session_plan")
            return result("agent_error", f"Invalid session plan: {exc}")

        self._print(f"Planned actions: {len(plan)}")
        self._print(f"Action plan: {json.dumps(plan_decision.arguments, ensure_ascii=False)}")
        logger.info(
            "session_plan_valid persona=%s gemini_requests=1 planned_actions=%s",
            self.persona.name,
            len(plan),
        )

        for action_index, (action, action_arguments) in enumerate(plan, start=1):
            if time.monotonic() - started_at >= self.max_session_seconds:
                return result("session_timeout", "Maximum session duration reached.")

            action_count += 1
            self._print(f"\nAction #{action_count}\nPlanned action: {action}\nArguments: {json.dumps(action_arguments, ensure_ascii=False)}")
            pre_action_fingerprint = self._state_fingerprint(current_observation)
            try:
                tool_result = self.dispatcher.dispatch(action, action_arguments)
            except Exception as exc:
                logger.exception("plan_action_failed action=%s index=%s", action, action_index)
                tool_result = None
                error = f"{action}:{type(exc).__name__}"
                tool_errors.append(error)
                history.append({"action": action, "arguments": action_arguments, "success": False, "error": error})
                self._print(f"Action error: {error}; ending without another Gemini request.")
                return result("agent_error", f"Action {action_index} failed: {exc}")

            result_data = tool_result.to_dict()
            self._print(f"Action result:\n{json.dumps(result_data, indent=2, ensure_ascii=False)}")
            history.append({"action": action, "arguments": action_arguments, "result": result_data})
            if not tool_result.success:
                tool_errors.append(f"{action}:{tool_result.error or 'failed'}")
                return result("agent_error", f"Action {action_index} ({action}) failed: {tool_result.error or tool_result.detail}")

            if tool_result.finished:
                reason = tool_result.data["reason"]
                return result(reason)

            try:
                current_observation = self.browser.observe_page()
                self.dispatcher.remember_observation(current_observation)
            except Exception as exc:
                logger.exception("post_action_observation_failed action=%s", action)
                tool_errors.append(f"observation:{type(exc).__name__}")
                return result("agent_error", f"Post-action observation failed: {exc}")

            post_action_fingerprint = self._state_fingerprint(current_observation)
            action_fingerprint = self._action_fingerprint(action, action_arguments)
            repeat_key = (action_fingerprint, post_action_fingerprint)
            if post_action_fingerprint == pre_action_fingerprint:
                repeat_counts[repeat_key] = repeat_counts.get(repeat_key, 0) + 1
            else:
                repeat_counts = {key: count for key, count in repeat_counts.items() if key[0] != action_fingerprint}
            if repeat_counts.get(repeat_key, 0) >= self.max_repeated_action_repetitions:
                return result("repeated_action_limit", f"Action {action!r} repeated without page progress.")

        return result("agent_error", "Validated plan ended without finish.")
