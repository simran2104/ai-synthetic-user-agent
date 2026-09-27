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
from .tools import BrowserToolDispatcher, NEXT_ACTION_SCHEMA


logger = logging.getLogger(__name__)
MAX_REPEATED_ACTIONS = 3


@dataclass(frozen=True)
class AgentRunResult:
    termination_reason: str
    actions: int
    detail: str = ""
    tool_history: tuple[dict[str, Any], ...] = field(default_factory=tuple)


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
        """Run until the model finishes or a configured safety limit is reached."""
        history: list[dict[str, Any]] = []
        action_count = 0
        started_at = time.monotonic()
        repeat_counts: dict[tuple[str, str], int] = {}
        last_result: dict[str, Any] | None = None

        try:
            current_observation = self.browser.observe_page()
            self.dispatcher.remember_observation(current_observation)
        except (BrowserSessionError, Exception) as exc:
            logger.exception("agent_error while observing initial page")
            return AgentRunResult("agent_error", 0, f"Initial observation failed: {exc}")

        self._print(f"Session started\nPersona: {self.persona.name}\nGoal: {self.persona.goal}\nCurrent URL: {current_observation['url']}")
        while True:
            if time.monotonic() - started_at >= self.max_session_seconds:
                return AgentRunResult("session_timeout", action_count, "Maximum session duration reached.", tuple(history))
            if action_count >= self.max_actions:
                return AgentRunResult("maximum_actions_reached", action_count, "Maximum action count reached.", tuple(history))

            try:
                decision_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps({
                        "persona": self.persona.name,
                        "goal": self.persona.goal,
                        "current_observation": compact_observation(current_observation),
                        "allowed_actions": self.dispatcher.allowed_actions(),
                        "previous_action_result": last_result,
                    }, ensure_ascii=False)},
                ]
                decision = self.llm.decide(decision_messages, [NEXT_ACTION_SCHEMA])
            except LLMError as exc:
                logger.exception("agent_error during structured LLM decision")
                return AgentRunResult("agent_error", action_count, str(exc), tuple(history))

            if decision.name != "next_action":
                return AgentRunResult(
                    "agent_error",
                    action_count,
                    f"Expected the next_action tool, received {decision.name!r}.",
                    tuple(history),
                )

            action_count += 1
            self._print(
                f"\nAction #{action_count}\nLLM tool: next_action\n"
                f"Arguments: {json.dumps(decision.arguments, ensure_ascii=False)}"
            )
            pre_action_fingerprint = self._state_fingerprint(current_observation)
            action_fingerprint = self._action_fingerprint("next_action", decision.arguments)

            result = self.dispatcher.dispatch_next_action(decision.arguments)
            result_data = result.to_dict()
            self._print(f"Tool result:\n{json.dumps(result_data, indent=2, ensure_ascii=False)}")
            history.append({"name": "next_action", "arguments": decision.arguments, "result": result_data})
            last_result = result_data

            if result.finished:
                reason = result.data["reason"]
                logger.info("agent_session_completed persona=%s actions=%s reason=%s", self.persona.name, action_count, reason)
                return AgentRunResult(reason, action_count, tool_history=tuple(history))

            try:
                current_observation = self.browser.observe_page()
                self.dispatcher.remember_observation(current_observation)
            except BrowserSessionError as exc:
                logger.exception("agent_error after tool execution")
                return AgentRunResult("agent_error", action_count, f"Post-action observation failed: {exc}", tuple(history))

            post_action_fingerprint = self._state_fingerprint(current_observation)
            repeat_key = (action_fingerprint, post_action_fingerprint)
            if post_action_fingerprint == pre_action_fingerprint:
                repeat_counts[repeat_key] = repeat_counts.get(repeat_key, 0) + 1
            else:
                repeat_counts = {
                    key: count for key, count in repeat_counts.items()
                    if key[0] != action_fingerprint
                }
            if repeat_counts.get(repeat_key, 0) >= self.max_repeated_action_repetitions:
                return AgentRunResult(
                    "repeated_action_limit",
                    action_count,
                    f"Action {decision.arguments.get('action')!r} repeated without page progress.",
                    tuple(history),
                )
