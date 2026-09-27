"""Run Gemini and Playwright browser tests."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from ai_agent.agent import Agent
from ai_agent.browser import BrowserSession, BrowserSessionError
from ai_agent.llm import GeminiLLM, LLMError
from ai_agent.scenarios import PersonaConfigurationError, load_personas
from ai_agent.tools import BrowserToolDispatcher, NEXT_ACTION_SCHEMA


SETTINGS_PATH = Path(__file__).resolve().parent / "agent_config" / "settings.json"
logger = logging.getLogger(__name__)


def load_settings(path: Path = SETTINGS_PATH) -> dict[str, Any]:
    """Load and minimally validate agent settings from JSON."""
    try:
        with path.open(encoding="utf-8") as settings_file:
            settings: Any = json.load(settings_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load settings from {path}: {exc}") from exc

    if not isinstance(settings, dict):
        raise ValueError("Agent settings must be a JSON object.")
    gemini_model = settings.get("gemini_model")
    if not isinstance(gemini_model, str) or not gemini_model.strip():
        raise ValueError("Agent settings must specify a non-empty 'gemini_model'.")
    gemini_timeout = settings.get("gemini_timeout_seconds")
    if isinstance(gemini_timeout, bool) or not isinstance(gemini_timeout, (int, float)) or gemini_timeout <= 0:
        raise ValueError("Agent settings must specify a positive 'gemini_timeout_seconds'.")
    for field in ("startup_timeout_seconds", "startup_retry_interval_seconds"):
        value = settings.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Agent settings must specify a positive number for {field!r}.")
    if not isinstance(settings.get("website_url"), str):
        raise ValueError("Agent settings must specify a 'website_url'.")
    if not isinstance(settings.get("headless"), bool):
        raise ValueError("Agent settings must specify a boolean 'headless' setting.")
    if not isinstance(settings.get("screenshots"), bool):
        raise ValueError("Agent settings must specify a boolean 'screenshots' setting.")
    for field in ("max_actions", "max_repeated_action_repetitions"):
        value = settings.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Agent settings must specify a positive integer for {field!r}.")
    for field in ("max_session_seconds",):
        value = settings.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"Agent settings must specify a positive number for {field!r}.")
    if not isinstance(settings.get("allow_cart_actions", False), bool):
        raise ValueError("Agent settings must specify a boolean 'allow_cart_actions' setting.")
    return settings


def run_browser_test(settings: dict[str, Any]) -> int:
    """Open the configured homepage, print its observation, and close Chromium."""
    try:
        with BrowserSession(
            website_url=settings["website_url"],
            headless=settings["headless"],
            screenshots=settings["screenshots"],
            startup_timeout_seconds=settings["startup_timeout_seconds"],
            startup_retry_interval_seconds=settings["startup_retry_interval_seconds"],
        ) as browser:
            browser.open_homepage()
            observation = browser.observe_page()
            print(f"URL: {browser.get_current_url()}")
            print(f"Title: {browser.get_page_title()}")
            print("Observation:")
            print(json.dumps(observation, indent=2, ensure_ascii=False))
            screenshot_path = browser.screenshot()
            if screenshot_path is not None:
                print(f"Screenshot: {screenshot_path}")
            else:
                print("Screenshot: disabled")
    except (BrowserSessionError, ValueError) as exc:
        print(f"Browser test failed: {exc}", file=sys.stderr)
        return 1
    return 0


def run_agent_test(settings: dict[str, Any], persona_name: str) -> int:
    """Run one configured persona with the active LLM provider and real browser."""
    try:
        if persona_name != "casual_browser":
            raise ValueError("Milestone 3 integration currently permits only the read-only casual_browser persona.")
        persona = load_personas().get(persona_name)
        if persona is None:
            raise PersonaConfigurationError(f"Persona {persona_name!r} is not configured.")
        llm = GeminiLLM(
            model=settings["gemini_model"],
            timeout_seconds=settings["gemini_timeout_seconds"],
        )
        with BrowserSession(
            website_url=settings["website_url"],
            headless=settings["headless"],
            screenshots=settings["screenshots"],
            startup_timeout_seconds=settings["startup_timeout_seconds"],
            startup_retry_interval_seconds=settings["startup_retry_interval_seconds"],
        ) as browser:
            browser.open_homepage()
            agent = Agent(
                llm=llm,
                browser=browser,
                persona=persona,
                max_actions=settings["max_actions"],
                max_session_seconds=settings["max_session_seconds"],
                max_repeated_action_repetitions=settings["max_repeated_action_repetitions"],
                allow_cart_actions=settings["allow_cart_actions"],
            )
            result = agent.run()
            screenshot_path = browser.screenshot(
                f"agent-{persona_name}-{datetime.now():%Y%m%d-%H%M%S}.png"
            )
            print(f"\nSession ended: {result.termination_reason}")
            print(f"Actions: {result.actions}")
            if result.detail:
                print(f"Detail: {result.detail}")
            if screenshot_path is not None:
                print(f"Screenshot: {screenshot_path}")
            return 1 if result.termination_reason in {
                "agent_error", "session_timeout", "repeated_action_limit", "maximum_actions_reached"
            } else 0
    except (BrowserSessionError, LLMError, PersonaConfigurationError, ValueError) as exc:
        logger.exception("agent_test_failed")
        print(f"Agent test failed: {exc}", file=sys.stderr)
        return 1

def run_gemini_test(settings: dict[str, Any]) -> int:
    """Make one minimal native next_action call without launching Playwright."""
    try:
        llm = GeminiLLM(
            model=settings["gemini_model"],
            timeout_seconds=settings["gemini_timeout_seconds"],
        )
        messages = [
            {
                "role": "system",
                "content": "You are a synthetic website user. Choose the next action. Return exactly one next_action function call. Do not provide an explanation.",
            },
            {
                "role": "user",
                "content": json.dumps({
                    "persona": "casual_browser",
                    "goal": "End this connectivity smoke test.",
                    "current_observation": {"url": settings["website_url"], "title": "Furniture Mart"},
                    "allowed_actions": ["finish"],
                }),
            },
        ]
        decision = llm.decide(messages, [NEXT_ACTION_SCHEMA])
        if decision.name != "next_action":
            raise ValueError(f"Expected next_action, received {decision.name!r}.")
        action, arguments = BrowserToolDispatcher.validate_next_action(decision.arguments, {"finish"})
        print(json.dumps({"tool": decision.name, "action": action, "arguments": arguments}, ensure_ascii=False))
    except (LLMError, ValueError) as exc:
        print(f"Gemini test failed: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the requested smoke test; browser mode never calls the LLM."""
    parser = argparse.ArgumentParser(description="Run Gemini and browser agent tests.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--browser-test", action="store_true", help="Open and observe the Furniture Mart homepage.")
    mode_group.add_argument("--agent-test", action="store_true", help="Run one LLM-driven browser persona.")
    mode_group.add_argument("--gemini-test", action="store_true", help="Test one Gemini next_action call without Playwright.")
    parser.add_argument("--persona", default="casual_browser", choices=("casual_browser",), help="Read-only persona for the first integration test.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        settings = load_settings()
        if args.browser_test:
            return run_browser_test(settings)
        if args.agent_test:
            return run_agent_test(settings, args.persona)
        if args.gemini_test:
            return run_gemini_test(settings)
        llm = GeminiLLM(
            model=settings["gemini_model"],
            timeout_seconds=settings["gemini_timeout_seconds"],
        )
        response = llm.generate("Reply with a short greeting identifying your configured provider.")
    except (LLMError, ValueError) as exc:
        print(f"LLM test failed: {exc}", file=sys.stderr)
        return 1

    print(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())