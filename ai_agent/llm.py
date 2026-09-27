"""Gemini-backed language model interface for the synthetic browser agent."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Gemini provider failure."""


class LLMProtocolError(LLMError):
    """Raised when Gemini does not return one valid structured function call."""


class GeminiConfigurationError(LLMError):
    """Raised when Gemini configuration is incomplete."""


class GeminiError(LLMError):
    """Raised when the Gemini API request fails."""


@dataclass(frozen=True)
class LLMToolDecision:
    name: str
    arguments: dict[str, Any]
    assistant_message: Any


@dataclass(frozen=True)
class LLMCallDiagnostics:
    model: str
    message_count: int
    message_characters: int
    tool_count: int
    tool_schema_characters: int
    request_started_at: str
    request_ended_at: str
    duration_seconds: float
    http_success: bool
    finish_reason: str | None
    response_content_characters: int
    tool_call_count: int
    parsed_tool_names: tuple[str, ...]
    error_type: str | None = None
    error: str | None = None
    generated_tokens: int | None = None
    provider: str = "gemini"

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "message_count": self.message_count,
            "message_characters": self.message_characters,
            "tool_count": self.tool_count,
            "tool_schema_characters": self.tool_schema_characters,
            "request_started_at": self.request_started_at,
            "request_ended_at": self.request_ended_at,
            "duration_seconds": self.duration_seconds,
            "http_success": self.http_success,
            "finish_reason": self.finish_reason,
            "response_content_characters": self.response_content_characters,
            "tool_call_count": self.tool_call_count,
            "parsed_tool_names": list(self.parsed_tool_names),
            "error_type": self.error_type,
            "error": self.error,
            "generated_tokens": self.generated_tokens,
        }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _serialize(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=False, separators=(",", ":"))


class GeminiLLM:
    """Google GenAI provider using native Interactions API function calling."""

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 90,
        client: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A Gemini model name is required.")
        if timeout_seconds <= 0:
            raise ValueError("Gemini timeout must be greater than zero.")
        configured_key = os.environ.get("GEMINI_API_KEY")
        if not configured_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY is not set. Configure it in the process environment before using Gemini."
            )

        from google import genai
        from google.genai import types

        self.model = model
        self.provider = "gemini"
        self.timeout_seconds = timeout_seconds
        self._client = client or genai.Client(
            api_key=configured_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        """Generate plain text for a provider connectivity check."""
        if not prompt.strip():
            raise ValueError("A non-empty prompt is required.")
        started = time.perf_counter()
        try:
            response = self._client.interactions.create(
                model=self.model,
                input=prompt,
                system_instruction=system_prompt or None,
                store=False,
            )
        except Exception as exc:
            safe_error = self._safe_error(exc)
            logger.error(
                "llm_request_failed %s",
                json.dumps({
                    "provider": "gemini",
                    "model": self.model,
                    "duration_seconds": round(time.perf_counter() - started, 3),
                    "response_success": False,
                    "error_type": type(exc).__name__,
                    "error": safe_error,
                }, sort_keys=True),
            )
            raise GeminiError(f"Gemini request failed ({type(exc).__name__}): {safe_error}") from exc
        output = getattr(response, "output_text", "") or ""
        logger.info(
            "llm_request_completed %s",
            json.dumps({
                "provider": "gemini",
                "model": self.model,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response_success": True,
                "response_status": getattr(response, "status", None) or "completed",
                "structured_tool_call": False,
            }, sort_keys=True),
        )
        if not output:
            raise GeminiError("Gemini returned no text output.")
        return output

    @staticmethod
    def _parse_function_call(response: Any) -> LLMToolDecision:
        steps = getattr(response, "steps", None) or []
        calls = [step for step in steps if getattr(step, "type", None) == "function_call"]
        if len(calls) != 1:
            raise LLMProtocolError("Gemini must return exactly one structured function call.")
        call = calls[0]
        name = getattr(call, "name", None)
        arguments = getattr(call, "arguments", None)
        if not isinstance(name, str) or not name:
            raise LLMProtocolError("Gemini returned a function call without a valid name.")
        if name != "next_action":
            raise LLMProtocolError(f"Gemini returned unsupported function {name!r}; expected 'next_action'.")
        if not isinstance(arguments, Mapping):
            raise LLMProtocolError(f"Gemini returned invalid arguments for function {name!r}.")
        return LLMToolDecision(name=name, arguments=dict(arguments), assistant_message=response)

    @staticmethod
    def _tool_declaration(tool_schema: dict[str, Any]) -> dict[str, Any]:
        function = tool_schema.get("function", {})
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError("The next_action tool requires an object parameter schema.")
        return {
            "type": "function",
            "name": function.get("name"),
            "description": function.get("description", ""),
            "parameters": parameters,
        }

    @staticmethod
    def _prepare_messages(messages: Sequence[Mapping[str, Any] | Any]) -> tuple[str, str]:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise ValueError("Gemini messages must be role/content mappings.")
            role = message.get("role", "user")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = _serialize(content)
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                user_parts.append(content)
            else:
                raise ValueError(f"Unsupported Gemini message role {role!r}.")
        if not user_parts:
            raise ValueError("Gemini requests require a user message.")
        return "\n\n".join(system_parts), "\n\n".join(user_parts)

    def _request(
        self,
        messages: Sequence[Mapping[str, Any] | Any],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> tuple[Any, dict[str, Any]]:
        if len(tool_schemas) != 1:
            raise ValueError("Gemini must receive exactly one structured tool.")
        tool = self._tool_declaration(tool_schemas[0])
        if tool["name"] != "next_action":
            raise ValueError("Gemini only accepts the structured next_action tool.")
        system_instruction, user_input = self._prepare_messages(messages)
        request_started = datetime.now(timezone.utc)
        start_time = time.perf_counter()
        metadata: dict[str, Any] = {
            "provider": "gemini",
            "model": self.model,
            "message_count": len(messages),
            "message_characters": len(_serialize(list(messages))),
            "tool_count": 1,
            "tool_schema_characters": len(_serialize(list(tool_schemas))),
            "request_started_at": request_started.isoformat(),
        }
        logger.info("llm_request_started %s", json.dumps(metadata, sort_keys=True))
        try:
            response = self._client.interactions.create(
                model=self.model,
                input=user_input,
                system_instruction=system_instruction or None,
                tools=[tool],
                generation_config={
                    "temperature": 0,
                    "tool_choice": {
                        "allowed_tools": {
                            "mode": "any",
                            "tools": ["next_action"],
                        }
                    },
                },
                store=False,
            )
        except Exception as exc:
            duration = time.perf_counter() - start_time
            failed = {
                **metadata,
                "request_ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(duration, 3),
                "response_success": False,
                "error_type": type(exc).__name__,
                "error": self._safe_error(exc),
            }
            logger.error("llm_request_failed %s", json.dumps(failed, sort_keys=True))
            error = GeminiError(f"Gemini request failed ({type(exc).__name__}): {self._safe_error(exc)}")
            error.diagnostics = failed
            raise error from exc

        calls = [step for step in (getattr(response, "steps", None) or []) if getattr(step, "type", None) == "function_call"]
        names = [getattr(call, "name", "") for call in calls]
        action = None
        if len(calls) == 1 and isinstance(getattr(calls[0], "arguments", None), Mapping):
            action = calls[0].arguments.get("action")
        result_metadata = {
            **metadata,
            "request_ended_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.perf_counter() - start_time, 3),
            "response_success": True,
            "response_status": getattr(response, "status", None) or "completed",
            "finish_reason": getattr(response, "status", None),
            "response_content_characters": len(getattr(response, "output_text", "") or ""),
            "tool_call_count": len(calls),
            "parsed_tool_names": names,
            "selected_action": action,
        }
        logger.info("llm_request_completed %s", json.dumps(result_metadata, sort_keys=True))
        return response, result_metadata

    @staticmethod
    def _safe_error(error: Exception) -> str:
        text = str(error)
        secret = os.environ.get("GEMINI_API_KEY")
        if secret:
            text = text.replace(secret, "[REDACTED]")
        return text[:1000]

    def decide(
        self,
        messages: Sequence[Mapping[str, Any] | Any],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> LLMToolDecision:
        response, _metadata = self._request(messages, tool_schemas)
        return self._parse_function_call(response)

    def diagnose_tool_call(
        self,
        label: str,
        messages: Sequence[Mapping[str, Any] | Any],
        tool_schemas: Sequence[dict[str, Any]],
    ) -> tuple[LLMCallDiagnostics, Any | None, LLMToolDecision | None, str | None]:
        response: Any | None = None
        decision: LLMToolDecision | None = None
        error_text: str | None = None
        try:
            response, metadata = self._request(messages, tool_schemas)
            try:
                decision = self._parse_function_call(response)
            except LLMProtocolError as exc:
                error_text = str(exc)
        except GeminiError as exc:
            metadata = getattr(exc, "diagnostics", {})
            error_text = self._safe_error(exc)
        diagnostics = LLMCallDiagnostics(
            model=self.model,
            message_count=metadata.get("message_count", len(messages)),
            message_characters=metadata.get("message_characters", len(_serialize(list(messages)))),
            tool_count=metadata.get("tool_count", len(tool_schemas)),
            tool_schema_characters=metadata.get("tool_schema_characters", len(_serialize(list(tool_schemas)))),
            request_started_at=metadata.get("request_started_at", ""),
            request_ended_at=metadata.get("request_ended_at", ""),
            duration_seconds=metadata.get("duration_seconds", 0.0),
            http_success=metadata.get("response_success", False),
            finish_reason=metadata.get("finish_reason"),
            response_content_characters=metadata.get("response_content_characters", 0),
            tool_call_count=metadata.get("tool_call_count", 0),
            parsed_tool_names=tuple(metadata.get("parsed_tool_names", [])),
            error_type=metadata.get("error_type"),
            error=error_text or metadata.get("error"),
            provider="gemini",
        )
        return diagnostics, response, decision, error_text