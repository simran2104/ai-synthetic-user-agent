"""Load synthetic-user personas from the agent configuration file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PERSONAS_PATH = Path(__file__).resolve().parent.parent / "agent_config" / "personas.json"
VALID_STATE_CHANGING_ACTIONS = frozenset({"cart", "enquiry"})


class PersonaConfigurationError(ValueError):
    """Raised when the persona configuration is missing or malformed."""


@dataclass(frozen=True)
class Persona:
    name: str
    description: str
    goal: str
    allowed_state_changing_actions: frozenset[str]


def load_personas(path: Path = PERSONAS_PATH) -> dict[str, Persona]:
    """Load persona descriptions and state-change permissions from JSON."""
    try:
        with path.open(encoding="utf-8") as config_file:
            data: Any = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise PersonaConfigurationError(f"Could not load personas from {path}: {exc}") from exc

    entries = data.get("personas") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        raise PersonaConfigurationError("Persona configuration must contain a 'personas' object.")

    personas: dict[str, Persona] = {}
    for name, entry in entries.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise PersonaConfigurationError("Each persona must have a name and an object definition.")
        description = entry.get("description")
        goal = entry.get("goal", description)
        actions = entry.get("allowed_state_changing_actions", [])
        if not isinstance(description, str) or not description.strip():
            raise PersonaConfigurationError(f"Persona {name!r} requires a non-empty description.")
        if not isinstance(goal, str) or not goal.strip():
            raise PersonaConfigurationError(f"Persona {name!r} requires a non-empty goal.")
        if not isinstance(actions, list) or any(action not in VALID_STATE_CHANGING_ACTIONS for action in actions):
            raise PersonaConfigurationError(
                f"Persona {name!r} has invalid state-changing action permissions."
            )
        personas[name] = Persona(name, description, goal, frozenset(actions))
    return personas