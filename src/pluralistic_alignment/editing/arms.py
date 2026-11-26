"""Editing arms: how one round turns the current text into the next.

An arm gets a `call(user_message) -> CallResult` function, the round input
(t-1) and the chain's original (t=0) text, and returns the round output. Arms
that need more than one call (critique-then-restore, recheck) implement the
same protocol, so the loop never changes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pluralistic_alignment.editing.clients import CallResult
from pluralistic_alignment.editing.prompts import (
    SINGLE_CALL_PROMPTS,
    build_user_message,
)

Call = Callable[[str], CallResult]


@dataclass(frozen=True)
class ArmStep:
    """Round output, summed token usage, and arm-specific fields to cache."""

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None
    extra: dict[str, Any] = field(default_factory=dict)


class Arm(Protocol):
    """One editing method."""

    @property
    def name(self) -> str:
        """Arm id, e.g. `A0`."""
        ...

    def step(self, call: Call, current_text: str, original_text: str) -> ArmStep:
        """Edit `current_text` once."""
        ...


@dataclass(frozen=True)
class SingleCallArm:
    """Instruction + current text, one call (A0 neutral, A1 steering)."""

    name: str
    prompt: str

    def step(self, call: Call, current_text: str, original_text: str) -> ArmStep:
        """One call; `original_text` is unused."""
        del original_text
        result = call(build_user_message(self.prompt, current_text))
        return ArmStep(result.text, result.prompt_tokens, result.completion_tokens)


def build_arm(name: str) -> Arm:
    """Arm by name (`A0`, `A1`)."""
    if name in SINGLE_CALL_PROMPTS:
        return SingleCallArm(name, SINGLE_CALL_PROMPTS[name])
    raise ValueError(f"unknown arm {name!r}; known: {sorted(SINGLE_CALL_PROMPTS)}")
