"""Output cleanup before an edit re-enters the loop, and wrapper detection."""

from __future__ import annotations

import re

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

# Chat-assistant boilerplate seen in ~30% of outputs before the format
# instruction was added (smoke test, 12 Jul 2026).
WRAPPER_PATTERN = re.compile(
    r"^(here'?s|here is|sure[,!]|certainly[,!])\b"
    r"|if you'?d like.{0,60}(also|further)"
    r"|^\*\*",
    re.IGNORECASE,
)


def strip_reasoning(text: str) -> str:
    """Drop one leading inline `<think>...</think>` block."""
    return _THINK_BLOCK.sub("", text, count=1)


def looks_wrapped(text: str) -> bool:
    """True if `text` contains chat-wrapper boilerplate."""
    return bool(WRAPPER_PATTERN.search(text))
