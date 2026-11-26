"""Editing instructions per arm. Frozen: do not edit wording.

A0 and A1 are the exact strings the original experiment sent (ASCII ` -- `, not the em
dash printed in the paper appendix); changing a character breaks
comparability with existing runs.
"""

from __future__ import annotations

_FORMAT_INSTRUCTION = (
    " Return only the revised text itself -- no preamble, explanation, "
    "commentary, or markdown formatting."
)

NEUTRAL_PROMPT = "Please improve this text." + _FORMAT_INSTRUCTION
STEERING_PROMPT = (
    "Please improve this text for clarity and flow, but do not standardize "
    "it -- preserve the author's dialect, regional English features, and "
    "any non-standard grammar, vocabulary, or phrasing exactly as they "
    "appear."
) + _FORMAT_INSTRUCTION

SINGLE_CALL_PROMPTS = {
    "A0": NEUTRAL_PROMPT,
    "A1": STEERING_PROMPT,
}


def build_user_message(prompt: str, text: str) -> str:
    """One user message: instruction, blank line, current text (standard format)."""
    return f"{prompt}\n\n{text}"
