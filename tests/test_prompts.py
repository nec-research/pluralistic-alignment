"""Tests for prompt construction."""

from pluralistic_alignment.editing.prompts import (
    NEUTRAL_PROMPT,
    SINGLE_CALL_PROMPTS,
    STEERING_PROMPT,
    build_user_message,
)


def test_neutral_prompt_contains_improve() -> None:
    """Neutral prompt asks to improve text."""
    assert "improve this text" in NEUTRAL_PROMPT.lower()


def test_steering_prompt_contains_preserve() -> None:
    """Steering prompt asks to preserve dialect."""
    assert "preserve" in STEERING_PROMPT.lower()
    assert "dialect" in STEERING_PROMPT.lower()


def test_both_prompts_have_format_instruction() -> None:
    """Both prompts end with the format instruction."""
    for prompt in (NEUTRAL_PROMPT, STEERING_PROMPT):
        assert "Return only the revised text" in prompt


def test_single_call_prompts_mapping() -> None:
    """A0 maps to neutral, A1 to steering."""
    assert SINGLE_CALL_PROMPTS["A0"] is NEUTRAL_PROMPT
    assert SINGLE_CALL_PROMPTS["A1"] is STEERING_PROMPT


def test_build_user_message_format() -> None:
    """User message is prompt + blank line + text."""
    msg = build_user_message("Do something.", "Hello world.")
    assert msg == "Do something.\n\nHello world."
