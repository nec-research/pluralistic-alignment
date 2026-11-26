"""Tests for editing output cleanup."""

from pluralistic_alignment.editing.postprocess import looks_wrapped, strip_reasoning


def test_strip_reasoning_removes_think_block() -> None:
    """Leading <think>...</think> is stripped."""
    text = "<think>internal reasoning here</think>The actual output."
    assert strip_reasoning(text) == "The actual output."


def test_strip_reasoning_preserves_text_without_think() -> None:
    """Text without think block passes through."""
    text = "Just normal output."
    assert strip_reasoning(text) == "Just normal output."


def test_strip_reasoning_only_first_block() -> None:
    """Only the first think block is removed."""
    text = "<think>first</think>middle<think>second</think>end"
    assert strip_reasoning(text) == "middle<think>second</think>end"


def test_strip_reasoning_multiline_think() -> None:
    """Multi-line think blocks are stripped."""
    text = "<think>\nline1\nline2\n</think>Output here."
    assert strip_reasoning(text) == "Output here."


def test_looks_wrapped_heres() -> None:
    """'Here's' prefix is detected as wrapper."""
    assert looks_wrapped("Here's the improved version of your text...")


def test_looks_wrapped_certainly_no_space() -> None:
    """'certainly!' directly followed by word char is detected."""
    assert looks_wrapped("certainly!Here is the revised text.")


def test_looks_wrapped_sure_no_space() -> None:
    """'sure!' directly followed by word char is detected."""
    assert looks_wrapped("sure!Here you go.")


def test_looks_wrapped_certainly_with_space_not_matched() -> None:
    """'Certainly, ' (space after punctuation) does not trigger the wrapper pattern."""
    assert not looks_wrapped("Certainly, here is the revised text.")


def test_looks_wrapped_bold_markdown() -> None:
    """Leading ** markdown is detected."""
    assert looks_wrapped("**Improved Text**\nThe actual content.")


def test_looks_wrapped_clean_text() -> None:
    """Normal text is not flagged."""
    assert not looks_wrapped("The quick brown fox jumps over the lazy dog.")


def test_looks_wrapped_if_youd_like() -> None:
    """'if you'd like...also/further' pattern is detected."""
    assert looks_wrapped("Let me know if you'd like me to also adjust the tone.")
