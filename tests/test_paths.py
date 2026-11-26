"""Tests for path utilities and private-data guard."""

import pytest

from pluralistic_alignment.paths import (
    DATA_DIR,
    PRIVATE_BLOCKS,
    PrivateDataError,
    check_output_path,
    is_private_path,
)


def test_is_private_path_under_data_private() -> None:
    """Path under data/private/ is private."""
    assert is_private_path(DATA_DIR / "private" / "flores" / "out.jsonl")


def test_is_private_path_normal_data() -> None:
    """Path under data/wiki/ is not private."""
    assert not is_private_path(DATA_DIR / "wiki" / "seeds.jsonl")


def test_check_output_path_raises_for_gated_block() -> None:
    """Writing a gated block outside private/ raises."""
    with pytest.raises(PrivateDataError):
        check_output_path(DATA_DIR / "public" / "out.jsonl", ["flores"])


def test_check_output_path_allows_private_dir() -> None:
    """Writing a gated block under private/ is allowed."""
    check_output_path(DATA_DIR / "private" / "out.jsonl", ["flores"])


def test_check_output_path_allows_non_gated() -> None:
    """Non-gated blocks can be written anywhere."""
    check_output_path(DATA_DIR / "wiki" / "out.jsonl", ["wikipedia"])


def test_flores_is_private() -> None:
    """FLORES is in the private blocks set."""
    assert "flores" in PRIVATE_BLOCKS
