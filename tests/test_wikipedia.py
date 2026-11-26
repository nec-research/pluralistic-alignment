"""Tests for the pinned Wikipedia seed sampler."""

from typing import Any

import datasets
import pytest

from pluralistic_alignment.data import wikipedia

GIT_SHA_LENGTH = 40


def test_stream_is_pinned_to_dataset_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stream is loaded at the pinned revision and shuffled with the given seed."""
    calls: dict[str, Any] = {}

    class FakeStream:
        def shuffle(self, seed: int, buffer_size: int) -> str:
            calls["shuffle"] = (seed, buffer_size)
            return "shuffled"

    def fake_load_dataset(*args: Any, **kwargs: Any) -> FakeStream:
        calls["args"], calls["kwargs"] = args, kwargs
        return FakeStream()

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    assert wikipedia.stream_wikipedia(seed=42, shuffle_buffer_size=10) == "shuffled"
    assert calls["kwargs"]["revision"] == wikipedia.DATASET_REVISION
    assert len(wikipedia.DATASET_REVISION) == GIT_SHA_LENGTH
    assert calls["shuffle"] == (42, 10)


def test_seed_record_carries_revision() -> None:
    """Each seed record stores the dataset revision it was sampled from."""
    record = wikipedia._seed_record({"name": "T", "abstract": "a b"}, "wiki-000", "now")
    assert record["source_revision"] == wikipedia.DATASET_REVISION
