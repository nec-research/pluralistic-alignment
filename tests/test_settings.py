"""Tests for model config resolution."""

from pathlib import Path

import pytest

from pluralistic_alignment.settings import (
    MODELS_CONFIG_ENV,
    MODELS_CONFIG_PATH,
    load_models_config,
    models_config_path,
)


def test_models_config_path_prefers_env_then_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """$PA_MODELS_CONFIG wins, then models.local.yaml, then the committed default."""
    default, local = tmp_path / "models.yaml", tmp_path / "models.local.yaml"
    monkeypatch.delenv(MODELS_CONFIG_ENV, raising=False)
    assert models_config_path(default, local) == default
    local.write_text("{}")
    assert models_config_path(default, local) == local
    monkeypatch.setenv(MODELS_CONFIG_ENV, str(tmp_path / "other.yaml"))
    assert models_config_path(default, local) == tmp_path / "other.yaml"


def test_committed_models_config_is_generic() -> None:
    """The committed config only points at localhost, never at internal hosts."""
    config = load_models_config(MODELS_CONFIG_PATH)
    urls = [m.base_url for m in config.vllm_models]
    assert urls
    assert all(url.startswith("http://localhost:") for url in urls)
