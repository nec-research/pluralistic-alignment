"""Runtime configuration: credentials from `.env` and model deployments from YAML.

The only module that reads secrets. Never print or log values loaded here; to
confirm a key is present, check `bool(value)`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import dotenv_values

from pluralistic_alignment.paths import CONFIGS_DIR, REPO_ROOT

ENV_PATH = REPO_ROOT / ".env"
MODELS_CONFIG_PATH = CONFIGS_DIR / "models.yaml"
LOCAL_MODELS_CONFIG_PATH = CONFIGS_DIR / "models.local.yaml"  # gitignored
MODELS_CONFIG_ENV = "PA_MODELS_CONFIG"


class MissingSettingError(RuntimeError):
    """Raised when a required environment variable is not set."""


@dataclass(frozen=True)
class AzureOpenAIConfig:
    """Azure OpenAI resource credentials."""

    api_key: str
    api_version: str
    azure_endpoint: str


@dataclass(frozen=True)
class VLLMConfig:
    """Shared credential for the self-hosted vLLM endpoints."""

    api_key: str


@dataclass(frozen=True)
class VLLMModelConfig:
    """One vLLM deployment: served model name and endpoint (not a secret)."""

    name: str
    base_url: str


@dataclass(frozen=True)
class ModelsConfig:
    """Which deployments to run, and per-deployment decoding quirks."""

    azure_models: tuple[str, ...]
    vllm_models: tuple[VLLMModelConfig, ...]
    fixed_temperature_models: tuple[str, ...]


def _require(name: str, env_path: Path) -> str:
    """Environment variable `name`; the process environment wins over `.env`."""
    value = os.environ.get(name) or dotenv_values(env_path).get(name)
    if not value:
        raise MissingSettingError(
            f"{name} is not set; copy .env.example to .env and fill it in"
        )
    return value


def load_azure_openai_config(env_path: Path = ENV_PATH) -> AzureOpenAIConfig:
    """Azure OpenAI credentials from AZURE_OPENAI_* variables."""
    return AzureOpenAIConfig(
        api_key=_require("AZURE_OPENAI_API_KEY", env_path),
        api_version=_require("AZURE_OPENAI_API_VERSION", env_path),
        azure_endpoint=_require("AZURE_OPENAI_ENDPOINT", env_path),
    )


def load_vllm_config(env_path: Path = ENV_PATH) -> VLLMConfig:
    """vLLM credential from VLLM_API_KEY."""
    return VLLMConfig(api_key=_require("VLLM_API_KEY", env_path))


def models_config_path(
    default: Path = MODELS_CONFIG_PATH, local: Path = LOCAL_MODELS_CONFIG_PATH
) -> Path:
    """$PA_MODELS_CONFIG, else `configs/models.local.yaml` if present, else default."""
    if override := os.environ.get(MODELS_CONFIG_ENV):
        return Path(override)
    return local if local.exists() else default


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Read the models config (see `models_config_path` for which file)."""
    path = path or models_config_path()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = yaml.safe_load(path.read_text()) or {}
    return ModelsConfig(
        azure_models=tuple(raw.get("azure_models") or ()),
        vllm_models=tuple(
            VLLMModelConfig(name=m["name"], base_url=m["base_url"])
            for m in (raw.get("vllm_models") or ())
        ),
        fixed_temperature_models=tuple(raw.get("fixed_temperature_models") or ()),
    )
