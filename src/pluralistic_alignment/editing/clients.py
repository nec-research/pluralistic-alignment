"""Model registry (Azure OpenAI + vLLM) and a single retrying chat call.

Adding an editor model is an entry in `configs/models.yaml`, not new code.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import APIError, AzureOpenAI, OpenAI, omit
from openai.types.chat import ChatCompletionMessageParam

from pluralistic_alignment.editing.postprocess import strip_reasoning
from pluralistic_alignment.settings import (
    AzureOpenAIConfig,
    ModelsConfig,
    VLLMConfig,
    load_azure_openai_config,
    load_vllm_config,
)


@dataclass(frozen=True)
class CallSettings:
    """Decoding and retry settings for every call in a run."""

    temperature: float = 0.7
    max_retries: int = 3
    sleep_between: float = 0.0


@dataclass(frozen=True)
class CallResult:
    """A cleaned completion plus token usage."""

    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class EditorModel:
    """One configured editor: client, deployment name, decoding capability."""

    name: str
    backend: str
    client: Any
    deployment: str
    supports_custom_temperature: bool = True


def build_models(
    config: ModelsConfig,
    azure_loader: Callable[[], AzureOpenAIConfig] = load_azure_openai_config,
    vllm_loader: Callable[[], VLLMConfig] = load_vllm_config,
) -> list[EditorModel]:
    """One `EditorModel` per deployment listed in `config`."""
    models: list[EditorModel] = []
    fixed = set(config.fixed_temperature_models)
    if config.azure_models:
        azure = azure_loader()
        client = AzureOpenAI(
            api_key=azure.api_key,
            api_version=azure.api_version,
            azure_endpoint=azure.azure_endpoint,
        )
        models.extend(
            EditorModel(name, "azure", client, name, name not in fixed)
            for name in config.azure_models
        )
    if config.vllm_models:
        vllm = vllm_loader()
        clients: dict[str, OpenAI] = {}
        for vm in config.vllm_models:
            client = clients.setdefault(
                vm.base_url, OpenAI(api_key=vllm.api_key, base_url=vm.base_url)
            )
            models.append(
                EditorModel(vm.name, "vllm", client, vm.name, vm.name not in fixed)
            )
    return models


def call_model(
    model: EditorModel,
    user_message: str,
    settings: CallSettings,
    sleep: Callable[[float], None] = time.sleep,
) -> CallResult:
    """One chat call with exponential-backoff retry on API errors."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": user_message}
    ]
    # Some deployments (gpt-5.6-luna) reject any explicit temperature.
    temperature = settings.temperature if model.supports_custom_temperature else omit
    last_exc: Exception | None = None
    for attempt in range(settings.max_retries):
        try:
            response = model.client.chat.completions.create(
                model=model.deployment, messages=messages, temperature=temperature
            )
        except APIError as exc:
            last_exc = exc
            backoff = 2**attempt
            print(
                f"  {model.name} APIError (attempt {attempt + 1}/"
                f"{settings.max_retries}): {exc} -- retrying in {backoff}s"
            )
            sleep(backoff)
            continue
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError(f"{model.name} returned an empty completion")
        text = strip_reasoning(content).strip()
        if not text:
            raise RuntimeError(
                f"{model.name} returned only a <think> block (likely truncated)"
            )
        usage = response.usage
        return CallResult(
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
    raise RuntimeError(
        f"{model.name} failed after {settings.max_retries} attempts"
    ) from last_exc
