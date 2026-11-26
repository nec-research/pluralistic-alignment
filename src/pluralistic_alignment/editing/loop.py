"""Iterated editing: output of round t-1 is the input of round t, t = 1..max_t.

Every (chain, round) is cached; a resumed run replays cached rounds and only
calls the model for missing ones. Arms are never mixed within a chain.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from pluralistic_alignment.editing.arms import Arm
from pluralistic_alignment.editing.cache import append_record, load_done
from pluralistic_alignment.editing.clients import (
    CallResult,
    CallSettings,
    EditorModel,
    call_model,
)
from pluralistic_alignment.paths import check_output_path
from pluralistic_alignment.schema import MAX_T, CacheKey, ChainKey, VariantKey

CallFn = Callable[[EditorModel, str, CallSettings], CallResult]


@dataclass(frozen=True)
class RunPlan:
    """What to run: round-0 texts x models x arms, up to `max_t`."""

    chains: Mapping[VariantKey, str]
    models: Sequence[EditorModel]
    arms: Sequence[Arm]
    max_t: int = MAX_T


@dataclass(frozen=True)
class _RunState:
    done: dict[CacheKey, str]
    out: IO[str]
    settings: CallSettings
    max_t: int


def _record(key: ChainKey, t: int, **fields: Any) -> dict[str, Any]:
    return {
        "block": key.block,
        "item_id": key.item_id,
        "variety": key.variety,
        "model": key.model,
        "arm": key.arm,
        "t": t,
        **fields,
    }


def _run_chain(
    key: ChainKey,
    t0_text: str,
    call: Callable[[str], CallResult],
    arm: Arm,
    state: _RunState,
) -> None:
    current = t0_text
    for t in range(1, state.max_t + 1):
        if key.at(t) in state.done:
            current = state.done[key.at(t)]
            continue
        try:
            step = arm.step(call, current, t0_text)
        except RuntimeError as exc:
            print(f"FAILED {key}/t={t}: {exc}")
            append_record(
                state.out,
                _record(
                    key,
                    t,
                    text=None,
                    prompt_tokens=None,
                    completion_tokens=None,
                    error=str(exc),
                ),
            )
            return  # a chain cannot continue past a failed round
        append_record(
            state.out,
            _record(
                key,
                t,
                text=step.text,
                prompt_tokens=step.prompt_tokens,
                completion_tokens=step.completion_tokens,
                error=None,
                **step.extra,
            ),
        )
        current = step.text
        if state.settings.sleep_between:
            time.sleep(state.settings.sleep_between)


def run(
    plan: RunPlan,
    out_path: Path,
    settings: CallSettings,
    call_fn: CallFn = call_model,
) -> None:
    """Run every chain in `plan`, appending new rounds to `out_path`."""
    check_output_path(out_path, {block for block, _, _ in plan.chains})
    done = load_done(out_path)
    print(f"{len(plan.chains)} chains loaded; {len(done)} rounds already cached")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as out:
        state = _RunState(done, out, settings, plan.max_t)
        for (block, item_id, variety), t0_text in plan.chains.items():
            for model in plan.models:

                def call(message: str, model: EditorModel = model) -> CallResult:
                    return call_fn(model, message, settings)

                for arm in plan.arms:
                    key = ChainKey(block, item_id, variety, model.name, arm.name)
                    _run_chain(key, t0_text, call, arm, state)
