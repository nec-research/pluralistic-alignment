# envs/multivalue/

A second, isolated `uv` environment used only to run Multi-VALUE
(`value-nlp`), which turns SAE (Standard American English) seed text into
5 English varieties + records which eWAVE features fired. Kept separate
from `../pyproject.toml` (the main pipeline env) because Multi-VALUE's
dependency chain (spaCy 3.7.4 -> thinc<8.3, stanza) forces version
constraints that the main env should not inherit:

| Constraint | Reason |
|---|---|
| `numpy<2` | `spacy==3.7.4` pulls `thinc<8.3`, whose compiled wheel is built against the numpy-1.x C ABI; numpy 2.x breaks it at import (`ValueError: numpy.dtype size changed`). |
| `torch<2.6` | stanza (Multi-VALUE's coref dependency) loads old pickled `.pt` model files; torch>=2.6 changed `torch.load`'s default to `weights_only=True`, which can't unpickle them. |
| `transformers<4.51` | See "Known security trade-off" below. |

## Known security trade-off (read before changing pins)

`transformers>=4.51` added a hard check (`check_torch_load_is_safe`,
mitigating CVE-2025-32434) that refuses to run `torch.load` at all unless
torch>=2.6 -- which directly conflicts with the `torch<2.6` pin above
(stanza's old pickle-format models can't load under torch>=2.6 either).
Pinning `transformers<4.51` here means this environment uses `torch.load`
in its old, less-safe mode (arbitrary code execution is possible if it
ever loads an untrusted pickle).

This is judged acceptable **only** because this environment loads exactly
two model artifacts, both from their official, trusted sources:
`en_core_web_sm` (spaCy's official model release) and stanza's official
English pretrained pipeline (tokenize/mwt/pos/lemma/depparse/coref,
including the BERT-based coref model). **Do not** use this environment to
load any other checkpoint, especially anything downloaded from an
untrusted or third-party source.

If this trade-off becomes unacceptable (e.g. Multi-VALUE or stanza ships
safetensors-format models), re-check whether the
constraints above can be relaxed.

## GPU usage

This environment uses default (GPU-enabled) torch wheels, which pull in
NVIDIA's proprietary CUDA runtime as separate pip packages
(`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, etc. -- licensed "NVIDIA
Proprietary Software" / "Other/Proprietary License"). **2026-07-12
decision**: allowed project-wide. "Commercial-compatible" was clarified to
mean "no legal blocker to commercial/research use," not "FOSS-only" --
NVIDIA's CUDA runtime EULA permits that use, and there is no way to get
GPU acceleration without depending on this proprietary runtime somewhere
(a different torch packaging just hides it from a dependency scanner
rather than removing it). See `[tool.pylic]` in this directory's
`pyproject.toml` for the by-name allowlist.

stanza's coref processor (a BERT model) is the bottleneck in
`transform_seeds.py` and is far faster on GPU than CPU per document.
Use `--gpus 0,1,2,3` (round-robins workers across those GPU ids) or omit
`--gpus` for CPU-only.

## Usage

```bash
cd envs/multivalue
uv sync          # also installs spaCy's en_core_web_sm and pip (see below)
uv run python transform_seeds.py --seeds-in ../../data/wiki/runs/my_run/seeds.jsonl \
    --out ../../data/wiki/runs/my_run/variants.jsonl --workers 4 --gpus 0,1,2,3
# defaults: reads ../../data/wiki/seeds.jsonl, appends to ../../data/wiki/variants.jsonl
```

`en_core_web_sm` and `pip` are declared as dependencies because
`multivalue/__init__.py` calls `spacy.cli.download("en_core_web_sm")` on every
import, which shells out to `pip`. uv-created venvs ship without pip, so
without them `import multivalue` fails with `No module named pip`.

The main pipeline (`../src/pluralistic_alignment/...`) never imports anything
from this environment directly -- it only reads the `data/variants.jsonl`
file this one produces. Nothing here is on the main `uv.lock`.
