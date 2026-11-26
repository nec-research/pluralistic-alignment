# Pluralistic Alignment

A toolkit for studying how language varieties are handled in LLM pipelines.

## Project Structure

```
src/pluralistic_alignment/
├── cli.py                  # CLI entry point (`pa` command)
├── schema.py               # Data model (ChainKey, VariantKey, Marker, etc.)
├── settings.py             # Environment config (.env loading)
├── paths.py                # Project path constants
├── data/
│   ├── wikipedia.py        # Seed sampling from structured Wikipedia
│   ├── variants.py         # Dialect variant loading/parsing
│   └── selection.py        # Seed filtering by marker count
├── editing/
│   ├── clients.py          # Azure OpenAI + vLLM client registry
│   ├── prompts.py          # Rewriting prompt templates
│   ├── arms.py             # Experimental arm protocols
│   ├── loop.py             # Main rewriting loop driver
│   ├── cache.py            # Resumable JSONL cache
│   └── postprocess.py      # Output cleanup
├── metrics/
│   ├── variety/markers.py  # Feature retention metric
│   ├── lexical.py          # MATTR/MTLD diversity
│   ├── semantic.py         # Embedding similarity
│   └── diversity.py        # Cross-variety convergence
└── analysis/
    ├── chains.py           # Build editing chains from cache
    ├── stats.py            # Bootstrap CIs, sign tests
    └── report.py           # Report generation
envs/multivalue/            # Isolated env for Multi-VALUE dialect injection
configs/models.yaml         # Which models to run, and where they are served
data/wiki/                  # Committed seeds + variants (130 Wikipedia items)
docs/evaluations/           # One markdown report per evaluation run
scripts/                    # One-off analysis helpers
```

## Quick Start

**Requirements:** Python 3.12, [uv](https://github.com/astral-sh/uv) 0.11.16+

### 1. Clone and set up

```bash
git clone <repo-url> && cd pluralistic-alignment

# Copy and fill in your API credentials
cp .env.example .env
# Edit .env: Azure OpenAI key/endpoint/version (only needed for the Azure models)
# and VLLM_API_KEY (any non-empty value for unauthenticated vLLM servers).

# Point the pipeline at YOUR model deployments (required)
cp configs/models.yaml configs/models.local.yaml   # gitignored, read instead of models.yaml
# Edit configs/models.local.yaml:
#   azure_models: your Azure OpenAI deployment names
#   vllm_models:  name = your server's served model name, base_url = its URL
#                 (the defaults assume vLLM on localhost:8010/8011/8014)
#   delete any model you don't have; `pa edit` calls every model listed
```

See [Configuration](#configuration) for details.

### 2. Install dependencies

```bash
UV_MALWARE_CHECK=1 uv sync --dev
```

If you hit a Python version mismatch:

```bash
UV_MALWARE_CHECK=1 uv sync --dev --python $(cat .python-version)
```

### 3. Run checks

```bash
make check    # full gate: lint, typecheck, tests, license check, audit
make test     # tests only
make lint     # lint + format only
```

### 4. Pipeline

The full pipeline has four steps. Each step reads the previous step's file, so
pass the same paths throughout. The example below is a small end-to-end run
(2 items, 5 varieties + SAE control, 2 models, both arms) written to its own
run directory so the committed `data/wiki/` files are left untouched:

```bash
RUN=data/wiki/runs/my_run        # gitignored; every output of this run goes here
mkdir -p $RUN

# Step 1: Sample seed paragraphs from English Wikipedia (streams
# wikimedia/structured-wikipedia from the Hugging Face Hub; needs internet).
uv run pa seeds --n 2 --out $RUN/seeds.jsonl
# Add more later with: uv run pa seeds --n 1 --append --out $RUN/seeds.jsonl
# Dataset revision and `datasets` version are pinned, so the default --seed 42
# reproduces the order of the committed data/wiki/seeds.jsonl.

# Step 2: Inject Multi-VALUE dialect features (separate env, see below).
# Writes one record per (seed, variety): 5 varieties + the unchanged SAE text.
cd envs/multivalue
uv sync
uv run python transform_seeds.py \
    --seeds-in ../../$RUN/seeds.jsonl --out ../../$RUN/variants.jsonl \
    --gpus 0          # optional: GPU id(s) for stanza coref; omit for CPU
cd ../..

# Step 3: Iterated rewriting (t = 1..5). Only items where every variety has
# >= 3 trackable markers are used (--min-markers); --n-items takes the first N
# qualifying items. If fewer qualify, a WARNING is printed: sample more seeds
# (step 1 with --append, preferably in batches, e.g. --n 10, since each call
# re-streams Wikipedia) and re-run step 2 (it resumes).
#   --arms   A0 = neutral edit prompt, A1 = variety-preserving (steering) prompt
#   --models subset of your models config (default: every listed model)
uv run pa edit --variants $RUN/variants.jsonl --n-items 2 --arms A0,A1 \
    --models gpt-5.4-mini,gemma-3-4b-it --out $RUN/rewrites.jsonl

# Step 4: Metrics report: feature retention + embedding similarity.
# Downloads Qwen/Qwen3-Embedding-0.6B (~1.2 GB) on first use; uses a GPU if
# visible (set CUDA_VISIBLE_DEVICES to choose one).
uv run pa report --seeds $RUN/seeds.jsonl --variants $RUN/variants.jsonl \
    --rewrites $RUN/rewrites.jsonl --results-dir results/my_run
```

`pa edit` and `transform_seeds.py` cache every finished record, so an
interrupted run resumes where it stopped when re-run with the same arguments.
To run several models in parallel, start one `pa edit` per model with its own
`--out` file and pass all of them to `pa report --rewrites a.jsonl b.jsonl ...`.
Do not point two processes at the same `--out` file.

Outputs of `pa report` (in `--results-dir`):

| File | Contents |
|---|---|
| `results.csv` | All metrics per (variety, model, arm, t) with bootstrap 95% CIs: `retention` / `retention_surface` / `retention_deletion` (dialect-feature retention), `semantic_preservation` (cosine to the original seed), `convergence` (pairwise cross-variety cosine), `convergence_to_sae`, `lexical_convergence`, MATTR/MTLD |
| `compounding_stat.csv`, `sign_test.json` | Does retention keep falling over rounds? |
| `audit_sample.jsonl` | 30 sampled marker-loss events for manual labelling |
| `results_table.tex`, `results.pdf` | Retention table and figure |

Defaults if you omit the flags: `pa seeds` writes 50 seeds to
`data/items/wikipedia_seeds.jsonl`; `transform_seeds.py` reads
`data/wiki/seeds.jsonl` and appends to `data/wiki/variants.jsonl` (the
committed files); `pa edit` uses `data/wiki/variants.jsonl`, 30 items, arm `A0`
and all models; `pa report` writes to `results/report/`.

**Note:** Step 2 uses a separate Python environment (`envs/multivalue/`) because
Multi-VALUE requires older `numpy`/`torch` versions. See
[envs/multivalue/README.md](envs/multivalue/README.md) for details and GPU setup.
On first run it downloads the stanza English models (~1 GB, to
`~/stanza_resources`, or `$STANZA_RESOURCES_DIR`) and the NLTK `cmudict` /
`wordnet` corpora (to `~/nltk_data`). Some paragraphs fail with
`AssertionError: Spacy and Stanza word mismatch` (a known Multi-VALUE
tokenizer issue, e.g. with abbreviations): this is logged as
`FAILED <seed>/<variety>` and recorded with `text: null`. The run continues,
and `pa edit` skips such items automatically.

Pre-computed seed data is included in `data/wiki/` (130 seeds + their
variants), so you can skip steps 1-2 and start from step 3 with
`--variants data/wiki/variants.jsonl` (and `--seeds data/wiki/seeds.jsonl` for
the report).

**Performance tip:** keep the checkout, `.venv` and the uv / Hugging Face
caches on a local disk. On an NFS home directory, `import openai` alone can
take minutes. For example:
`export UV_CACHE_DIR=/local/disk/uv-cache HF_HOME=/local/disk/hf`.

## Configuration

- **`.env`**: credentials (Azure OpenAI key/endpoint/version, `VLLM_API_KEY`). See `.env.example`.
- **`configs/models.yaml`**: model deployment registry. `azure_models` are
  deployment names on your Azure OpenAI resource; `vllm_models` give each
  self-hosted model's served name and `base_url` (defaults: `localhost`
  ports). `fixed_temperature_models` lists deployments that reject a custom
  temperature. **Don't edit it for your own servers.** Copy it to
  `configs/models.local.yaml` (gitignored) and edit the copy: `pa` reads
  `$PA_MODELS_CONFIG` if set, else `configs/models.local.yaml` if it exists,
  else `configs/models.yaml`. Remove models you cannot reach, or pick a subset
  with `pa edit --models`.

To serve an open model yourself, any OpenAI-compatible server works, e.g.
`vllm serve google/gemma-3-4b-it --served-model-name gemma-3-4b-it --port 8010`.

## Extending

To add a new rewriting model:
1. Add it to `configs/models.local.yaml` (your setup) or `configs/models.yaml` (shared default)
2. Ensure it's reachable via OpenAI-compatible API (Azure or vLLM)

To add a new metric:
1. Create a module under `src/pluralistic_alignment/metrics/`
2. Integrate it in `analysis/report.py`

To add a new data source:
1. Create a loader under `src/pluralistic_alignment/data/`
2. Register it in the CLI (`cli.py`)

## Development

Uses [Conventional Commits](https://www.conventionalcommits.org). Run `make check` before pushing.

| Tool | Purpose |
|---|---|
| **uv** | Environment and dependency management |
| **Ruff** | Linting and formatting |
| **basedpyright** | Type checking |
| **pytest** | Testing |
| **pre-commit** | Git hooks (lint, format, conventional commits) |
