# data/wiki: Wikipedia seeds and Multi-VALUE variants

| File | Contents |
|---|---|
| `seeds.jsonl` | 130 English Wikipedia abstracts (30–120 words, prose only), `wiki-000` … `wiki-129` |
| `variants.jsonl` | For each seed: the unchanged SAE text plus 5 Multi-VALUE varieties (Indian, Nigerian, Colloquial Singapore, African American Vernacular, Australian English), with the fired eWAVE rules in `executed_rules` |

## Source and version

| | |
|---|---|
| Dataset | [`wikimedia/structured-wikipedia`](https://huggingface.co/datasets/wikimedia/structured-wikipedia), config `enwiki_namespace_0`, split `train` (streamed) |
| Hub revision | `417c267bb457fa645c22eb3b5c77764963194c70` (`DATASET_REVISION` in `src/pluralistic_alignment/data/wikipedia.py`) |
| Sampled | 2026-07-12 (`retrieved_at` in each record) |
| Library | `datasets` 5.0.0 at sampling time; pinned to 5.0.1, which gives the identical order |
| Sampler | `uv run pa seeds`: `--seed 42`, shuffle buffer 10,000, 30–120 words, list/disambiguation/markup pages excluded |
| Variants | `envs/multivalue/transform_seeds.py --seed 42` (`value-nlp` 0.3); see the note below on reproducing them |

The shuffled stream order depends on **both** the Hub revision and the
`datasets` version: with `datasets` 4.8.5 or 5.2.0 the same seed gives
different articles. Both are pinned, so `pa seeds --n N` reproduces the first N
records of `seeds.jsonl` (checked on 2026-11-25: repeated runs and
`--append` reproduced the committed seeds exactly).

**Reproducing `variants.jsonl`.** Multi-VALUE's output for a paragraph depends on
its position in the run, not only on its text. Each dialect is seeded once and
its random generator advances with every paragraph it processes, so the
`--workers` count and the order of `seeds.jsonl` matter. The first paragraph
of a run (or of a worker's shard) is reproduced exactly: `wiki-000`, and
`wiki-001` when transformed on its own, match this file byte for byte.
Treat the committed `variants.jsonl` as the canonical July 2026 data.

## License

The text is from English Wikipedia and is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). `variants.jsonl`
is an adaptation of it, under the same license. Each record links its source
article (`url`). This license applies to the files in this directory, not to
the code in this repository (see `LICENSE`).
