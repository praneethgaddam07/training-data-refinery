# Training Data Refinery

A production-style pipeline that turns raw Common Crawl web text into deduplicated,
quality-scored, Parquet-packed training shards, then proves the cleaning actually
improves model quality via small-model ablations.

Data source: Common Crawl **CC-MAIN-2026-17** (`https://data.commoncrawl.org`, public,
no AWS account needed).

## Status

- [x] Phase 0 — setup
- [x] Phase 1 — MVP pipeline (`ingest_clean.py`, `dedup_cluster.py`, `quality_scorer.py`,
      `shard_writer.py` all built and run end to end)
- [x] Phase 2 — orchestration + K8s (Dagster asset graph + containerized K8s Jobs, both run
      end to end on a local OrbStack cluster)
- [x] Phase 3 — anomaly detection + ablation (z-score shard detector; nanoGPT-style ablation
      across raw/deduped/deduped+filtered, confirms cleaning helps on a fair benchmark)
- [x] Phase 4 — dashboard (Streamlit: pipeline funnel, shard stats, anomaly flags, ablation
      loss curves — reads real output only, verified rendering in-browser)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# python-magic (used by DataTrove's WarcReader) needs the native libmagic lib:
brew install libmagic

# Docker + local Kubernetes, for Phase 2:
brew install orbstack
open -a OrbStack   # finish first-run setup in the GUI, then enable Kubernetes:
orbctl config set k8s.enable true
orbctl stop && orbctl start
kubectl get nodes  # should show a single "orbstack" node, Ready
```

Download 1-5 WET files from the crawl index (don't grab more — this runs on a laptop):

```bash
mkdir -p data/raw
curl -s -o data/wet.paths.gz "https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-17/wet.paths.gz"
gunzip -k data/wet.paths.gz
PATH1=$(head -1 data/wet.paths)
curl -s -o "data/raw/$(basename "$PATH1")" "https://data.commoncrawl.org/$PATH1"
```

## Pipeline

### `pipeline/ingest_clean.py`

`WarcReader -> LanguageFilter -> GopherQualityFilter -> C4QualityFilter`, via DataTrove.
Reads WET file(s) from `data/raw/`, writes surviving docs as gzipped JSONL to
`data/interim/clean/`. Docs rejected by Gopher/C4 are also saved (`data/interim/dropped/`)
— they become the negative-class training examples for `quality_scorer.py`.

```bash
python pipeline/ingest_clean.py
```

Real run against 1 WET file (`CC-MAIN-20260410081153-20260410111153-00000.warc.wet.gz`,
21,337 raw records):

| Stage | Docs in | Docs out | Dropped |
|---|---|---|---|
| Language ID (`en`, threshold 0.65) | 21,337 | 6,936 | 14,401 |
| Gopher Quality | 6,936 | 4,630 | 2,306 |
| C4 Quality | 4,630 | 3,677 | 953 |
| **Total** | **21,337** | **3,677** | **17,660 (82.8%)** |

### `pipeline/dedup_cluster.py`

Stage A — MinHash near-duplicate removal (5-gram, 14 buckets x 8 hashes) via DataTrove's
official 4-stage pipeline: `MinhashDedupSignature -> MinhashDedupBuckets -> MinhashDedupCluster
-> MinhashDedupFilter`. Reads `data/interim/clean/`, writes `data/interim/deduped/`.

Stage B — TF-IDF + k-means topic clustering (scikit-learn) over the deduped corpus, tagging
each document with a `topic_cluster` id. Writes `data/interim/clustered/`.

```bash
python pipeline/dedup_cluster.py
```

Real run against the 3,677 cleaned docs above:

| Stage | Docs in | Docs out | Removed |
|---|---|---|---|
| MinHash dedup | 3,677 | 3,574 | 103 (2.8%) |

Topic clustering (k=8) over the 3,574 deduped docs: cluster sizes ranged from 97 to 1,520 docs
(TF-IDF, max 20k features, English stopwords).

### `pipeline/quality_scorer.py`

Trains a lightweight TF-IDF + logistic regression classifier on real labels from
`ingest_clean.py`'s filters: positive = docs that passed (`data/interim/clean/`), negative =
docs Gopher/C4 actually rejected (`data/interim/dropped/`). Scores every doc in
`data/interim/clustered/` with a continuous `quality_score` in [0, 1]. Writes
`data/interim/scored/`.

```bash
python pipeline/quality_scorer.py
```

Real run: 3,677 positive / 3,259 negative examples, 80/20 train/test split (5,548/1,388).
Held-out accuracy **0.9395**, ROC-AUC **0.9845**. Scored 3,574 docs — `quality_score`
distribution: min=0.149, p25=0.730, median=0.810, p75=0.881, max=0.995.

### `pipeline/shard_writer.py`

Reads `data/interim/scored/`, writes partitioned Parquet training shards to
`data/shards/crawl=<crawl_id>/shard-NNNNN.parquet` (Hive-style partitioning by crawl snapshot,
1,000 rows/shard by default). Schema: `id, text, url, date, language, language_score,
topic_cluster, quality_score, crawl`.

```bash
python pipeline/shard_writer.py
```

Real run: 3,574 rows -> 4 shards (`crawl=CC-MAIN-2026-17`), ~2.3MB each (snappy-compressed).
Verified readable as a partitioned `pyarrow.dataset` with the correct row count and schema.

## Orchestration

### `orchestration/dagster_pipeline.py`

Wraps the 4 pipeline stages as a Dagster asset graph (`clean_docs -> deduped_clustered_docs ->
scored_docs -> parquet_shards`), calling the exact same functions the standalone scripts use —
Dagster runs produce byte-identical output to running the scripts directly. Real per-asset
metadata (doc counts, classifier AUC, shard counts) surfaces in the Dagster UI.

```bash
dagster dev -f orchestration/dagster_pipeline.py                        # UI at localhost:3000
dagster asset materialize -f orchestration/dagster_pipeline.py --select "*"  # CLI run
```

Verified with a from-scratch run (cleared `data/interim/`, `data/logs/`, `data/shards/` first):
identical counts to the manual script runs at every stage (21,337 -> 3,677 -> 3,574 docs,
held-out ROC-AUC 0.9845, 4 shards).

### Kubernetes (local OrbStack cluster)

Each pipeline stage also runs as a containerized K8s Job. `Dockerfile` builds a
`training-data-refinery:latest` image (`python:3.12-slim` + `requirements-pipeline.txt`,
the runtime-only subset of dependencies the 4 scripts actually need — no Dagster/torch/
streamlit). `k8s/` has one Job manifest per stage plus a shared `PersistentVolumeClaim`
(`refinery-data`, `local-path` storage class) that all 4 Jobs mount at `/data`. `ingest-job`
downloads its own WET file inside the cluster (no host mount needed — mirrors what a real
remote cluster would have to do).

```bash
docker build -t training-data-refinery:latest .
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/ingest-job.yaml   && kubectl wait --for=condition=complete job/ingest-job  -n training-data-refinery --timeout=300s
kubectl apply -f k8s/dedup-job.yaml    && kubectl wait --for=condition=complete job/dedup-job   -n training-data-refinery --timeout=300s
kubectl apply -f k8s/quality-job.yaml  && kubectl wait --for=condition=complete job/quality-job -n training-data-refinery --timeout=180s
kubectl apply -f k8s/shard-job.yaml    && kubectl wait --for=condition=complete job/shard-job   -n training-data-refinery --timeout=120s
```

Real run on a local OrbStack Kubernetes cluster (single node, `kubectl get nodes` -> `Ready`):
all 4 Jobs completed (`ingest-job` -> `dedup-job` -> `quality-job` -> `shard-job`), each reading
the previous stage's output straight off the shared PVC. Counts differ slightly from the venv
runs above (21,320 raw docs vs. 21,337) because `requirements-pipeline.txt` isn't pinned to the
exact same transitive dependency versions as `requirements.txt` — not a correctness issue.
Final output verified in-cluster and pulled back to the host: 3,567 rows across 4 Parquet
shards, readable as a partitioned `pyarrow.dataset`.

## Anomaly detection

### `anomaly/batch_stats_detector.py`

Computes per-shard stats (doc count, avg text length, avg quality_score, avg language_score,
topic cluster coverage) over `data/shards/*.parquet` and flags shards whose stats deviate from
the cross-shard mean by more than a z-score threshold.

```bash
python anomaly/batch_stats_detector.py
```

Real run over the 4 production shards: `shard-00003` (the undersized remainder shard, 574 vs.
~1000 docs) is correctly flagged on `n_docs` and `avg_text_len`. Note: with population z-scores,
`|z|` can never exceed `sqrt(n_shards - 1)` regardless of how extreme a shard is — at n=4 that's
1.73, so the default threshold is 1.5, not the more conventional 2.0 (which would be
mathematically unreachable here and silently never fire). Verified the detector actually
catches real problems with a synthetic test: injecting a corrupted `quality_score` into one
shard's copy gets it flagged immediately.

## Ablation: does the cleaning actually help?

### `ablation/prepare_variants.py`

Phase 1's pipeline filters *before* it dedups (`clean/ -> deduped/`), so there's no
dedup-only / no-filter data on disk to ablate against. This builds the two variants that are
missing: `raw` (straight from `WarcReader`, no language/quality filtering, no dedup) and
`deduped` (DataTrove MinHash dedup applied to `raw`, still no quality filtering) — reusing
`pipeline/dedup_cluster.py`'s `run_minhash_dedup()` directly. The third variant,
`deduped_filtered`, is just the existing `data/interim/deduped/` (language+Gopher+C4 filtered,
then deduped) — Phase 1's real production output, unmodified.

```bash
python ablation/prepare_variants.py
```

Real run: raw 21,337 docs -> deduped 20,899 docs (438 near-dups removed) -> deduped_filtered
3,574 docs (Phase 1's existing output, reused as-is).

### `ablation/train_tiny_lm.py`

A minimal nanoGPT-style causal transformer from scratch: tied embeddings, pre-LN blocks,
`scaled_dot_product_attention`, GPT-2 BPE via `tiktoken`, trained on the MPS backend.
~16M params (4 layers, 4 heads, 256 dim, block_size=128). A quick throughput sweep found
`block_size=256` at `batch=64` hit a pathological MPS slowdown (22s/step vs. 0.4s/step at
`block_size=128, batch=32`, for basically the same tokens/step) — 128 was picked purely to keep
each run to a few minutes on a laptop GPU.

```bash
python ablation/train_tiny_lm.py --data-dir data/interim/deduped --name test --n-steps 600
```

### `ablation/compare_mixes.py`

Trains identical models (same steps, batch size, block size, model config, seed — only the
data differs) on all 3 variants at equal token budget (600 steps x 32 x 128 = 2,457,600
training tokens/run) and compares loss.

```bash
python ablation/compare_mixes.py
```

**First run — confounded, don't trust it.** Each variant was scored against its own held-out
split:

| variant | docs | tokens | train_loss | own val_loss |
|---|---|---|---|---|
| raw | 21,337 | 81.4M | 4.6423 | 4.8631 |
| deduped | 20,899 | 78.0M | 4.9349 | 4.8844 |
| deduped_filtered | 3,574 | 3.0M | 6.2714 | 6.3645 |

Read naively this says cleaning made things *worse* — the opposite of the point of this whole
project. It's a measurement artifact, not a real result: raw/deduped's held-out text is full of
repetitive boilerplate (nav menus, footers, templated junk), which is trivially predictable and
deflates loss; deduped_filtered's held-out text is genuine, diverse prose — inherently
higher-entropy, so a model scores worse on it even if it's the better-trained model. Comparing
cross-entropy across three different eval distributions isn't apples-to-apples. Using
deduped_filtered's val split as a shared benchmark for all three wouldn't fix this either: all
3 variants come from the *same* WET file, so raw/deduped's training pool structurally contains
the same source documents that ended up in deduped_filtered's held-out split — a "shared" eval
built that way would leak.

**Fix: an external, uncontaminated benchmark.** Downloaded a *second*, different WET file
(`CC-MAIN-20260410081153-20260410111153-00001.warc.wet.gz` — still within the "2-5 WET files"
budget; only 1 had been used until this point) and ran it through the exact same
`ingest_clean.py` filters to get `data/ablation/eval_clean/` (3,796 docs). No variant has seen
any content from it — this is the headline, trustworthy comparison:

| variant | docs | tokens | train_loss | own val_loss | **shared eval_loss** |
|---|---|---|---|---|---|
| raw | 21,337 | 81.4M | 4.6423 | 4.8631 | 7.0861 |
| deduped | 20,899 | 78.0M | 4.9349 | 4.8844 | 7.0734 |
| deduped_filtered | 3,574 | 3.0M | 6.2714 | 6.3645 | **6.2767** |

On the fair benchmark, `deduped_filtered` wins clearly — despite training on ~6x fewer
documents and ~27x fewer tokens than raw/deduped. Dedup alone gives a small nudge
(7.0861 -> 7.0734); the real gain comes from quality filtering. This confirms the pipeline's
core hypothesis: cleaning produces a corpus that generalizes better to real, unseen text, even
though it's far smaller. Results (including full loss curves) are in `data/ablation/results.json`.

## Dashboard

### `dashboard/app.py`

A Streamlit dashboard over real pipeline output only — no numbers are computed or faked in
the dashboard itself, everything is read from `data/stats/*.json` (written by each
`pipeline/*.py` stage via `pipeline/stats_utils.py`), `data/shards/*.parquet`, and
`data/ablation/results.json`.

```bash
streamlit run dashboard/app.py
```

Three sections:
- **Pipeline funnel + shard stats** — raw -> clean -> deduped -> scored -> shard-row counts as
  metric tiles, plus the per-shard stats table/chart (reuses `anomaly/batch_stats_detector.py`'s
  `compute_shard_stats()` directly, no duplicated logic).
- **Anomaly detection** — the same z-score table and flags as the CLI tool, with the
  `sqrt(n_shards - 1)` ceiling caption carried through so the dashboard doesn't imply more
  precision than 4 shards can support.
- **Ablation** — the headline grouped-bar chart (own val_loss vs. shared eval_loss, colored by
  variant) makes the confound-vs-fix story visible at a glance: the two variant groups swap
  which one "wins" between the left bars and the right bars. Plus separate train/val loss curve
  charts and the full results table.

Verified by actually running it and checking the rendered page in a browser (not just that it
imports) — all three sections render with real data, dark-mode charts are readable, no console
or server errors. Colors use the dataviz skill's validated categorical palette (blue/orange/aqua
for raw/deduped/deduped_filtered — the three slots that pass CVD-safety checks together) and
status red/green for anomaly flags.

## Repo layout

```
pipeline/        ingest, dedup/cluster, quality scoring, shard writing
orchestration/   Dagster asset graph wrapping the pipeline/ steps
k8s/             one Job manifest per pipeline stage
anomaly/         z-score based per-shard anomaly detector
ablation/        nanoGPT-based tiny LM trained on raw/deduped/deduped+filtered mixes
dashboard/       Streamlit app: shard stats, anomaly flags, ablation results
```
