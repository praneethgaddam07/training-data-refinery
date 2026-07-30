"""
Phase 3 - prepare_variants.py

Builds the 3 data variants used by compare_mixes.py for the ablation:

  raw              - straight from WarcReader, no language/quality filtering, no dedup
  deduped          - MinHash dedup applied to the raw variant (no quality filtering)
  deduped_filtered - pipeline/dedup_cluster.py's existing output: language+Gopher+C4
                      filtered, then MinHash deduped (data/interim/deduped/)

"raw" and "deduped" don't exist yet because Phase 1's pipeline filters before it dedups
(clean/ -> deduped/), so there's no dedup-only-no-filter variant on disk. This script
fills that gap by running WarcReader (unfiltered) and DataTrove's MinHash pipeline
directly against it, reusing pipeline/dedup_cluster.py's run_minhash_dedup() as-is.

Output: data/ablation/raw/, data/ablation/deduped/
        (deduped_filtered reuses data/interim/deduped/ in place)
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers import JsonlWriter

from pipeline.dedup_cluster import DEFAULT_DEDUPED_DIR as DEDUPED_FILTERED_DIR
from pipeline.dedup_cluster import count_jsonl_docs, run_minhash_dedup
from pipeline.ingest_clean import DEFAULT_RAW_DIR

DEFAULT_ABLATION_DIR = REPO_ROOT / "data" / "ablation"
RAW_VARIANT_DIR = DEFAULT_ABLATION_DIR / "raw"
RAW_MINHASH_DIR = DEFAULT_ABLATION_DIR / "raw_minhash"
DEDUPED_VARIANT_DIR = DEFAULT_ABLATION_DIR / "deduped"
LOG_DIR = REPO_ROOT / "data" / "logs" / "ablation_prep"


def build_raw_variant(raw_wet_dir: str, out_dir: str, log_dir: str):
    pipeline = [
        WarcReader(raw_wet_dir, glob_pattern="*.warc.wet.gz", doc_progress=True),
        JsonlWriter(out_dir),
    ]
    executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1, workers=1, logging_dir=log_dir)
    executor.run()


def main():
    print("=== building 'raw' variant (WarcReader only, no filters, no dedup) ===")
    build_raw_variant(str(DEFAULT_RAW_DIR), str(RAW_VARIANT_DIR), str(LOG_DIR / "raw"))
    n_raw = count_jsonl_docs(str(RAW_VARIANT_DIR))
    print(f"raw docs: {n_raw}")

    print("\n=== building 'deduped' variant (MinHash dedup on raw, no quality filtering) ===")
    run_minhash_dedup(
        str(RAW_VARIANT_DIR), str(RAW_MINHASH_DIR), str(DEDUPED_VARIANT_DIR), str(LOG_DIR / "dedup"), tasks=1
    )
    n_deduped = count_jsonl_docs(str(DEDUPED_VARIANT_DIR))
    print(f"deduped docs: {n_deduped} (removed {n_raw - n_deduped} near-dups from raw)")

    n_filtered = count_jsonl_docs(str(DEDUPED_FILTERED_DIR))
    print(f"\n'deduped_filtered' variant reuses existing {DEDUPED_FILTERED_DIR} ({n_filtered} docs)")

    print("\n=== summary ===")
    print(f"raw:              {n_raw} docs   -> {RAW_VARIANT_DIR}")
    print(f"deduped:          {n_deduped} docs   -> {DEDUPED_VARIANT_DIR}")
    print(f"deduped_filtered: {n_filtered} docs   -> {DEDUPED_FILTERED_DIR}")


if __name__ == "__main__":
    main()
