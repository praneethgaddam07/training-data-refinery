"""
Phase 1 - ingest_clean.py

WarcReader -> LanguageFilter -> GopherQualityFilter -> C4QualityFilter

Reads raw Common Crawl WET file(s) from data/raw/, applies DataTrove's
standard language + quality filters, and writes the surviving documents
to data/interim/clean/ as gzipped JSONL. Prints raw doc count in vs.
clean doc count out.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import C4QualityFilter, GopherQualityFilter, LanguageFilter
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.writers import JsonlWriter
from stats_utils import write_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_CLEAN_DIR = REPO_ROOT / "data" / "interim" / "clean"
DEFAULT_DROPPED_DIR = REPO_ROOT / "data" / "interim" / "dropped"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs" / "ingest_clean"

# populated by the count_docs pipeline steps below; only valid for
# LocalPipelineExecutor(workers=1), which runs in-process (no multiprocessing)
counts = {"raw": 0, "clean": 0}


def count_docs(key: str):
    """Pipeline step that passes documents through unchanged while counting them."""

    def _step(data, rank: int = 0, world_size: int = 1):
        n = 0
        for doc in data:
            n += 1
            yield doc
        counts[key] = n

    return _step


def build_pipeline(raw_dir: str, clean_dir: str, dropped_dir: str, glob_pattern: str = "*.warc.wet.gz"):
    # Gopher/C4 rejects are saved (not just dropped) so quality_scorer.py has real
    # negative-class examples: documents a heuristic quality filter actually rejected.
    return [
        WarcReader(raw_dir, glob_pattern=glob_pattern, doc_progress=True),
        count_docs("raw"),
        LanguageFilter(languages=["en"], language_threshold=0.65),
        GopherQualityFilter(exclusion_writer=JsonlWriter(f"{dropped_dir}/gopher")),
        C4QualityFilter(exclusion_writer=JsonlWriter(f"{dropped_dir}/c4")),
        count_docs("clean"),
        JsonlWriter(clean_dir),
    ]


def main():
    parser = argparse.ArgumentParser(description="Clean raw Common Crawl WET file(s) with DataTrove filters.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR), help="Folder containing *.warc.wet.gz files")
    parser.add_argument("--clean-dir", default=str(DEFAULT_CLEAN_DIR), help="Output folder for cleaned JSONL")
    parser.add_argument("--dropped-dir", default=str(DEFAULT_DROPPED_DIR), help="Output folder for rejected docs")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="DataTrove executor logging/completion dir")
    parser.add_argument("--glob", default="*.warc.wet.gz")
    args = parser.parse_args()

    pipeline = build_pipeline(args.raw_dir, args.clean_dir, args.dropped_dir, args.glob)
    executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1, workers=1, logging_dir=str(args.log_dir))
    executor.run()

    raw_count = counts["raw"]
    clean_count = counts["clean"]
    dropped = raw_count - clean_count
    pct = (dropped / raw_count * 100) if raw_count else 0.0

    print("\n=== ingest_clean.py summary ===")
    print(f"Raw docs in:    {raw_count}")
    print(f"Clean docs out: {clean_count}")
    print(f"Dropped:        {dropped} ({pct:.1f}%)")
    print(f"Clean output written to: {args.clean_dir}")
    print(f"Rejected docs (Gopher/C4) written to: {args.dropped_dir}")

    write_stats(
        "ingest_clean",
        {"raw_docs": raw_count, "clean_docs": clean_count, "dropped": dropped, "dropped_pct": round(pct, 2)},
    )


if __name__ == "__main__":
    main()
