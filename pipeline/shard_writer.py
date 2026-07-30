"""
Phase 1 - shard_writer.py

Reads deduped + topic-clustered + quality-scored JSONL documents and writes them
out as partitioned Parquet training shards: data/shards/crawl=<crawl_id>/shard-NNNNN.parquet

Input:  data/interim/scored/  (output of quality_scorer.py)
Output: data/shards/crawl=<crawl_id>/shard-*.parquet
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow as pa
import pyarrow.parquet as pq

from stats_utils import write_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "interim" / "scored"
DEFAULT_SHARDS_DIR = REPO_ROOT / "data" / "shards"
DEFAULT_CRAWL_ID = "CC-MAIN-2026-17"

SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("text", pa.string()),
        ("url", pa.string()),
        ("date", pa.string()),
        ("language", pa.string()),
        ("language_score", pa.float32()),
        ("topic_cluster", pa.int32()),
        ("quality_score", pa.float32()),
        ("crawl", pa.string()),
    ]
)


def load_docs(input_dir: str):
    for f in sorted(Path(input_dir).glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                yield json.loads(line)


def to_row(doc: dict, crawl_id: str) -> dict:
    meta = doc.get("metadata", {})
    return {
        "id": doc["id"],
        "text": doc["text"],
        "url": meta.get("url"),
        "date": meta.get("date"),
        "language": meta.get("language"),
        "language_score": meta.get("language_score"),
        "topic_cluster": meta.get("topic_cluster", -1),
        "quality_score": meta.get("quality_score"),
        "crawl": crawl_id,
    }


def write_shards(input_dir: str, shards_dir: str, crawl_id: str, rows_per_shard: int):
    partition_dir = Path(shards_dir) / f"crawl={crawl_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    shard_rows = []
    shard_idx = 0
    total_rows = 0
    shard_counts = []

    def flush():
        nonlocal shard_idx
        if not shard_rows:
            return
        table = pa.Table.from_pylist(shard_rows, schema=SCHEMA)
        out_path = partition_dir / f"shard-{shard_idx:05d}.parquet"
        pq.write_table(table, out_path, compression="snappy")
        shard_counts.append((out_path, len(shard_rows)))
        shard_idx += 1
        shard_rows.clear()

    for doc in load_docs(input_dir):
        shard_rows.append(to_row(doc, crawl_id))
        total_rows += 1
        if len(shard_rows) >= rows_per_shard:
            flush()
    flush()

    return total_rows, shard_counts


def main():
    parser = argparse.ArgumentParser(description="Write partitioned Parquet training shards.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--shards-dir", default=str(DEFAULT_SHARDS_DIR))
    parser.add_argument("--crawl-id", default=DEFAULT_CRAWL_ID, help="crawl snapshot id used as the partition key")
    parser.add_argument("--rows-per-shard", type=int, default=1000)
    args = parser.parse_args()

    total_rows, shard_counts = write_shards(args.input_dir, args.shards_dir, args.crawl_id, args.rows_per_shard)

    print("=== shard_writer.py summary ===")
    print(f"Crawl partition: crawl={args.crawl_id}")
    print(f"Total rows written: {total_rows}")
    print(f"Shards written: {len(shard_counts)}")
    shard_details = []
    for path, n in shard_counts:
        size_kb = path.stat().st_size / 1024
        print(f"  {path.name}: {n} rows, {size_kb:.1f} KB")
        shard_details.append({"name": path.name, "rows": n, "size_kb": round(size_kb, 1)})
    print(f"Output written to: {Path(args.shards_dir) / f'crawl={args.crawl_id}'}")

    write_stats(
        "shard_writer",
        {
            "crawl_id": args.crawl_id,
            "total_rows": total_rows,
            "n_shards": len(shard_counts),
            "rows_per_shard_target": args.rows_per_shard,
            "shards": shard_details,
        },
    )


if __name__ == "__main__":
    main()
