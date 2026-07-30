"""
Phase 1 - shard_writer.py

Reads deduped + topic-clustered + quality-scored JSONL documents and writes them
out as partitioned Parquet training shards: <shards-dir>/crawl=<crawl_id>/shard-NNNNN.parquet

Input:  data/interim/scored/  (output of quality_scorer.py)
Output: data/shards/crawl=<crawl_id>/shard-*.parquet          (default, local disk)
        s3://<bucket>/<prefix>/crawl=<crawl_id>/shard-*.parquet (pass an s3:// --shards-dir)

The AWS swap-in is opt-in per-invocation, not a hidden global switch: pass
--shards-dir s3://my-bucket/shards and pipeline/storage.py routes the Parquet
writes through pyarrow's S3FileSystem instead of local disk. Nothing changes
for local runs (see README "AWS variant" section).
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyarrow as pa
import pyarrow.parquet as pq

import storage
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
    filesystem, base_path = storage.resolve(shards_dir)
    partition_path = f"{base_path.rstrip('/')}/crawl={crawl_id}"
    if not storage.is_s3(shards_dir):
        filesystem.create_dir(partition_path, recursive=True)
    # S3 has no directories to create -- keys are created implicitly on write

    shard_rows = []
    shard_idx = 0
    total_rows = 0
    shard_counts = []

    def flush():
        nonlocal shard_idx
        if not shard_rows:
            return
        table = pa.Table.from_pylist(shard_rows, schema=SCHEMA)
        out_path = f"{partition_path}/shard-{shard_idx:05d}.parquet"
        pq.write_table(table, out_path, filesystem=filesystem, compression="snappy")
        size_bytes = filesystem.get_file_info(out_path).size
        shard_counts.append({"path": out_path, "name": out_path.rsplit("/", 1)[-1], "rows": len(shard_rows), "size_bytes": size_bytes})
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

    t0 = time.perf_counter()
    total_rows, shard_counts = write_shards(args.input_dir, args.shards_dir, args.crawl_id, args.rows_per_shard)
    elapsed = time.perf_counter() - t0

    backend = "s3" if storage.is_s3(args.shards_dir) else "local"
    print("=== shard_writer.py summary ===")
    print(f"Storage backend: {backend}")
    print(f"Wall time: {elapsed:.3f}s ({total_rows / elapsed:.1f} docs/sec)")
    print(f"Crawl partition: crawl={args.crawl_id}")
    print(f"Total rows written: {total_rows}")
    print(f"Shards written: {len(shard_counts)}")
    shard_details = []
    for shard in shard_counts:
        size_kb = shard["size_bytes"] / 1024
        print(f"  {shard['name']}: {shard['rows']} rows, {size_kb:.1f} KB")
        shard_details.append({"name": shard["name"], "rows": shard["rows"], "size_kb": round(size_kb, 1)})
    print(f"Output written to: {args.shards_dir.rstrip('/')}/crawl={args.crawl_id}")

    write_stats(
        "shard_writer",
        {
            "storage_backend": backend,
            "crawl_id": args.crawl_id,
            "total_rows": total_rows,
            "n_shards": len(shard_counts),
            "rows_per_shard_target": args.rows_per_shard,
            "shards": shard_details,
            "elapsed_sec": round(elapsed, 3),
            "docs_per_sec": round(total_rows / elapsed, 1),
        },
    )


if __name__ == "__main__":
    main()
