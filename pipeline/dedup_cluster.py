"""
Phase 1 - dedup_cluster.py

Stage A: DataTrove MinHash near-duplicate removal (5-gram, 14 buckets x 8 hashes),
         via the official 4-stage pipeline:
         MinhashDedupSignature -> MinhashDedupBuckets -> MinhashDedupCluster -> MinhashDedupFilter
Stage B: TF-IDF + k-means topic clustering over the deduped corpus (scikit-learn),
         tagging each surviving document with a `topic_cluster` id.

Input:  data/interim/clean/      (output of ingest_clean.py)
Output: data/interim/deduped/    (minhash-deduped JSONL)
        data/interim/clustered/  (deduped JSONL + topic_cluster metadata)
"""

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
    MinhashDedupSignature,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.utils.typeshelper import Languages
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from stats_utils import write_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN_DIR = REPO_ROOT / "data" / "interim" / "clean"
DEFAULT_MINHASH_DIR = REPO_ROOT / "data" / "interim" / "minhash"
DEFAULT_DEDUPED_DIR = REPO_ROOT / "data" / "interim" / "deduped"
DEFAULT_CLUSTERED_DIR = REPO_ROOT / "data" / "interim" / "clustered"
DEFAULT_LOG_DIR = REPO_ROOT / "data" / "logs" / "dedup_cluster"

MINHASH_CONFIG = MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8)


def count_jsonl_docs(folder: str) -> int:
    total = 0
    for f in Path(folder).glob("*.jsonl.gz"):
        with gzip.open(f, "rt") as fh:
            total += sum(1 for _ in fh)
    return total


def run_minhash_dedup(clean_dir: str, minhash_dir: str, deduped_dir: str, log_dir: str, tasks: int):
    # the same reader (same files, same shard assignment) must be used in stage 1 and stage 4
    # so that document indices line up with the .remove files produced by stage 3
    input_reader = JsonlReader(clean_dir)

    stage1 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            MinhashDedupSignature(
                output_folder=f"{minhash_dir}/signatures", config=MINHASH_CONFIG, language=Languages.english
            ),
        ],
        tasks=tasks,
        logging_dir=f"{log_dir}/signatures",
    )
    stage2 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupBuckets(
                input_folder=f"{minhash_dir}/signatures",
                output_folder=f"{minhash_dir}/buckets",
                config=MINHASH_CONFIG,
            )
        ],
        tasks=MINHASH_CONFIG.num_buckets,
        logging_dir=f"{log_dir}/buckets",
        depends=stage1,
    )
    stage3 = LocalPipelineExecutor(
        pipeline=[
            MinhashDedupCluster(
                input_folder=f"{minhash_dir}/buckets",
                output_folder=f"{minhash_dir}/remove_ids",
                config=MINHASH_CONFIG,
            )
        ],
        tasks=1,
        logging_dir=f"{log_dir}/clusters",
        depends=stage2,
    )
    stage4 = LocalPipelineExecutor(
        pipeline=[
            input_reader,
            MinhashDedupFilter(input_folder=f"{minhash_dir}/remove_ids"),
            JsonlWriter(output_folder=deduped_dir),
        ],
        tasks=tasks,
        logging_dir=f"{log_dir}/filter",
        depends=stage3,
    )
    return stage4.run()


def run_topic_clustering(deduped_dir: str, clustered_dir: str, n_clusters: int, random_state: int = 42):
    deduped_dir = Path(deduped_dir)
    clustered_dir = Path(clustered_dir)
    clustered_dir.mkdir(parents=True, exist_ok=True)

    docs = []
    for f in sorted(deduped_dir.glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                docs.append(json.loads(line))

    if not docs:
        raise RuntimeError(f"No deduped documents found in {deduped_dir}")

    texts = [d["text"] for d in docs]
    k = min(n_clusters, len(texts))

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2)
    tfidf = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(tfidf)

    for doc, label in zip(docs, labels):
        doc.setdefault("metadata", {})["topic_cluster"] = int(label)

    out_path = clustered_dir / "00000.jsonl.gz"
    with gzip.open(out_path, "wt") as fh:
        for doc in docs:
            fh.write(json.dumps(doc) + "\n")

    return len(docs), k, Counter(labels.tolist())


def main():
    parser = argparse.ArgumentParser(description="MinHash-dedup and topic-cluster a cleaned Common Crawl corpus.")
    parser.add_argument("--clean-dir", default=str(DEFAULT_CLEAN_DIR))
    parser.add_argument("--minhash-dir", default=str(DEFAULT_MINHASH_DIR))
    parser.add_argument("--deduped-dir", default=str(DEFAULT_DEDUPED_DIR))
    parser.add_argument("--clustered-dir", default=str(DEFAULT_CLUSTERED_DIR))
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--tasks", type=int, default=1, help="number of input file shards (stage 1 & 4 must match)")
    parser.add_argument("--n-clusters", type=int, default=8)
    args = parser.parse_args()

    print("=== Stage A: MinHash near-duplicate removal ===")
    t0 = time.perf_counter()
    run_minhash_dedup(args.clean_dir, args.minhash_dir, args.deduped_dir, args.log_dir, tasks=args.tasks)
    minhash_elapsed = time.perf_counter() - t0

    clean_count = count_jsonl_docs(args.clean_dir)
    deduped_count = count_jsonl_docs(args.deduped_dir)
    removed = clean_count - deduped_count
    pct = (removed / clean_count * 100) if clean_count else 0.0
    print("\n=== dedup summary ===")
    print(f"Clean docs in:            {clean_count}")
    print(f"Deduped docs out:         {deduped_count}")
    print(f"Near-duplicates removed:  {removed} ({pct:.1f}%)")
    print(f"Stage A wall time:        {minhash_elapsed:.3f}s")

    print("\n=== Stage B: TF-IDF + k-means topic clustering ===")
    t0 = time.perf_counter()
    n_docs, k, cluster_sizes = run_topic_clustering(args.deduped_dir, args.clustered_dir, n_clusters=args.n_clusters)
    cluster_elapsed = time.perf_counter() - t0
    print(f"Clustered {n_docs} docs into {k} topic clusters")
    for cid in sorted(cluster_sizes):
        print(f"  cluster {cid}: {cluster_sizes[cid]} docs")
    print(f"Output written to: {args.clustered_dir}")
    print(f"Stage B wall time: {cluster_elapsed:.3f}s")

    write_stats(
        "dedup_cluster",
        {
            "clean_docs_in": clean_count,
            "deduped_docs_out": deduped_count,
            "near_dups_removed": removed,
            "near_dups_removed_pct": round(pct, 2),
            "n_topic_clusters": k,
            "cluster_sizes": {str(cid): int(cluster_sizes[cid]) for cid in sorted(cluster_sizes)},
            "minhash_elapsed_sec": round(minhash_elapsed, 3),
            "clustering_elapsed_sec": round(cluster_elapsed, 3),
        },
    )


if __name__ == "__main__":
    main()
