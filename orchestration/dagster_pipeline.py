"""
Phase 2 - dagster_pipeline.py

Wraps the Phase 1 pipeline/ scripts as a Dagster asset graph:

    clean_docs -> deduped_clustered_docs -> scored_docs -> parquet_shards

Each asset calls straight into the same functions the standalone scripts use
(pipeline/ingest_clean.py, dedup_cluster.py, quality_scorer.py, shard_writer.py),
so running via Dagster produces byte-identical output to running the scripts
directly. Real doc counts / metrics are surfaced as Dagster asset metadata.

Run locally with the Dagster UI:
    dagster dev -f orchestration/dagster_pipeline.py

Or materialize all assets from the CLI:
    dagster asset materialize -f orchestration/dagster_pipeline.py --select "*"
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from dagster import AssetExecutionContext, Definitions, MaterializeResult, MetadataValue, asset
from datatrove.executor.local import LocalPipelineExecutor

from pipeline.dedup_cluster import DEFAULT_CLUSTERED_DIR, DEFAULT_DEDUPED_DIR, DEFAULT_MINHASH_DIR
from pipeline.dedup_cluster import DEFAULT_LOG_DIR as DEDUP_LOG_DIR
from pipeline.dedup_cluster import count_jsonl_docs, run_minhash_dedup, run_topic_clustering
from pipeline.ingest_clean import DEFAULT_CLEAN_DIR, DEFAULT_DROPPED_DIR, DEFAULT_RAW_DIR
from pipeline.ingest_clean import DEFAULT_LOG_DIR as INGEST_LOG_DIR
from pipeline.ingest_clean import build_pipeline as build_ingest_pipeline
from pipeline.ingest_clean import counts as ingest_counts
from pipeline.quality_scorer import DEFAULT_SCORE_INPUT_DIR, DEFAULT_SCORED_DIR
from pipeline.quality_scorer import DEFAULT_CLEAN_DIR as QS_CLEAN_DIR
from pipeline.quality_scorer import DEFAULT_DROPPED_DIRS as QS_DROPPED_DIRS
from pipeline.quality_scorer import score_corpus, train_classifier
from pipeline.shard_writer import DEFAULT_CRAWL_ID, DEFAULT_SHARDS_DIR
from pipeline.shard_writer import DEFAULT_INPUT_DIR as SHARD_INPUT_DIR
from pipeline.shard_writer import write_shards

N_TOPIC_CLUSTERS = 8


@asset(group_name="training_data_refinery")
def clean_docs(context: AssetExecutionContext) -> MaterializeResult:
    """WarcReader -> LanguageFilter -> GopherQualityFilter -> C4QualityFilter (pipeline/ingest_clean.py)."""
    pipeline = build_ingest_pipeline(str(DEFAULT_RAW_DIR), str(DEFAULT_CLEAN_DIR), str(DEFAULT_DROPPED_DIR))
    executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1, workers=1, logging_dir=str(INGEST_LOG_DIR))
    executor.run()

    raw = ingest_counts["raw"]
    clean = ingest_counts["clean"]
    context.log.info(f"ingest_clean: raw={raw} clean={clean}")

    return MaterializeResult(
        metadata={
            "raw_docs": raw,
            "clean_docs": clean,
            "dropped_pct": MetadataValue.float(round((raw - clean) / raw * 100, 2)) if raw else 0.0,
        }
    )


@asset(deps=[clean_docs], group_name="training_data_refinery")
def deduped_clustered_docs(context: AssetExecutionContext) -> MaterializeResult:
    """MinHash near-dup removal + TF-IDF/k-means topic clustering (pipeline/dedup_cluster.py)."""
    run_minhash_dedup(
        str(DEFAULT_CLEAN_DIR), str(DEFAULT_MINHASH_DIR), str(DEFAULT_DEDUPED_DIR), str(DEDUP_LOG_DIR), tasks=1
    )
    clean_count = count_jsonl_docs(str(DEFAULT_CLEAN_DIR))
    deduped_count = count_jsonl_docs(str(DEFAULT_DEDUPED_DIR))

    n_docs, k, cluster_sizes = run_topic_clustering(
        str(DEFAULT_DEDUPED_DIR), str(DEFAULT_CLUSTERED_DIR), n_clusters=N_TOPIC_CLUSTERS
    )
    context.log.info(f"dedup_cluster: clean={clean_count} deduped={deduped_count} clusters={k}")

    return MaterializeResult(
        metadata={
            "clean_docs_in": clean_count,
            "deduped_docs_out": deduped_count,
            "near_dups_removed": clean_count - deduped_count,
            "n_topic_clusters": k,
        }
    )


@asset(deps=[deduped_clustered_docs], group_name="training_data_refinery")
def scored_docs(context: AssetExecutionContext) -> MaterializeResult:
    """TF-IDF + logistic regression quality scoring (pipeline/quality_scorer.py)."""
    vectorizer, clf, metrics = train_classifier(str(QS_CLEAN_DIR), [str(d) for d in QS_DROPPED_DIRS])
    n_docs, scores = score_corpus(vectorizer, clf, str(DEFAULT_SCORE_INPUT_DIR), str(DEFAULT_SCORED_DIR))
    context.log.info(f"quality_scorer: scored={n_docs} auc={metrics['roc_auc']:.4f}")

    return MaterializeResult(
        metadata={
            "n_scored": n_docs,
            "held_out_accuracy": MetadataValue.float(round(metrics["accuracy"], 4)),
            "held_out_roc_auc": MetadataValue.float(round(metrics["roc_auc"], 4)),
            "median_quality_score": MetadataValue.float(round(float(np.median(scores)), 4)),
        }
    )


@asset(deps=[scored_docs], group_name="training_data_refinery")
def parquet_shards(context: AssetExecutionContext) -> MaterializeResult:
    """Partitioned Parquet shard writing (pipeline/shard_writer.py)."""
    total_rows, shard_counts = write_shards(str(SHARD_INPUT_DIR), str(DEFAULT_SHARDS_DIR), DEFAULT_CRAWL_ID, 1000)
    context.log.info(f"shard_writer: rows={total_rows} shards={len(shard_counts)}")

    return MaterializeResult(
        metadata={
            "total_rows": total_rows,
            "n_shards": len(shard_counts),
            "crawl_partition": DEFAULT_CRAWL_ID,
        }
    )


defs = Definitions(assets=[clean_docs, deduped_clustered_docs, scored_docs, parquet_shards])
