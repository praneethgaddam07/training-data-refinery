"""
Smoke tests: one per pipeline stage, run against the small synthetic fixture in
tests/fixtures/sample.warc.wet.gz (see tests/fixtures/generate_fixture.py). Each
test asserts the stage's real output exists with the expected shape/schema --
these are not exhaustive correctness tests (Phase 1's README documents the real
numbers from the actual Common Crawl run), just "does this stage run end to end
and produce something sane," fast enough to run on every commit.

Run with: pytest tests/
"""

import gzip
import json
import sys
from pathlib import Path

import pyarrow.dataset as ds
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datatrove.executor.local import LocalPipelineExecutor

from pipeline.dedup_cluster import count_jsonl_docs, run_minhash_dedup, run_topic_clustering
from pipeline.ingest_clean import build_pipeline
from pipeline.quality_scorer import score_corpus, train_classifier
from pipeline.shard_writer import write_shards

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"


def _count_jsonl(folder: Path) -> int:
    total = 0
    for f in folder.glob("*.jsonl.gz"):
        with gzip.open(f, "rt") as fh:
            total += sum(1 for _ in fh)
    return total


def _first_doc(folder: Path) -> dict:
    sample_file = next(folder.glob("*.jsonl.gz"))
    with gzip.open(sample_file, "rt") as fh:
        return json.loads(fh.readline())


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Runs the full pipeline once against the fixture; each test below asserts on
    the specific stage's output within this shared run."""
    base = tmp_path_factory.mktemp("pipeline_smoke")
    dirs = {name: base / name for name in ["clean", "dropped", "minhash", "deduped", "clustered", "scored", "shards"]}
    log_dir = base / "logs"

    pipeline = build_pipeline(
        str(FIXTURE_DIR), str(dirs["clean"]), str(dirs["dropped"]), glob_pattern="sample.warc.wet.gz"
    )
    LocalPipelineExecutor(pipeline=pipeline, tasks=1, workers=1, logging_dir=str(log_dir / "ingest")).run()

    run_minhash_dedup(str(dirs["clean"]), str(dirs["minhash"]), str(dirs["deduped"]), str(log_dir / "dedup"), tasks=1)
    run_topic_clustering(str(dirs["deduped"]), str(dirs["clustered"]), n_clusters=2)

    vectorizer, clf, _ = train_classifier(
        str(dirs["clean"]), [str(dirs["dropped"] / "gopher"), str(dirs["dropped"] / "c4")]
    )
    score_corpus(vectorizer, clf, str(dirs["clustered"]), str(dirs["scored"]))

    write_shards(str(dirs["scored"]), str(dirs["shards"]), "TEST-CRAWL", rows_per_shard=100)

    return dirs


def test_ingest_clean_smoke(pipeline_run):
    clean_count = _count_jsonl(pipeline_run["clean"])
    assert clean_count > 0, "ingest_clean produced no clean docs"
    assert clean_count < 12, "the fixture's foreign/short/junk docs should have been filtered out"

    doc = _first_doc(pipeline_run["clean"])
    assert "text" in doc and isinstance(doc["text"], str) and doc["text"]
    assert "id" in doc
    assert "url" in doc["metadata"]
    assert doc["metadata"]["language"] == "en"


def test_dedup_cluster_smoke(pipeline_run):
    clean_count = _count_jsonl(pipeline_run["clean"])
    deduped_count = _count_jsonl(pipeline_run["deduped"])
    assert deduped_count > 0
    assert deduped_count <= clean_count, "dedup must never increase the doc count"
    assert deduped_count < clean_count, "the fixture has a deliberate near-duplicate pair MinHash should catch"

    clustered_count = _count_jsonl(pipeline_run["clustered"])
    assert clustered_count == deduped_count, "clustering must not drop or add docs"

    doc = _first_doc(pipeline_run["clustered"])
    assert isinstance(doc["metadata"]["topic_cluster"], int)


def test_quality_scorer_smoke(pipeline_run):
    clustered_count = _count_jsonl(pipeline_run["clustered"])
    scored_count = _count_jsonl(pipeline_run["scored"])
    assert scored_count == clustered_count, "quality_scorer must score every doc, not drop any"

    doc = _first_doc(pipeline_run["scored"])
    score = doc["metadata"]["quality_score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_shard_writer_smoke(pipeline_run):
    shards_dir = pipeline_run["shards"]
    parquet_files = list(shards_dir.rglob("*.parquet"))
    assert len(parquet_files) > 0, "shard_writer produced no Parquet files"

    table = ds.dataset(str(shards_dir), partitioning="hive").to_table()
    expected_columns = [
        "id",
        "text",
        "url",
        "date",
        "language",
        "language_score",
        "topic_cluster",
        "quality_score",
        "crawl",
    ]
    assert table.schema.names == expected_columns
    assert table.num_rows == _count_jsonl(pipeline_run["scored"])
    assert table.column("crawl").to_pylist()[0] == "TEST-CRAWL"
