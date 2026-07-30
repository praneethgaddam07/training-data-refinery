"""
Phase 3 - batch_stats_detector.py

Computes per-shard statistics over data/shards/*.parquet and flags shards whose
stats deviate from the cross-shard mean by more than a z-score threshold.

Note: with only a handful of shards, z-scores are low statistical power — this
is meant as a smoke-test signal ("does anything look obviously off"), not a
rigorous outlier test. It will report that plainly if too few shards exist.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHARDS_DIR = REPO_ROOT / "data" / "shards"
# NOTE: with population z-scores, |z| can never exceed sqrt(n-1) regardless of how
# extreme a single shard is (e.g. n=4 -> max possible |z| = 1.73). The default here
# is chosen relative to that ceiling; main() also prints the actual ceiling for the
# shard count on hand so the threshold's meaning is never silently wrong.
DEFAULT_Z_THRESHOLD = 1.5

METRICS = ["n_docs", "avg_text_len", "avg_quality_score", "avg_language_score", "n_topic_clusters_present"]


def compute_shard_stats(shards_dir: str) -> list[dict]:
    dataset = ds.dataset(shards_dir, partitioning="hive")
    shards_root = Path(shards_dir)
    stats = []
    for fragment in dataset.get_fragments():
        table = fragment.to_table()
        n = table.num_rows
        text_lens = pc.utf8_length(table["text"]).to_numpy(zero_copy_only=False)
        quality = table["quality_score"].to_numpy(zero_copy_only=False)
        lang_score = table["language_score"].to_numpy(zero_copy_only=False)
        topic = table["topic_cluster"].to_numpy(zero_copy_only=False)

        shard_id = str(Path(fragment.path).relative_to(shards_root))
        stats.append(
            {
                "shard": shard_id,
                "n_docs": n,
                "avg_text_len": float(np.mean(text_lens)),
                "avg_quality_score": float(np.mean(quality)),
                "avg_language_score": float(np.mean(lang_score)),
                "n_topic_clusters_present": float(len(set(topic.tolist()))),
            }
        )
    return sorted(stats, key=lambda s: s["shard"])


def flag_anomalies(stats: list[dict], metrics: list[str], z_threshold: float) -> dict:
    results = {s["shard"]: {"flags": []} for s in stats}
    for metric in metrics:
        values = np.array([s[metric] for s in stats], dtype=float)
        mean = values.mean()
        std = values.std(ddof=0)
        for s, v in zip(stats, values):
            z = 0.0 if std == 0 else (v - mean) / std
            results[s["shard"]][metric] = {"value": float(v), "z": float(z)}
            if abs(z) > z_threshold:
                results[s["shard"]]["flags"].append(f"{metric} (z={z:+.2f})")
    return results


def main():
    parser = argparse.ArgumentParser(description="Flag anomalous Parquet shards via per-metric z-scores.")
    parser.add_argument("--shards-dir", default=str(DEFAULT_SHARDS_DIR))
    parser.add_argument("--z-threshold", type=float, default=DEFAULT_Z_THRESHOLD)
    args = parser.parse_args()

    stats = compute_shard_stats(args.shards_dir)
    print(f"=== batch_stats_detector.py: {len(stats)} shards found, z-threshold={args.z_threshold} ===")

    if len(stats) < 2:
        print("Need at least 2 shards to compute z-scores. Nothing to compare.")
        return

    z_ceiling = np.sqrt(len(stats) - 1)
    print(f"(population z-scores with n={len(stats)} shards can never exceed |z|={z_ceiling:.2f}, "
          f"regardless of how extreme a shard is — treat this as a smoke signal, not a verdict)")

    results = flag_anomalies(stats, METRICS, args.z_threshold)

    for s in stats:
        r = results[s["shard"]]
        tag = " <-- ANOMALY" if r["flags"] else ""
        print(f"\n{s['shard']}{tag}")
        for metric in METRICS:
            v = r[metric]["value"]
            z = r[metric]["z"]
            print(f"  {metric:<28} {v:>12.4f}  (z={z:+.2f})")
        for f in r["flags"]:
            print(f"    -> flagged: {f}")

    n_flagged = sum(1 for s in stats if results[s["shard"]]["flags"])
    print(f"\n{n_flagged}/{len(stats)} shard(s) flagged at |z| > {args.z_threshold}")


if __name__ == "__main__":
    main()
