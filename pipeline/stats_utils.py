"""Tiny helper: persist a pipeline stage's run summary as JSON for the dashboard to read."""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATS_DIR = REPO_ROOT / "data" / "stats"


def write_stats(stage: str, stats: dict, stats_dir: str = None):
    out_dir = Path(stats_dir) if stats_dir else DEFAULT_STATS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{stage}.json", "w") as f:
        json.dump(stats, f, indent=2)
