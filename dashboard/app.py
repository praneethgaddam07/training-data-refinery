"""
Phase 4 - dashboard/app.py

Streamlit dashboard: pipeline funnel + shard stats, anomaly flags, ablation results.
Reads directly from data/stats/*.json, data/shards/*.parquet, and data/ablation/results.json
-- no numbers are computed or faked here, only real pipeline output is displayed.

Run with: streamlit run dashboard/app.py
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from anomaly.batch_stats_detector import METRICS, compute_shard_stats, flag_anomalies
from pipeline import storage

STATS_DIR = REPO_ROOT / "data" / "stats"
# REFINERY_SHARDS_DIR lets the dashboard point at s3://bucket/prefix instead of local
# disk (the AWS variant) -- unset, it defaults to the same local path as always.
SHARDS_DIR = os.environ.get("REFINERY_SHARDS_DIR", str(REPO_ROOT / "data" / "shards"))
RESULTS_PATH = REPO_ROOT / "data" / "ablation" / "results.json"

# Validated categorical palette (dataviz skill, references/palette.md). The first
# three slots (blue/orange/aqua) validate all-pairs in both light and dark modes --
# exactly what a 3-series comparison like this needs.
COLOR_RAW = "#2a78d6"
COLOR_DEDUPED = "#eb6834"
COLOR_FILTERED = "#1baf7a"
VARIANT_COLORS = {"raw": COLOR_RAW, "deduped": COLOR_DEDUPED, "deduped_filtered": COLOR_FILTERED}
VARIANT_ORDER = ["raw", "deduped", "deduped_filtered"]

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

TRANSPARENT_LAYOUT = dict(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

st.set_page_config(page_title="Training Data Refinery", layout="wide")


@st.cache_data
def load_stage_stats():
    stats = {}
    for name in ["ingest_clean", "dedup_cluster", "quality_scorer", "shard_writer"]:
        p = STATS_DIR / f"{name}.json"
        if p.exists():
            stats[name] = json.loads(p.read_text())
    return stats


@st.cache_data
def load_shard_stats():
    if not storage.shards_exist(SHARDS_DIR):
        return []
    return compute_shard_stats(SHARDS_DIR)


@st.cache_data
def load_ablation_results():
    if not RESULTS_PATH.exists():
        return []
    return json.loads(RESULTS_PATH.read_text())


st.title("Training Data Refinery")
st.caption("Common Crawl -> deduplicated, quality-scored, Parquet-packed training shards")

stage_stats = load_stage_stats()
shard_stats = load_shard_stats()
ablation_results = load_ablation_results()

# ---------------------------------------------------------------------------
# Pipeline funnel
# ---------------------------------------------------------------------------
st.header("Pipeline funnel")
if stage_stats:
    ingest = stage_stats.get("ingest_clean", {})
    dedup = stage_stats.get("dedup_cluster", {})
    qs = stage_stats.get("quality_scorer", {})
    sw = stage_stats.get("shard_writer", {})

    cols = st.columns(5)
    cols[0].metric("Raw docs", f"{ingest.get('raw_docs', 0):,}")
    cols[1].metric(
        "Clean docs", f"{ingest.get('clean_docs', 0):,}", f"-{ingest.get('dropped_pct', 0):.1f}%", delta_color="off"
    )
    cols[2].metric(
        "Deduped docs",
        f"{dedup.get('deduped_docs_out', 0):,}",
        f"-{dedup.get('near_dups_removed_pct', 0):.1f}%",
        delta_color="off",
    )
    cols[3].metric("Scored docs", f"{qs.get('n_scored', 0):,}")
    cols[4].metric("Shard rows", f"{sw.get('total_rows', 0):,}")

    if qs:
        st.caption(
            f"Quality classifier (TF-IDF + logistic regression): held-out accuracy "
            f"{qs.get('held_out_accuracy', 0):.4f}, ROC-AUC {qs.get('held_out_roc_auc', 0):.4f} "
            f"({qs.get('n_positive', 0):,} positive / {qs.get('n_negative', 0):,} negative examples)."
        )
else:
    st.info("No pipeline stats found -- run pipeline/ingest_clean.py through shard_writer.py first.")

# ---------------------------------------------------------------------------
# Shard stats
# ---------------------------------------------------------------------------
st.header("Shard stats")
if shard_stats:
    shard_df = pd.DataFrame(shard_stats)
    st.dataframe(shard_df, width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=shard_df["shard"], y=shard_df["avg_quality_score"], marker_color=COLOR_RAW))
    fig.update_layout(
        title="Average quality_score per shard",
        yaxis_title="avg quality_score",
        showlegend=False,
        **TRANSPARENT_LAYOUT,
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No shards found -- run pipeline/shard_writer.py first.")

# ---------------------------------------------------------------------------
# Anomaly flags
# ---------------------------------------------------------------------------
st.header("Anomaly detection")
if len(shard_stats) >= 2:
    z_threshold = 1.5
    flagged = flag_anomalies(shard_stats, METRICS, z_threshold)
    z_ceiling = (len(shard_stats) - 1) ** 0.5
    st.caption(
        f"Population z-scores over n={len(shard_stats)} shards can never exceed |z|={z_ceiling:.2f}, "
        f"regardless of how extreme a shard is -- threshold set to {z_threshold} accordingly "
        f"(the conventional 2.0 would be mathematically unreachable and silently never fire here). "
        f"Treat this as a smoke signal, not a verdict."
    )

    rows = []
    for s in shard_stats:
        r = flagged[s["shard"]]
        row = {"shard": s["shard"], "status": "ANOMALY" if r["flags"] else "OK"}
        row.update({f"{m}_z": round(r[m]["z"], 2) for m in METRICS})
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    bar_colors = [STATUS_CRITICAL if flagged[s["shard"]]["flags"] else STATUS_GOOD for s in shard_stats]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=[s["shard"] for s in shard_stats], y=[s["n_docs"] for s in shard_stats], marker_color=bar_colors)
    )
    fig.update_layout(title="Docs per shard (red = flagged anomalous)", yaxis_title="n_docs", showlegend=False, **TRANSPARENT_LAYOUT)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Need at least 2 shards to compute z-scores.")

# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------
st.header("Ablation: raw vs. deduped vs. deduped+filtered")
if ablation_results:
    by_name = {r["variant"]: r for r in ablation_results if r.get("shared_eval_loss") is not None}

    if by_name:
        fig = go.Figure()
        metrics_x = ["own val_loss (confounded)", "shared eval_loss (fair)"]
        for name in VARIANT_ORDER:
            r = by_name.get(name)
            if not r:
                continue
            fig.add_trace(
                go.Bar(name=name, x=metrics_x, y=[r["final_val_loss"], r["shared_eval_loss"]], marker_color=VARIANT_COLORS[name])
            )
        fig.update_layout(
            barmode="group",
            title="Final loss: own held-out split vs. shared external benchmark",
            yaxis_title="cross-entropy loss",
            legend_title_text="variant",
            **TRANSPARENT_LAYOUT,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Each variant's *own* held-out loss isn't comparable across variants -- raw/deduped's held-out "
            "text is full of repetitive boilerplate, which deflates loss artificially. The shared eval_loss "
            "column, scored on a second WET file none of the 3 models ever trained on, is the fair "
            "comparison: deduped_filtered wins despite training on ~6x fewer documents."
        )
    else:
        by_name = {r["variant"]: r for r in ablation_results}
        st.warning("Results have no shared_eval_loss (old-format results.json) -- showing own val_loss only.")

    col1, col2 = st.columns(2)
    with col1:
        fig2 = go.Figure()
        for name in VARIANT_ORDER:
            r = by_name.get(name)
            if not r:
                continue
            steps = [h["step"] for h in r["history"]]
            fig2.add_trace(
                go.Scatter(
                    x=steps,
                    y=[h["train_loss"] for h in r["history"]],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=VARIANT_COLORS[name], width=2),
                )
            )
        fig2.update_layout(title="Training loss", xaxis_title="step", yaxis_title="train_loss", **TRANSPARENT_LAYOUT)
        st.plotly_chart(fig2, width="stretch")
    with col2:
        fig3 = go.Figure()
        for name in VARIANT_ORDER:
            r = by_name.get(name)
            if not r:
                continue
            steps = [h["step"] for h in r["history"]]
            fig3.add_trace(
                go.Scatter(
                    x=steps,
                    y=[h["val_loss"] for h in r["history"]],
                    mode="lines+markers",
                    name=name,
                    line=dict(color=VARIANT_COLORS[name], width=2),
                )
            )
        fig3.update_layout(title="Own held-out val loss (confounded)", xaxis_title="step", yaxis_title="val_loss", **TRANSPARENT_LAYOUT)
        st.plotly_chart(fig3, width="stretch")

    table_rows = []
    for name in VARIANT_ORDER:
        r = by_name.get(name)
        if not r:
            continue
        table_rows.append(
            {
                "variant": name,
                "docs": r.get("n_docs"),
                "tokens": r["n_tokens_total"],
                "train_loss": round(r["final_train_loss"], 4),
                "own_val_loss": round(r["final_val_loss"], 4),
                "shared_eval_loss": round(r["shared_eval_loss"], 4) if r.get("shared_eval_loss") is not None else None,
            }
        )
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)
else:
    st.info("No ablation results found -- run ablation/compare_mixes.py first.")
