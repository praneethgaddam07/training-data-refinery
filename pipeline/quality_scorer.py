"""
Phase 1 - quality_scorer.py

Trains a lightweight TF-IDF + logistic regression classifier on real labels
produced by ingest_clean.py's heuristic filters:
  positive (1) = docs that passed Language/Gopher/C4 filtering (data/interim/clean/)
  negative (0) = docs that Gopher/C4 actually rejected (data/interim/dropped/{gopher,c4}/)

The classifier then scores every document in the deduped+clustered corpus with a
continuous quality_score in [0, 1] (P(kept)), which downstream steps (e.g. the
ablation's "deduped+filtered" variant) can threshold on.

Input:  data/interim/clean/               (positive class)
        data/interim/dropped/{gopher,c4}/ (negative class)
        data/interim/clustered/           (corpus to score)
Output: data/interim/scored/              (clustered docs + quality_score metadata)
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from stats_utils import write_stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CLEAN_DIR = REPO_ROOT / "data" / "interim" / "clean"
DEFAULT_DROPPED_DIRS = [
    REPO_ROOT / "data" / "interim" / "dropped" / "gopher",
    REPO_ROOT / "data" / "interim" / "dropped" / "c4",
]
DEFAULT_SCORE_INPUT_DIR = REPO_ROOT / "data" / "interim" / "clustered"
DEFAULT_SCORED_DIR = REPO_ROOT / "data" / "interim" / "scored"


def load_jsonl(folder) -> list[dict]:
    docs = []
    for f in sorted(Path(folder).glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                docs.append(json.loads(line))
    return docs


def train_classifier(clean_dir: str, dropped_dirs: list, random_state: int = 42):
    positives = load_jsonl(clean_dir)
    negatives = [doc for d in dropped_dirs for doc in load_jsonl(d)]

    texts = [d["text"] for d in positives] + [d["text"] for d in negatives]
    labels = [1] * len(positives) + [0] * len(negatives)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=random_state, stratify=labels
    )

    vectorizer = TfidfVectorizer(max_features=20000, stop_words="english", min_df=2)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    return vectorizer, clf, metrics


def score_corpus(vectorizer, clf, input_dir: str, scored_dir: str):
    docs = load_jsonl(input_dir)
    if not docs:
        raise RuntimeError(f"No documents found in {input_dir}")

    texts = [d["text"] for d in docs]
    X = vectorizer.transform(texts)
    scores = clf.predict_proba(X)[:, 1]

    for doc, score in zip(docs, scores):
        doc.setdefault("metadata", {})["quality_score"] = float(score)

    out_dir = Path(scored_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "00000.jsonl.gz"
    with gzip.open(out_path, "wt") as fh:
        for doc in docs:
            fh.write(json.dumps(doc) + "\n")

    return len(docs), np.array(scores)


def main():
    parser = argparse.ArgumentParser(description="Score a corpus with a lightweight TF-IDF quality classifier.")
    parser.add_argument("--clean-dir", default=str(DEFAULT_CLEAN_DIR))
    parser.add_argument("--dropped-dirs", nargs="+", default=[str(d) for d in DEFAULT_DROPPED_DIRS])
    parser.add_argument("--score-input-dir", default=str(DEFAULT_SCORE_INPUT_DIR))
    parser.add_argument("--scored-dir", default=str(DEFAULT_SCORED_DIR))
    args = parser.parse_args()

    print("=== Training quality classifier (TF-IDF + logistic regression) ===")
    vectorizer, clf, metrics = train_classifier(args.clean_dir, args.dropped_dirs)
    print(f"Positive examples (passed filters): {metrics['n_positive']}")
    print(f"Negative examples (rejected by Gopher/C4): {metrics['n_negative']}")
    print(f"Train/test split: {metrics['n_train']}/{metrics['n_test']}")
    print(f"Held-out accuracy: {metrics['accuracy']:.4f}")
    print(f"Held-out ROC-AUC:  {metrics['roc_auc']:.4f}")

    print("\n=== Scoring corpus ===")
    n_docs, scores = score_corpus(vectorizer, clf, args.score_input_dir, args.scored_dir)
    dist = {
        "min": float(scores.min()),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "max": float(scores.max()),
    }
    print(f"Scored {n_docs} docs")
    print(
        f"quality_score distribution: min={dist['min']:.3f} p25={dist['p25']:.3f} "
        f"median={dist['median']:.3f} p75={dist['p75']:.3f} max={dist['max']:.3f}"
    )
    print(f"Output written to: {args.scored_dir}")

    write_stats(
        "quality_scorer",
        {
            "n_positive": metrics["n_positive"],
            "n_negative": metrics["n_negative"],
            "n_train": metrics["n_train"],
            "n_test": metrics["n_test"],
            "held_out_accuracy": round(metrics["accuracy"], 4),
            "held_out_roc_auc": round(metrics["roc_auc"], 4),
            "n_scored": n_docs,
            "quality_score_distribution": dist,
        },
    )


if __name__ == "__main__":
    main()
