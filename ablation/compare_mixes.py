"""
Phase 3 - compare_mixes.py

Trains identical tiny GPT models on 3 data variants at equal token budget
(same n_steps, batch_size, block_size, model config, seed) and compares
validation loss. This is the experiment that shows whether the pipeline's
cleaning actually helps model quality, holding compute fixed:

  raw              - data/ablation/raw/     (unfiltered, undeduped WET text)
  deduped          - data/ablation/deduped/ (MinHash deduped, still unfiltered)
  deduped_filtered - data/interim/deduped/  (language+Gopher+C4 filtered, then deduped)

Each variant's *own* held-out val loss is not a fair cross-variant comparison:
raw/deduped's held-out text is full of repetitive boilerplate (trivially
predictable, deflates loss), while deduped_filtered's is genuine diverse prose
(inherently higher-entropy). All three variants are also derived from the same
WET file, so using deduped_filtered's val split as a "shared" benchmark for
raw/deduped would leak (raw/deduped's training pool structurally contains those
same source documents). So we additionally score every trained model on a
genuinely external, uncontaminated benchmark: data/ablation/eval_clean/, built
by running a *second*, different WET file through the same ingest_clean.py
filters. No variant has seen any content from it. This shared_eval_loss is the
headline comparison.
"""

import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ablation.prepare_variants import DEDUPED_VARIANT_DIR, RAW_VARIANT_DIR
from ablation.train_tiny_lm import GPTConfig, TrainConfig, tokenize_corpus, train_variant
from pipeline.dedup_cluster import DEFAULT_DEDUPED_DIR as DEDUPED_FILTERED_DIR
from pipeline.dedup_cluster import count_jsonl_docs

RESULTS_PATH = REPO_ROOT / "data" / "ablation" / "results.json"
SHARED_EVAL_DIR = REPO_ROOT / "data" / "ablation" / "eval_clean"

VARIANTS = [
    ("raw", RAW_VARIANT_DIR),
    ("deduped", DEDUPED_VARIANT_DIR),
    ("deduped_filtered", DEDUPED_FILTERED_DIR),
]


def main():
    model_cfg = GPTConfig(n_layer=4, n_head=4, n_embd=256, block_size=128)
    train_cfg = TrainConfig(n_steps=600, batch_size=32, block_size=128, seed=1337)

    print("=== ablation: raw vs. deduped vs. deduped+filtered, equal token budget ===")
    print(f"model config: {model_cfg}")
    print(f"train config: {train_cfg}")
    print(f"token budget: {train_cfg.n_steps * train_cfg.batch_size * train_cfg.block_size:,} tokens/run")

    shared_eval_tokens = tokenize_corpus(str(SHARED_EVAL_DIR))
    print(f"shared external eval benchmark ({SHARED_EVAL_DIR}): {len(shared_eval_tokens):,} tokens\n")

    results = []
    for name, data_dir in VARIANTS:
        n_docs = count_jsonl_docs(str(data_dir))
        # fresh dataclass copies: train_variant mutates train_cfg.device / model_cfg.block_size in place
        result = train_variant(
            name,
            str(data_dir),
            dataclasses.replace(model_cfg),
            dataclasses.replace(train_cfg),
            shared_eval_data=shared_eval_tokens,
        )
        result["n_docs"] = n_docs
        results.append(result)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== ablation summary (equal token budget, equal steps, equal model config) ===")
    header = f"{'variant':<20} {'docs':>8} {'tokens':>12} {'train_loss':>12} {'own_val_loss':>13} {'shared_eval_loss':>17}"
    print(header)
    for r in results:
        print(
            f"{r['variant']:<20} {r['n_docs']:>8,} {r['n_tokens_total']:>12,} "
            f"{r['final_train_loss']:>12.4f} {r['final_val_loss']:>13.4f} {r['shared_eval_loss']:>17.4f}"
        )

    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
