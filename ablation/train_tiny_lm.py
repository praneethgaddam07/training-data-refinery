"""
Phase 3 - train_tiny_lm.py

A minimal nanoGPT-style causal transformer (tied embeddings, pre-LN blocks,
GPT-2 BPE via tiktoken), trained on the MPS backend. compare_mixes.py calls
train_variant() 3 times, once per data mix, with identical model config and
step/batch/block-size (== identical token budget) so the only thing that
differs between runs is the data.
"""

import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying (nanoGPT-style)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

_ENC = tiktoken.get_encoding("gpt2")
EOT = _ENC.eot_token


def load_jsonl_texts(jsonl_dir: str) -> list[str]:
    texts = []
    for f in sorted(Path(jsonl_dir).glob("*.jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                texts.append(json.loads(line)["text"])
    return texts


def tokenize_corpus(jsonl_dir: str, seed: int = 1234) -> np.ndarray:
    """Loads all docs, shuffles them (fixed seed), concatenates as
    doc1<|endoftext|>doc2<|endoftext|>..., returns a uint16 token array."""
    texts = load_jsonl_texts(jsonl_dir)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(texts))
    chunks = []
    for i in order:
        chunks.append(np.array(_ENC.encode_ordinary(texts[i]), dtype=np.uint16))
        chunks.append(np.array([EOT], dtype=np.uint16))
    return np.concatenate(chunks)


def train_val_split(tokens: np.ndarray, val_fraction: float = 0.1):
    split = int(len(tokens) * (1 - val_fraction))
    return tokens[:split], tokens[split:]


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str, generator: torch.Generator):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n_steps: int = 600
    batch_size: int = 32
    # 128, not 256: a quick MPS throughput sweep found block_size=256 at batch=64 hit a
    # pathological slowdown (22s/step vs. 0.4s/step at block=128, batch=32) on this backend
    # for basically the same tokens/step throughput — 128 keeps each run to a few minutes.
    block_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    seed: int = 1337
    device: str = None


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def estimate_loss(model, data, cfg: TrainConfig, generator):
    model.eval()
    losses = []
    for _ in range(cfg.eval_iters):
        x, y = get_batch(data, cfg.block_size, cfg.batch_size, cfg.device, generator)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train_variant(
    name: str,
    data_dir: str,
    model_cfg: GPTConfig,
    train_cfg: TrainConfig,
    shared_eval_data: np.ndarray = None,
) -> dict:
    device = train_cfg.device or get_device()
    train_cfg.device = device
    model_cfg.block_size = train_cfg.block_size

    print(f"\n--- variant: {name} ({data_dir}) ---")
    t0 = time.time()
    tokens = tokenize_corpus(data_dir)
    train_data, val_data = train_val_split(tokens, val_fraction=0.1)
    print(
        f"tokenized in {time.time() - t0:.1f}s: {len(tokens):,} tokens total "
        f"({len(train_data):,} train / {len(val_data):,} val)"
    )

    torch.manual_seed(train_cfg.seed)
    model = GPT(model_cfg).to(device)
    print(f"model params: {model.num_params():,}  device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    generator = torch.Generator().manual_seed(train_cfg.seed)

    history = []
    t0 = time.time()
    for step in range(1, train_cfg.n_steps + 1):
        x, y = get_batch(train_data, train_cfg.block_size, train_cfg.batch_size, device, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.n_steps:
            val_loss = estimate_loss(model, val_data, train_cfg, generator)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss})
            print(f"  step {step:>5}/{train_cfg.n_steps}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")
    elapsed = time.time() - t0

    tokens_processed = train_cfg.n_steps * train_cfg.batch_size * train_cfg.block_size
    print(f"done in {elapsed:.1f}s ({tokens_processed:,} training tokens processed)")

    shared_eval_loss = None
    if shared_eval_data is not None:
        # more eval_iters than the in-training checks: this number is the headline metric
        eval_cfg = TrainConfig(**{**train_cfg.__dict__, "eval_iters": 50})
        shared_eval_loss = estimate_loss(model, shared_eval_data, eval_cfg, generator)
        print(f"shared held-out eval loss (external, uncontaminated benchmark): {shared_eval_loss:.4f}")

    return {
        "variant": name,
        "n_tokens_total": int(len(tokens)),
        "n_train_tokens": int(len(train_data)),
        "n_val_tokens": int(len(val_data)),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "shared_eval_loss": shared_eval_loss,
        "history": history,
        "elapsed_sec": elapsed,
        "tokens_processed": tokens_processed,
        "model_params": model.num_params(),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train a single tiny GPT on one data variant.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--name", default="run")
    parser.add_argument("--n-steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()

    model_cfg = GPTConfig(block_size=args.block_size)
    train_cfg = TrainConfig(n_steps=args.n_steps, batch_size=args.batch_size, block_size=args.block_size)
    result = train_variant(args.name, args.data_dir, model_cfg, train_cfg)
    print({k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
