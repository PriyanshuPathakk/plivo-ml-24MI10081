"""Run 1: Training improvements only, original byte-level architecture.
Isolates impact of training recipe from architecture changes.
"""
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    """Original baseline architecture — unchanged."""
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True  # only change: enable weight tying


class SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init)
        # Scale residual projections
        for blk in self.blocks:
            nn.init.normal_(blk.attn.proj.weight, std=0.02/math.sqrt(2*cfg.n_layer))
            nn.init.normal_(blk.mlp[2].weight, std=0.02/math.sqrt(2*cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def cosine_lr(step, warmup, total, lr_max, lr_min):
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def main():
    torch.manual_seed(42)
    text = open("../data/train_corpus.txt", encoding="utf-8").read()
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    print(f"corpus: {len(ids):,} tokens")

    cfg = Config()
    model = GPT(cfg)
    print(f"model: {model.n_params():,} params")

    # Training improvements: AdamW + cosine LR + grad clip + grad accum
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=6e-4, betas=(0.9, 0.95))

    steps, warmup, batch, accum = 2000, 200, 16, 4
    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        lr = cosine_lr(step, warmup, steps, 6e-4, 6e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr
        total_loss = 0.0
        for _ in range(accum):
            x, y = get_batch(ids, cfg.block_size, batch, "cpu")
            _, loss = model(x, y)
            (loss / accum).backward()
            total_loss += loss.item() / accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total_loss)
        if step % 200 == 0 or step == 1:
            print(f"step {step:5d}  loss {sum(losses[-200:])/len(losses[-200:]):.4f}  "
                  f"lr {lr:.2e}  ({(time.time()-t0)/step*1000:.0f}ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": steps, "train_loss_curve": losses}, "ckpt_run1.pt")
    print(f"done in {time.time()-t0:.0f}s, final loss {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
