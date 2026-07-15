import json
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Config:
    vocab_size = 256
    block_size = 128
    n_layer = 4
    n_head = 4
    n_embd = 160
    dropout = 0.0
    tie_weights = True

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

def main():
    cfg = Config()
    ckpt = torch.load("ckpt_run1.pt", map_location="cpu", weights_only=True)
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    text = open("../data/dev_eval.txt", encoding="utf-8").read()
    n_bytes = len(text.encode("utf-8"))
    ids = torch.tensor(list(text.encode("utf-8")), dtype=torch.long)
    block, stride = cfg.block_size, max(1, cfg.block_size // 2)
    
    total_nll, n_scored = 0.0, 0
    scored = 1
    
    with torch.no_grad():
        while scored < len(ids):
            start = max(0, scored - stride)
            end = min(len(ids), start + block)
            window = ids[start:end]
            logits, _ = model(window[None, :])
            logp = torch.log_softmax(logits[0], dim=-1)
            targets = ids[start + 1:end]
            nll = -logp[torch.arange(len(targets)), targets]
            offset = scored - (start + 1)
            total_nll += nll[offset:].sum().item()
            n_scored += len(nll) - offset
            scored = end
            
    bpb = total_nll / math.log(2) / n_bytes
    print(json.dumps({
        "bpb": round(bpb, 4),
        "n_params": model.n_params(),
        "steps": ckpt.get("steps")
    }))

if __name__ == "__main__":
    main()
