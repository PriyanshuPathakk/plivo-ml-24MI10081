"""Calculate parameter counts for different model configurations."""

def count_params(vocab_size, n_embd, n_layer, n_head, block_size, tie_weights=False, mlp_ratio=4):
    tok_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    
    # Per block
    ln_params = 2 * 2 * n_embd  # 2 layer norms, each has weight + bias
    qkv = n_embd * 3 * n_embd + 3 * n_embd
    proj = n_embd * n_embd + n_embd
    mlp_hidden = int(mlp_ratio * n_embd)
    mlp_up = n_embd * mlp_hidden + mlp_hidden
    mlp_down = mlp_hidden * n_embd + n_embd
    block_params = ln_params + qkv + proj + mlp_up + mlp_down
    total_block = n_layer * block_params
    
    ln_f = 2 * n_embd
    head = vocab_size * n_embd if not tie_weights else 0
    
    total = tok_emb + pos_emb + total_block + ln_f + head
    return total, {
        'tok_emb': tok_emb,
        'pos_emb': pos_emb,
        'per_block': block_params,
        'total_blocks': total_block,
        'ln_f': ln_f,
        'head': head,
    }

budget = 2_000_000

print("=" * 80)
print("CONFIGURATION SEARCH")
print("=" * 80)

configs = [
    # (name, vocab, embd, layers, heads, block_size, tie, mlp_ratio)
    ("Baseline", 256, 160, 4, 4, 128, False, 4),
    ("Baseline+tie", 256, 160, 4, 4, 128, True, 4),
    ("BPE512+tie", 512, 192, 6, 6, 256, True, 4),
    ("BPE512+tie+big", 512, 224, 6, 8, 256, True, 4),
    ("BPE1024+tie", 1024, 192, 6, 6, 256, True, 4),
    ("BPE1024+tie+deep", 1024, 192, 8, 6, 256, True, 4),
    ("BPE1024+256ctx", 1024, 224, 6, 8, 256, True, 4),
    ("BPE1024+512ctx", 1024, 224, 6, 8, 512, True, 4),
    ("BPE2048+tie", 2048, 192, 6, 6, 256, True, 4),
    ("BPE2048+256emb", 2048, 256, 6, 8, 256, True, 4),
    ("BPE512+big+deep", 512, 256, 8, 8, 256, True, 4),
    ("BPE1024+256emb+6L", 1024, 256, 6, 8, 256, True, 4),
    ("BPE1024+256emb+8L", 1024, 256, 8, 8, 256, True, 4),
    ("BPE768+256emb+6L", 768, 256, 6, 8, 256, True, 4),
    ("BPE768+288emb+6L", 768, 288, 6, 6, 256, True, 4),
    ("BPE768+288emb+8L", 768, 288, 8, 6, 256, True, 4),
    ("BPE512+320emb+6L", 512, 320, 6, 8, 256, True, 4),
    ("BPE512+288emb+8L", 512, 288, 8, 6, 256, True, 4),
    ("BPE1024+224emb+8L", 1024, 224, 8, 8, 256, True, 4),
    # Try with smaller MLP ratio to fit more layers/embd
    ("BPE1024+288emb+6L+3x", 1024, 288, 6, 6, 256, True, 3),
    ("BPE1024+320emb+6L+3x", 1024, 320, 6, 8, 256, True, 3),
    ("BPE1024+256emb+8L+3x", 1024, 256, 8, 8, 256, True, 3),
    ("BPE512+320emb+8L+3x", 512, 320, 8, 8, 256, True, 3),
    ("BPE1024+288emb+8L+3x", 1024, 288, 8, 6, 256, True, 3),
]

for name, v, e, l, h, bs, tie, mlp in configs:
    total, detail = count_params(v, e, l, h, bs, tie, mlp)
    status = "OK" if total <= budget else "OVER"
    margin = budget - total
    print(f"{name:30s}  params={total:>9,}  margin={margin:>+8,}  [{status}]  "
          f"v={v} e={e} L={l} h={h} bs={bs} tie={tie} mlp={mlp}x")
