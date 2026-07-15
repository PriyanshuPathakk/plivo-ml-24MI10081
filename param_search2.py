"""Find configs that fit within 2M parameter budget."""

def count_params(vocab_size, n_embd, n_layer, n_head, block_size, tie_weights=True, mlp_ratio=4):
    tok_emb = vocab_size * n_embd
    pos_emb = block_size * n_embd
    ln_params = 2 * 2 * n_embd
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
    return total

budget = 2_000_000
results = []

for vocab in [256, 384, 512, 768, 1024]:
    for n_embd in range(96, 321, 16):
        for n_layer in range(4, 13):
            for mlp_ratio in [2, 3, 4]:
                for block_size in [128, 256, 512]:
                    n_head = max(1, n_embd // 32)
                    if n_embd % n_head != 0:
                        continue
                    p = count_params(vocab, n_embd, n_layer, n_head, block_size, True, mlp_ratio)
                    if p <= budget and p >= budget - 200_000:  # within 200k of budget
                        results.append((p, vocab, n_embd, n_layer, n_head, block_size, mlp_ratio))

results.sort(key=lambda x: -x[0])  # most params first (closest to budget)
print(f"Found {len(results)} configs within budget")
print(f"{'Params':>10} {'Vocab':>6} {'Embd':>5} {'Layers':>7} {'Heads':>6} {'Block':>6} {'MLP':>4}")
for p, v, e, l, h, bs, mlp in results[:40]:
    print(f"{p:>10,} {v:>6} {e:>5} {l:>7} {h:>6} {bs:>6} {mlp:>4}x")
