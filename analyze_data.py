"""Quick analysis of training data."""
import os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train_corpus.txt')
text = open(data_path, encoding='utf-8').read()

total_chars = len(text)
total_bytes = len(text.encode('utf-8'))
print(f"Total chars: {total_chars:,}")
print(f"Total bytes: {total_bytes:,}")
print(f"Bytes/char ratio: {total_bytes/total_chars:.2f}")

lines = text.split('\n')
print(f"Lines: {len(lines):,}")

eng = sum(1 for c in text if ord(c) < 128)
hindi = sum(1 for c in text if 0x0900 <= ord(c) <= 0x097F)
print(f"ASCII chars: {eng:,} ({100*eng/total_chars:.1f}%)")
print(f"Devanagari chars: {hindi:,} ({100*hindi/total_chars:.1f}%)")

# Byte-level token count
byte_tokens = total_bytes
print(f"\nByte-level tokens: {byte_tokens:,}")
print(f"With block_size=128, approx sequences: {byte_tokens//128:,}")

# Check unique byte values
byte_data = text.encode('utf-8')
unique_bytes = len(set(byte_data))
print(f"Unique byte values: {unique_bytes}")

# Sample text
print("\n--- First 300 chars ---")
print(text[:300])
print("\n--- Sample from middle ---")
mid = len(text) // 2
print(text[mid:mid+300])

# Count parameters for baseline
print("\n--- Baseline parameter count ---")
vocab = 256
n_embd = 160
n_layer = 4
n_head = 4
block_size = 128

tok_emb = vocab * n_embd
pos_emb = block_size * n_embd
# Per block: ln1, attn(qkv + proj), ln2, mlp(up + down)
ln_params = 2 * n_embd  # weight + bias per LN
qkv = n_embd * 3 * n_embd + 3 * n_embd
proj = n_embd * n_embd + n_embd
mlp_up = n_embd * 4 * n_embd + 4 * n_embd
mlp_down = 4 * n_embd * n_embd + n_embd
block_params = 2 * ln_params + qkv + proj + mlp_up + mlp_down
total_block = n_layer * block_params
ln_f = 2 * n_embd
head = vocab * n_embd  # no bias

total = tok_emb + pos_emb + total_block + ln_f + head
print(f"tok_emb: {tok_emb:,}")
print(f"pos_emb: {pos_emb:,}")
print(f"per block: {block_params:,}")
print(f"total blocks ({n_layer}): {total_block:,}")
print(f"ln_f: {ln_f:,}")
print(f"head: {head:,}")
print(f"TOTAL: {total:,}")
print(f"Budget remaining: {2_000_000 - total:,}")
