"""Optimized BPE tokenizer training.

Key optimization: Instead of scanning the entire byte stream for each merge,
we split text into "words" (chunks split on whitespace/newlines), count word
frequencies, and track pair counts at the word level. This reduces complexity
from O(corpus_size × num_merges) to O(unique_words × avg_word_len × num_merges).

For a 7.3MB corpus with ~500K unique words, this is ~100x faster than naive BPE.

Usage:
    python train_bpe.py --data ../data/train_corpus.txt --vocab_size 1024
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter

# Fix Windows console encoding for Hindi/Devanagari output
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def get_word_freqs(text):
    """Split text into words and count frequencies.
    Words are split on whitespace but keep the whitespace as a prefix
    of the following word (GPT-2 style pre-tokenization).
    """
    # Split into chunks: each chunk is a sequence of non-space chars,
    # optionally preceded by spaces. This preserves whitespace info.
    # We also split on newlines to keep words manageable.
    chunks = re.findall(rb'[^\S\n]*\S+|\n', text)
    freqs = Counter(chunks)
    return freqs


def train_bpe(text_bytes, num_merges, verbose=True):
    """Train BPE using word-frequency-weighted pair counting."""
    t0 = time.time()

    # Step 1: Build word frequency table
    word_freqs = get_word_freqs(text_bytes)
    if verbose:
        print(f"  Pre-tokenized into {len(word_freqs):,} unique words "
              f"({sum(word_freqs.values()):,} total) in {time.time()-t0:.1f}s")

    # Convert words to tuple-of-ints for BPE processing
    # word_splits[word_bytes] = list of token ids (initially byte values)
    word_splits = {}
    for word_bytes, freq in word_freqs.items():
        word_splits[word_bytes] = list(word_bytes)  # bytes are already ints

    # Step 2: Count all pairs weighted by word frequency
    def count_pairs():
        pairs = Counter()
        for word_bytes, tokens in word_splits.items():
            freq = word_freqs[word_bytes]
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i+1])] += freq
        return pairs

    merges = []
    vocab = {i: bytes([i]) for i in range(256)}

    pair_counts = count_pairs()

    for merge_i in range(num_merges):
        if not pair_counts:
            if verbose:
                print(f"  No more pairs at merge {merge_i}")
            break

        # Find most frequent pair
        best_pair = pair_counts.most_common(1)[0]
        pair, count = best_pair
        if count < 2:
            # Not worth merging singletons
            if verbose:
                print(f"  Stopping at merge {merge_i}: best pair count = {count}")
            break

        new_id = 256 + merge_i
        a, b = pair

        # Merge this pair in all words and update pair counts incrementally
        new_word_splits = {}
        for word_bytes, tokens in word_splits.items():
            if len(tokens) < 2:
                new_word_splits[word_bytes] = tokens
                continue

            freq = word_freqs[word_bytes]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == a
                        and tokens[i+1] == b):
                    # Before merging, decrement pair counts for affected neighbors
                    if i > 0:
                        old_left = (tokens[i-1] if i == len(new_tokens) or not new_tokens
                                    else new_tokens[-1], a)
                        # Actually we need to handle this more carefully
                        pass
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word_splits[word_bytes] = new_tokens

        word_splits = new_word_splits
        merges.append((a, b))
        vocab[new_id] = vocab[a] + vocab[b]

        # Recount pairs (simpler than incremental updates, still fast
        # because we iterate over unique words not corpus)
        pair_counts = count_pairs()

        if verbose and ((merge_i + 1) % 100 == 0 or merge_i < 3
                        or merge_i == num_merges - 1):
            try:
                decoded = vocab[new_id].decode('utf-8')
                text_repr = repr(decoded)
            except UnicodeDecodeError:
                text_repr = repr(vocab[new_id])
            elapsed = time.time() - t0
            print(f"  merge {merge_i+1:4d}/{num_merges}  "
                  f"pair=({a:4d},{b:4d})  count={count:>7,}  "
                  f"text={text_repr:25s}  ({elapsed:.1f}s)")

    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "bpe_merges.json")

    print(f"Loading corpus from {args.data}...")
    text = open(args.data, encoding="utf-8").read()
    
    # Fast BPE: The corpus is 7.3MB, mostly English, with Hindi at the end.
    # Training pure-python BPE on 7.3MB takes too long for the speedrun.
    # We sample 1MB of English from the start and all of the Hindi from the end.
    sample_text = text[:1000000] + text[-350000:]
    text_bytes = sample_text.encode("utf-8")
    
    print(f"Full Corpus: {len(text.encode('utf-8')):,} bytes")
    print(f"Sampled for BPE: {len(text_bytes):,} bytes")

    num_merges = args.vocab_size - 256
    print(f"Training BPE: {num_merges} merges -> vocab {args.vocab_size}")
    t0 = time.time()
    merges = train_bpe(text_bytes, num_merges)
    print(f"BPE training done in {time.time()-t0:.1f}s, {len(merges)} merges")

    # Save
    with open(args.out, "w") as f:
        json.dump({"merges": merges, "vocab_size": 256 + len(merges)}, f)
    print(f"Saved to {args.out}")

    # Verify roundtrip
    print("\nVerifying lossless roundtrip...")
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tokenizer import load
    tok = load(args.out)

    # Test on beginning (English)
    sample_en = text[:5000]
    enc = tok.encode(sample_en)
    dec = tok.decode(enc)
    assert dec == sample_en, f"English roundtrip FAILED!"
    print(f"  English sample: {len(sample_en.encode('utf-8')):,} bytes → "
          f"{len(enc):,} tokens ({len(sample_en.encode('utf-8'))/len(enc):.2f}x)")

    # Test on end (Hindi)
    sample_hi = text[-5000:]
    enc_hi = tok.encode(sample_hi)
    dec_hi = tok.decode(enc_hi)
    assert dec_hi == sample_hi, "Hindi roundtrip FAILED!"
    print(f"  Hindi sample: {len(sample_hi.encode('utf-8')):,} bytes → "
          f"{len(enc_hi):,} tokens ({len(sample_hi.encode('utf-8'))/len(enc_hi):.2f}x)")

    # Full corpus
    full_enc = tok.encode(text)
    full_dec = tok.decode(full_enc)
    assert full_dec == text, "FULL ROUNDTRIP FAILED!"
    print(f"  Full corpus: {len(text_bytes):,} bytes → "
          f"{len(full_enc):,} tokens ({len(text_bytes)/len(full_enc):.2f}x)")


if __name__ == "__main__":
    main()
