"""Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
unseen text — and treats a Devanagari character as 3 tokens. Think about
what that does to your model's context window and your token budget on the
Hindi part of the corpus.
"""
import json

class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)

def load(path=None):
    """Return the tokenizer used by evaluate.py."""
    return ByteTokenizer()
