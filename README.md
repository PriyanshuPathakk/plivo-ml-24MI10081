# LLM Speedrun Challenge

This repository contains my submission for the 2,000 Step LLM Speedrun challenge.

## Objective
The goal was to take a mediocre baseline GPT-style language model and optimize it to achieve the lowest possible bits-per-byte (bpb) score on a held-out dataset, subject to strict hardware and budget constraints:
- Maximum 2,000,000 parameters
- Maximum 2,000 training steps
- Training data restricted to a 7MB mixed English/Hindi corpus (`train_corpus.txt`)
- Pure PyTorch/numpy only (no compiled C-extensions like FlashAttention or external tokenizer libraries)

## Approach
Instead of blindly throwing architectural changes at the model, I started by auditing the optimization pipeline. The baseline model was severely handicapped by its training recipe (using a constant learning rate, basic Adam, and no gradient accumulation). 

By implementing a modern training loop (Cosine decay with warmup, AdamW, and gradient accumulation for a larger effective batch size), I was able to drop the evaluation bpb from the baseline's **2.37** down to **2.08** while staying well under the parameter limit (~1.3M params).

I attempted to build a pure Python BPE tokenizer to better handle the Hindi text (where characters are 3 bytes in UTF-8), but the O(N^2) complexity of Python loops on a 7MB corpus proved intractable within the strict 120-minute speedrun limit. I documented this failure mode in the `RUNLOG.md` and pivoted back to the byte-level tokenizer to guarantee a working, verifiable submission on time.

## How to Evaluate
You can evaluate the submitted checkpoint (`ckpt.pt`) using the official evaluation script:
```bash
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```
