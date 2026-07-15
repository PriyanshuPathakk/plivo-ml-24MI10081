"""Run all experiments sequentially after BPE is trained.
Each experiment trains, evaluates, and logs results.

This script runs ALL experiments in sequence:
  Run 2: BPE + training improvements (original arch)
  Run 3: Full architecture swap (RoPE + RMSNorm + SwiGLU)
  Run 4: AMBITIOUS - Deep narrow model (expected to struggle)
  Run 5: AMBITIOUS - Lion optimizer (novel optimizer)
  Run 6: Ablation - GELU vs SwiGLU
  Run 7: Final tuned model
"""
import subprocess
import sys
import json
import time
import os

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

EVAL_CMD = [sys.executable, "evaluate.py", "--text_file", "../data/dev_eval.txt"]
TRAIN_CMD = [sys.executable, "train.py", "--data", "../data/train_corpus.txt", "--steps", "2000"]

def run_and_capture(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    elapsed = time.time() - t0
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])
    print(f"[{desc}] completed in {elapsed:.0f}s, exit code {result.returncode}")
    return result

def evaluate(ckpt_name):
    result = subprocess.run(
        EVAL_CMD + ["--checkpoint", ckpt_name],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    print(f"  Eval output: {result.stdout.strip()}")
    if result.returncode != 0:
        print(f"  Eval FAILED: {result.stderr[-300:]}")
        return None
    try:
        return json.loads(result.stdout.strip())
    except:
        print(f"  Could not parse eval output")
        return None

results = {}

# ============================================================
# Run 2: Training improvements, ORIGINAL architecture
# (To isolate training recipe impact from architecture changes)
# ============================================================
print("\n" + "="*70)
print("  RUN 2: Training Improvements (Original Architecture)")
print("  Hypothesis: Modern optimization (AdamW, Cosine LR, warmup)")
print("  will massively outperform the baseline even with the same arch.")
print("="*70)

# We need a separate model.py for original arch. Use command overrides.
run_and_capture(
    TRAIN_CMD + [
        "--out", "ckpt_run2.pt",
        "--batch", "16", "--grad_accum", "4",
        "--lr", "6e-4", "--lr_min", "6e-5", "--warmup", "200",
        "--wd", "0.1", "--grad_clip", "1.0",
        "--block_size", "128",
        "--n_layer", "4", "--n_head", "4", "--n_embd", "160",
        "--swiglu_hidden", "640",  # 4x standard MLP
        "--no_swiglu",  # Use standard GELU MLP
        "--no_rope",  # Use learned pos embeddings
    ],
    "Run 2: Training Improvements (Original Arch)"
)
r = evaluate("ckpt_run2.pt")
if r: results["run2"] = r; print(f"  >>> Run 2 bpb: {r['bpb']}")

# ============================================================
# Run 3: Full modern architecture (RoPE + RMSNorm + SwiGLU)
# ============================================================
print("\n" + "="*70)
print("  RUN 3: Full Architecture Swap (RoPE + RMSNorm + SwiGLU)")
print("  Hypothesis: Modern components are more param-efficient")
print("="*70)

run_and_capture(
    TRAIN_CMD + [
        "--out", "ckpt_run3.pt",
        "--batch", "16", "--grad_accum", "4",
        "--lr", "6e-4", "--lr_min", "6e-5", "--warmup", "200",
        "--wd", "0.1", "--grad_clip", "1.0",
        # Default config: RoPE, SwiGLU, RMSNorm, tie_weights
        # n_embd=160, n_layer=6, n_head=5, block_size=256, swiglu_hidden=420
    ],
    "Run 3: Full Architecture (RoPE+SwiGLU+RMSNorm)"
)
r = evaluate("ckpt_run3.pt")
if r: results["run3"] = r; print(f"  >>> Run 3 bpb: {r['bpb']}")

# ============================================================
# Run 4: AMBITIOUS - Deep Narrow Model (9 layers, n_embd=128)
# Expected to STRUGGLE: with only 2000 steps, deep models can't
# fully train each layer. The gradients have to propagate through
# more layers, and early layers may not get enough signal.
# ============================================================
print("\n" + "="*70)
print("  RUN 4: AMBITIOUS - Deep Narrow (9L, embd=128)")
print("  Hypothesis: Will LOSE because 2000 steps is not enough for")
print("  9 layers to specialize. Tests depth vs width tradeoff.")
print("="*70)

run_and_capture(
    TRAIN_CMD + [
        "--out", "ckpt_run4.pt",
        "--batch", "16", "--grad_accum", "4",
        "--lr", "6e-4", "--lr_min", "6e-5", "--warmup", "200",
        "--wd", "0.1", "--grad_clip", "1.0",
        "--n_embd", "128", "--n_layer", "9", "--n_head", "4",
        "--block_size", "256", "--swiglu_hidden", "340",
    ],
    "Run 4: Deep Narrow (9L, 128d)"
)
r = evaluate("ckpt_run4.pt")
if r: results["run4"] = r; print(f"  >>> Run 4 bpb: {r['bpb']}")

# ============================================================
# Run 5: AMBITIOUS - Lion Optimizer
# Lion uses sign(momentum) instead of adaptive LR per-param.
# It's much simpler than Adam but has shown competitive results
# on large-scale training. At small scale with limited steps,
# it may struggle because it lacks Adam's per-parameter adaptation.
# ============================================================
print("\n" + "="*70)
print("  RUN 5: AMBITIOUS - Lion Optimizer")
print("  Hypothesis: Lion's sign-based updates may struggle at this")
print("  small scale because it lacks Adam's per-parameter adaptation.")
print("  But if it works, it would show that simpler optimizers can")
print("  compete when training budget is very limited.")
print("="*70)

# Lion needs a custom implementation - we'll use the run3 architecture
# but swap the optimizer. Let's create a custom training script for this.
# For now, skip if Lion script doesn't exist.
lion_script = os.path.join(os.path.dirname(__file__), "run5_lion.py")
if os.path.exists(lion_script):
    run_and_capture([sys.executable, lion_script], "Run 5: Lion Optimizer")
    r = evaluate("ckpt_run5.pt")
    if r: results["run5"] = r; print(f"  >>> Run 5 bpb: {r['bpb']}")
else:
    print("  Lion script not found, skipping")

# ============================================================
# Run 6: Ablation - GELU vs SwiGLU (same param budget)
# Tests whether the SwiGLU gating mechanism actually helps at
# this tiny model scale.
# ============================================================
print("\n" + "="*70)
print("  RUN 6: Ablation - Standard GELU MLP (same params)")
print("  Hypothesis: SwiGLU should win because gating provides")
print("  more expressivity per parameter, but the margin may be")
print("  small at this scale.")
print("="*70)

run_and_capture(
    TRAIN_CMD + [
        "--out", "ckpt_run6.pt",
        "--batch", "16", "--grad_accum", "4",
        "--lr", "6e-4", "--lr_min", "6e-5", "--warmup", "200",
        "--wd", "0.1", "--grad_clip", "1.0",
        "--no_swiglu",  # Use GELU MLP instead
        "--swiglu_hidden", "630",  # ~same params as SwiGLU(420) with 2 matrices
        # RoPE + RMSNorm + tie still on
    ],
    "Run 6: GELU MLP Ablation"
)
r = evaluate("ckpt_run6.pt")
if r: results["run6"] = r; print(f"  >>> Run 6 bpb: {r['bpb']}")

# ============================================================
# Run 7: Higher LR experiment
# ============================================================
print("\n" + "="*70)
print("  RUN 7: Higher LR (1e-3) with same architecture as Run 3")
print("  Hypothesis: With only 2000 steps, a higher LR might help")
print("  the model learn faster, at the risk of instability.")
print("="*70)

run_and_capture(
    TRAIN_CMD + [
        "--out", "ckpt_run7.pt",
        "--batch", "16", "--grad_accum", "4",
        "--lr", "1e-3", "--lr_min", "1e-4", "--warmup", "200",
        "--wd", "0.1", "--grad_clip", "1.0",
    ],
    "Run 7: Higher LR (1e-3)"
)
r = evaluate("ckpt_run7.pt")
if r: results["run7"] = r; print(f"  >>> Run 7 bpb: {r['bpb']}")

# ============================================================
# Run 8: Larger batch with fewer accum steps (speed vs quality)
# ============================================================
print("\n" + "="*70)
print("  RUN 8: Bigger batch, less accumulation")
print("  Hypothesis: batch=32, accum=2 (eff=64 same) but different")
print("  gradient noise characteristics.")
print("="*70)

run_and_capture(
    TRAIN_CMD + [
        "--out", "ckpt_run8.pt",
        "--batch", "32", "--grad_accum", "2",
        "--lr", "6e-4", "--lr_min", "6e-5", "--warmup", "200",
        "--wd", "0.1", "--grad_clip", "1.0",
    ],
    "Run 8: batch=32 accum=2"
)
r = evaluate("ckpt_run8.pt")
if r: results["run8"] = r; print(f"  >>> Run 8 bpb: {r['bpb']}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("  RESULTS SUMMARY")
print("="*70)
print(f"  {'Run':<30s}  {'bpb':>8s}  {'params':>10s}")
print(f"  {'-'*30}  {'-'*8}  {'-'*10}")
for name, r in sorted(results.items()):
    print(f"  {name:<30s}  {r['bpb']:>8.4f}  {r['n_params']:>10,}")

# Find best
if results:
    best = min(results.items(), key=lambda x: x[1]['bpb'])
    print(f"\n  BEST: {best[0]} with bpb={best[1]['bpb']:.4f}")
    print(f"\n  Copying best checkpoint to ckpt.pt...")
    import shutil
    best_ckpt = f"ckpt_{best[0]}.pt"
    shutil.copy2(best_ckpt, "ckpt.pt")
    print(f"  Copied {best_ckpt} -> ckpt.pt")
