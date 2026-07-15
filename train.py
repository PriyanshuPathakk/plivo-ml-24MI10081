"""Improved trainer with cosine LR, AdamW, gradient clipping, gradient
accumulation, and proper logging.

HARD CAPS (checked at grading, violations = disqualified run):
  * max 2,000 optimizer steps in the run that produces your checkpoint
  * max 2,000,000 total parameters
  * training text: the provided train_corpus.txt only
  * pure PyTorch / numpy / stdlib; no pretrained anything

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def cosine_lr(step, warmup, total, lr_max, lr_min):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup:
        return lr_max * step / warmup
    if step >= total:
        return lr_min
    progress = (step - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=4,
                    help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--lr_min", type=float, default=6e-5,
                    help="Minimum learning rate (cosine schedule floor)")
    ap.add_argument("--warmup", type=int, default=200,
                    help="Warmup steps for learning rate")
    ap.add_argument("--wd", type=float, default=0.1,
                    help="Weight decay for AdamW")
    ap.add_argument("--grad_clip", type=float, default=1.0,
                    help="Max gradient norm for clipping")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=100)
    # Architecture overrides
    ap.add_argument("--n_embd", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=None)
    ap.add_argument("--n_head", type=int, default=None)
    ap.add_argument("--block_size", type=int, default=None)
    ap.add_argument("--swiglu_hidden", type=int, default=None)
    ap.add_argument("--no_rope", action="store_true")
    ap.add_argument("--no_swiglu", action="store_true")
    ap.add_argument("--no_tie", action="store_true")
    ap.add_argument("--dropout", type=float, default=None)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    # Load data and tokenizer
    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")
    print(f"compression ratio: {len(text.encode('utf-8'))/len(ids):.2f}x")

    # Configure model
    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    # Apply overrides
    if args.n_embd is not None:
        cfg.n_embd = args.n_embd
    if args.n_layer is not None:
        cfg.n_layer = args.n_layer
    if args.n_head is not None:
        cfg.n_head = args.n_head
    if args.block_size is not None:
        cfg.block_size = args.block_size
    if args.swiglu_hidden is not None:
        cfg.swiglu_hidden = args.swiglu_hidden
    if args.no_rope:
        cfg.use_rope = False
    if args.no_swiglu:
        cfg.use_swiglu = False
    if args.no_tie:
        cfg.tie_weights = False
    if args.dropout is not None:
        cfg.dropout = args.dropout

    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params  (cap: {MAX_PARAMS:,}, margin: {MAX_PARAMS - n:,})")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params, got {n:,}"

    # Print architecture summary
    print(f"config: embd={cfg.n_embd} layers={cfg.n_layer} heads={cfg.n_head} "
          f"block={cfg.block_size} vocab={cfg.vocab_size}")
    print(f"        rope={cfg.use_rope} swiglu={cfg.use_swiglu} "
          f"tie={cfg.tie_weights} hidden={cfg.swiglu_hidden}")

    # AdamW with weight decay (modern practice: decay only non-embedding, non-norm)
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)
    optim_groups = [
        {"params": decay_params, "weight_decay": args.wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=args.lr,
                            betas=(0.9, 0.95), fused=False)

    print(f"training: steps={args.steps} batch={args.batch} "
          f"accum={args.grad_accum} effective_batch={args.batch * args.grad_accum}")
    print(f"          lr={args.lr} lr_min={args.lr_min} warmup={args.warmup} "
          f"wd={args.wd} grad_clip={args.grad_clip}")

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        # Update learning rate
        lr = cosine_lr(step, args.warmup, args.steps, args.lr, args.lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        total_loss = 0.0
        for micro in range(args.grad_accum):
            x, y = get_batch(ids, cfg.block_size, args.batch, device)
            _, loss = model(x, y)
            loss = loss / args.grad_accum
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(total_loss)

        if step % args.log_every == 0 or step == 1:
            avg = sum(losses[-args.log_every:]) / len(losses[-args.log_every:])
            elapsed = time.time() - t0
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({elapsed/step*1000:.0f} ms/step, {elapsed:.0f}s elapsed)")

    # Save checkpoint — every public config attribute is included
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    elapsed = time.time() - t0
    print(f"saved {args.out}  ({elapsed:.0f}s total, final loss {losses[-1]:.4f})")


if __name__ == "__main__":
    main()
