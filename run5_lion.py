"""Run 5: AMBITIOUS - Lion Optimizer experiment.

Lion (Evolved Sign Momentum) uses sign(momentum) instead of Adam's
adaptive per-parameter learning rates. Published by Google Brain in 2023.

Key differences from Adam:
1. Updates are just sign(momentum) - all parameters get same magnitude update
2. Uses two momentum terms but NO second moment (no v_t)
3. Much simpler, but loses Adam's ability to adapt to per-parameter gradients
4. Needs lower learning rate and higher weight decay than Adam

HYPOTHESIS: At this tiny scale (2M params, 2000 steps), Lion will likely
UNDERPERFORM AdamW because:
- Adam's per-parameter adaptation is crucial when different layers have
  very different gradient scales (embeddings vs attention vs MLP)
- 2000 steps is too few for Lion's sign-based updates to accumulate
  enough information
- However, if Lion works, it would be a remarkable finding for
  resource-constrained training

This is deliberately an ambitious experiment designed to test and fail.
"""
import math
import sys
import time
import torch
from model import GPT, Config
import tokenizer as tokenizer_mod

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


class Lion(torch.optim.Optimizer):
    """Lion optimizer (Chen et al., 2023).

    Implements the Evolved Sign Momentum optimizer.
    update = sign(beta1 * m + (1-beta1) * grad)
    m = beta2 * m + (1-beta2) * grad
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay (decoupled, like AdamW)
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Update: sign of interpolation between momentum and gradient
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-group['lr'])

                # Momentum update (EMA of gradient)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


def cosine_lr(step, warmup, total, lr_max, lr_min):
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
    torch.manual_seed(42)
    text = open("../data/train_corpus.txt", encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(ids):,} tokens (vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg)
    print(f"model: {model.n_params():,} params")

    # Lion needs LOWER lr and HIGHER weight decay than Adam
    # Typical: lr=3e-5 to 1e-4, wd=0.3 to 1.0
    lr_max = 1e-4
    lr_min = 1e-5
    wd = 0.3

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = Lion([
        {"params": decay, "weight_decay": wd},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.99))

    steps, warmup, batch, accum = 2000, 200, 16, 4
    print(f"Lion optimizer: lr={lr_max}, wd={wd}, betas=(0.9, 0.99)")
    print(f"Training: steps={steps}, batch={batch}, accum={accum}")

    model.train()
    t0 = time.time()
    losses = []
    for step in range(1, steps + 1):
        lr = cosine_lr(step, warmup, steps, lr_max, lr_min)
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
            avg = sum(losses[-200:]) / len(losses[-200:])
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({(time.time()-t0)/step*1000:.0f}ms/step)")

    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_") and not callable(getattr(cfg, k))},
                "steps": steps, "train_loss_curve": losses}, "ckpt_run5.pt")
    print(f"done in {time.time()-t0:.0f}s, final loss {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
