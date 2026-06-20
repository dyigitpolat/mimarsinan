"""Diagnose the high-S optimization wall: PER-LAYER, PER-S surrogate-gradient
signal during genuine cascaded-TTFS fine-tuning.

The genuine cascade fires greedily on the running partial sum; the surrogate
gradient flows backward through one ATan Heaviside per neuron per layer, ramp-
integrated over S cycles. The high-S failure (d=9 S32: combo 0.911, ceiling
0.965) must be one of: gradient VANISHING, EXPLODING, or BIASED/NOISY credit
assignment. This script measures each, per layer and per S, over the first
~STEPS steps of a fresh genuine FT, and prints which layers / which S break.

Metrics per layer (perceptron weight param), per step:
  - gnorm        : ||grad||_2                       (magnitude / vanish-explode)
  - gmax         : max |grad element|               (explode tail)
  - dead_frac    : fraction of weight elements with EXACTLY-zero grad
                   (no surrogate signal reached the param -> dead-neuron cascade)
  - dead_out     : fraction of OUTPUT neurons whose entire weight-row grad is 0
                   (the death-cascade unit: a fully dead post-synaptic neuron)
  - snr          : per-sample gradient SNR = ||E[g]|| / mean_elem(std_sample[g])
                   estimated from K microbatch gradient samples. Low SNR = noisy
                   credit assignment (the surrogate signal is drowned by variance).
  - cos_consec   : cosine(grad_t, grad_{t-1}) running -> directional consistency.

Usage:  source env/bin/activate && python docs/.../experiments/diag_gradient.py
"""

from __future__ import annotations

import os
import sys
import time

import torch

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from recipe_harness import (  # noqa: E402
    build, genuine_logits, genuine_acc, kd_ce_loss, teacher_logits,
)

STEPS = 200
BS = 256
LR = 2e-3
K_MICRO = 4           # microbatches for per-sample SNR estimate
LOG_EVERY = 25        # print a snapshot table this often
SEED = 0


def _weight_params(flow):
    """Ordered (idx, label, weight_param) for every perceptron's linear weight."""
    out = []
    for i, p in enumerate(flow.get_perceptrons()):
        enc = getattr(p, "is_encoding_layer", False)
        label = f"L{i}{'(enc)' if enc else '(out)' if i == len(flow.get_perceptrons()) - 1 else ''}"
        p.layer.weight.requires_grad_(True)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(True)
        out.append((i, label, p.layer.weight))
    return out


def _grad_vec(params):
    """Flatten per-param grads into one vector (zeros where grad is None)."""
    return torch.cat([
        (w.grad.reshape(-1) if w.grad is not None else torch.zeros_like(w).reshape(-1))
        for w in params
    ])


def _per_layer_grad(flow, x, y, base, S):
    """One backward pass; return {layer_idx: grad tensor (cloned)} for weights."""
    wp = _weight_params(flow)
    for _, _, w in wp:
        if w.grad is not None:
            w.grad = None
    logits = genuine_logits(flow, x, S)
    loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=0.3)
    loss.backward()
    grads = {}
    for i, _, w in wp:
        grads[i] = (w.grad.detach().clone() if w.grad is not None
                    else torch.zeros_like(w))
    return grads, float(loss.detach())


def _snr_per_layer(flow, x, y, base, S, k):
    """Per-sample gradient SNR per layer from k disjoint microbatch grad samples.

    SNR = ||mean_micro(g)|| / mean_elem( std_micro(g) ). High = consistent signal,
    low = the surrogate gradient is dominated by sample-to-sample variance
    (noisy / biased credit assignment).
    """
    n = x.shape[0]
    m = n // k
    samples: dict[int, list] = {}
    for j in range(k):
        xj, yj = x[j * m:(j + 1) * m], y[j * m:(j + 1) * m]
        gj, _ = _per_layer_grad(flow, xj, yj, base, S)
        for i, g in gj.items():
            samples.setdefault(i, []).append(g)
    snr = {}
    for i, gs in samples.items():
        stack = torch.stack(gs, 0)               # (k, *shape)
        mean = stack.mean(0)
        std = stack.std(0)
        num = float(mean.norm())
        den = float(std.mean()) + 1e-30
        snr[i] = num / den
    return snr


def _stats(grad):
    flat = grad.reshape(-1)
    gnorm = float(flat.norm())
    gmax = float(flat.abs().max())
    dead_frac = float((flat == 0).float().mean())
    # output-neuron death: a full weight row all-zero
    rows = grad.reshape(grad.shape[0], -1)
    dead_out = float((rows.abs().sum(1) == 0).float().mean())
    return gnorm, gmax, dead_frac, dead_out


def run_for_S(depth, S, steps=STEPS, seed=SEED):
    print(f"\n{'='*78}\n=== genuine FT gradient diagnosis: depth={depth}  S={S} ===")
    t0 = time.time()
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
    print(f"built ({time.time()-t0:.1f}s)  cont={cont:.4f}  "
          f"cold_genuine={genuine_acc(flow, xte, yte, S):.4f}")
    flow = flow.double()
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=LR)
    n_layers = len(flow.get_perceptrons())

    g = torch.Generator().manual_seed(seed)
    prev_gvec = None
    # accumulators over steps: list of per-layer dicts
    history = {i: {"gnorm": [], "dead_frac": [], "dead_out": [],
                   "gmax": [], "snr": [], "cos": []} for i in range(n_layers)}
    cos_running = {i: [] for i in range(n_layers)}
    prev_grad = {i: None for i in range(n_layers)}

    for step in range(steps):
        idx = torch.randint(0, xtr.shape[0], (BS,), generator=g).to(xtr.device)
        x, y = xtr[idx].double(), ytr[idx]

        grads, loss = _per_layer_grad(flow, x, y, base, S)
        # SNR is expensive (k extra backwards); sample it on log steps only.
        snr = (_snr_per_layer(flow, x, y, base, S, K_MICRO)
               if (step % LOG_EVERY == 0 or step == steps - 1) else None)

        for i in range(n_layers):
            gnorm, gmax, dead_frac, dead_out = _stats(grads[i])
            history[i]["gnorm"].append(gnorm)
            history[i]["gmax"].append(gmax)
            history[i]["dead_frac"].append(dead_frac)
            history[i]["dead_out"].append(dead_out)
            if snr is not None:
                history[i]["snr"].append(snr[i])
            gflat = grads[i].reshape(-1)
            if prev_grad[i] is not None and gflat.norm() > 0 and prev_grad[i].norm() > 0:
                cos = float(torch.dot(gflat, prev_grad[i]) /
                            (gflat.norm() * prev_grad[i].norm()))
                cos_running[i].append(cos)
            prev_grad[i] = gflat.clone()

        # take the optimizer step on the FULL-batch grad (already populated)
        gvec = _grad_vec([flow.get_perceptrons()[i].layer.weight
                          for i in range(n_layers)])
        opt.step()
        opt.zero_grad()
        prev_gvec = gvec

        if step % LOG_EVERY == 0 or step == steps - 1:
            ga = genuine_acc(flow, xte, yte, S)
            print(f"\n-- step {step:3d}  loss={loss:.4f}  genuine_acc={ga:.4f} --")
            print(f"{'layer':>8} {'gnorm':>10} {'gmax':>10} {'dead%':>7} "
                  f"{'deadOut%':>9} {'snr':>8} {'cosCons':>8}")
            for i in range(n_layers):
                h = history[i]
                cc = (sum(cos_running[i][-LOG_EVERY:]) / len(cos_running[i][-LOG_EVERY:])
                      if cos_running[i] else float("nan"))
                s = h["snr"][-1] if h["snr"] else float("nan")
                p = flow.get_perceptrons()[i]
                lbl = (f"L{i}(enc)" if getattr(p, "is_encoding_layer", False)
                       else f"L{i}(out)" if i == n_layers - 1 else f"L{i}")
                print(f"{lbl:>8} {h['gnorm'][-1]:>10.2e} {h['gmax'][-1]:>10.2e} "
                      f"{100*h['dead_frac'][-1]:>6.1f}% {100*h['dead_out'][-1]:>8.1f}% "
                      f"{s:>8.3f} {cc:>8.3f}")

    final_acc = genuine_acc(flow, xte, yte, S)
    print(f"\n>> depth={depth} S={S}: final genuine_acc after {steps} steps = "
          f"{final_acc:.4f}  (cont={cont:.4f}, gap={cont-final_acc:+.4f})  "
          f"[{time.time()-t0:.0f}s]")
    return history, cos_running, cont, final_acc, n_layers


def _summary(tag, history, cos_running, n_layers):
    """Per-layer mean over the run: the headline numbers for the wall verdict."""
    print(f"\n--- SUMMARY {tag}: per-layer mean over run ---")
    print(f"{'layer':>8} {'mean_gnorm':>11} {'mean_gmax':>11} {'mean_dead%':>11} "
          f"{'mean_deadOut%':>14} {'mean_snr':>9} {'mean_cos':>9}")
    rows = []
    for i in range(n_layers):
        h = history[i]
        mg = sum(h["gnorm"]) / len(h["gnorm"])
        mx = sum(h["gmax"]) / len(h["gmax"])
        md = 100 * sum(h["dead_frac"]) / len(h["dead_frac"])
        mdo = 100 * sum(h["dead_out"]) / len(h["dead_out"])
        ms = (sum(h["snr"]) / len(h["snr"])) if h["snr"] else float("nan")
        mc = (sum(cos_running[i]) / len(cos_running[i])) if cos_running[i] else float("nan")
        print(f"L{i:>7} {mg:>11.2e} {mx:>11.2e} {md:>10.1f}% {mdo:>13.1f}% "
              f"{ms:>9.3f} {mc:>9.3f}")
        rows.append((i, mg, mx, md, mdo, ms, mc))
    return rows


def main():
    torch.manual_seed(SEED)
    results = {}
    for S in (16, 32):
        hist, cos, cont, acc, nl = run_for_S(9, S)
        rows = _summary(f"d9_S{S}", hist, cos, nl)
        results[S] = (rows, cont, acc)

    print(f"\n{'#'*78}\n# VERDICT: d=9 S16 vs S32 (the wall)\n{'#'*78}")
    for S in (16, 32):
        rows, cont, acc = results[S]
        print(f"\nS={S}: final genuine_acc={acc:.4f}  cont={cont:.4f}  gap={cont-acc:+.4f}")
        # report the deepest layers (where death cascades concentrate)
        deep = [r for r in rows if r[0] >= 5]
        print("  deep layers (L5..L8) mean: "
              f"gnorm={sum(r[1] for r in deep)/len(deep):.2e}  "
              f"dead%={sum(r[3] for r in deep)/len(deep):.1f}  "
              f"deadOut%={sum(r[4] for r in deep)/len(deep):.1f}  "
              f"snr={sum(r[5] for r in deep)/len(deep):.3f}  "
              f"cos={sum(r[6] for r in deep)/len(deep):.3f}")


if __name__ == "__main__":
    main()
