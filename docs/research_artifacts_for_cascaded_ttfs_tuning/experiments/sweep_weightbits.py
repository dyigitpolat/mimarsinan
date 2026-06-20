"""A2 — WEIGHT-QUANTIZATION loss source.

Decompose the ANN->deployed loss: how much does snapping the converted-flow
weights to W integer bits cost, and is the PRODUCTION 5-bit setting near-lossless
or a real source? We reuse the EXACT production quantizer
(``NormalizationAwarePerceptronQuantization``, rate=1.0 == the legacy full
symmetric-integer quantization the pipeline's ``WeightQuantizationStep`` runs at
``weight_bits``): per perceptron it fuses BN into the linear, takes
``scale = q_max / max(|W_eff|, |b_eff|)``, rounds the effective weight AND bias to
the integer grid, clips to ``[q_min, q_max]``, rescales. That is bit-for-bit the
deployed weight grid (the IR ``core_matrix`` is the same rounded matrix; the chip
threshold absorbs ``scale``).

Two protocols, both reporting the loss as a DELTA from the continuous ANN (cont):

  STAIRCASE (analytical ceiling, no FT confound): quantize the COLD converted flow,
    measure the round-staircase forward (cycle_accurate OFF, the optimal linear
    timing decode == the ANN-level ceiling). Isolates pure weight-quant on the
    lossless analytical forward. We report both the delta vs cont AND vs the FP
    staircase (the quant-only delta, factoring out the S-level ceiling).

  GENUINE (deployed path): FT a genuine-healthy cascade flow with the combo
    recipe (the genuine single-spike ramp deploy path recovers to ~0.95-0.96),
    THEN quantize its weights post-hoc and measure the genuine cascade accuracy.
    This is the deployment-relevant weight-quant loss: what 5-bit costs the model
    you would actually ship. (Post-hoc quant; a quant-aware FT mitigation is
    measured separately below.)

Sweep W in {3,4,5,6,8,16} at S=16 and S=32; interaction = does quant cost more at
high or low S? Mitigation tested: quant-AWARE fine-tune (re-FT a few steps AFTER
quantizing, with the production quantizer re-applied each step as a hard STE) to
recover the post-hoc 5-bit genuine drop.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/sweep_weightbits.py
"""

from __future__ import annotations

import copy
import os
import sys
import time

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from ft_budget import build  # noqa: E402
from lif_vs_ttfs import ttfs_staircase_acc, ttfs_genuine_acc  # noqa: E402
import recipe_combo as combo  # noqa: E402
from recipe_harness import batches, genuine_logits, kd_ce_loss, teacher_logits  # noqa: E402
from mimarsinan.transformations.normalization_aware_perceptron_quantization import (  # noqa: E402
    NormalizationAwarePerceptronQuantization,
)

WBITS = (3, 4, 5, 6, 8, 16)
DEPTH = 6
PROD_BITS = 5
FT_STEPS = 500


def quantize_flow(flow, bits, device):
    """Apply the production per-perceptron symmetric integer weight quant in place.

    The production ``get_effective_weight`` divides ``W (out,in)`` by a SCALAR
    ``activation_scale``. combo-FT promotes ``activation_scale`` to a per-output-
    channel vector ``(out,)``, which mis-broadcasts. We reshape it to ``(out,1)``
    for the weight path for the duration of the transform (the bias path wants
    ``(out,)``); since the quantizer rescales by the same scale it cancels and the
    grid is identical to the scalar case per output channel."""
    q = NormalizationAwarePerceptronQuantization(bits=bits, device=str(device), rate=1.0)
    for p in flow.get_perceptrons():
        scale = p.activation_scale
        if scale.dim() == 1:
            _quantize_perchannel(p, bits, q)
        else:
            q.transform(p)
    return flow


def _quantize_perchannel(p, bits, q):
    """Per-output-channel-scale variant: quantize each output row on the shared
    per-perceptron grid (scale = q_max / max over all rows of |W_eff|,|b_eff|),
    exactly as production does, but with the per-channel activation_scale folded
    into the effective weight/bias by hand so the (out,) scale broadcasts right."""
    import torch as _t
    from mimarsinan.transformations.perceptron.perceptron_transformer import (
        PerceptronTransformer,
    )
    pt = PerceptronTransformer()
    s = p.activation_scale.reshape(-1, 1)            # (out,1)
    # effective weight/bias with per-channel scale (mirror get_effective_*)
    import torch.nn as _nn
    if isinstance(p.normalization, _nn.Identity):
        w_eff = p.layer.weight.data / s
        b_src = (_t.zeros(p.layer.weight.shape[0], device=s.device, dtype=s.dtype)
                 if p.layer.bias is None else p.layer.bias.data)
        b_eff = b_src / s.reshape(-1)
    else:
        u, beta, mean = pt._get_u_beta_mean(p.normalization)
        w_eff = (p.layer.weight.data * u.unsqueeze(-1)) / s
        b_src = (_t.zeros(p.layer.weight.shape[0], device=s.device, dtype=s.dtype)
                 if p.layer.bias is None else p.layer.bias.data)
        b_eff = ((b_src - mean) * u + beta) / s.reshape(-1)
    p_max = max(float(w_eff.abs().max()), float(b_eff.abs().max()), 1e-12)
    grid = q.q_max / p_max
    qw = _t.clamp(_t.round(w_eff * grid), q.q_min, q.q_max) / grid
    qb = _t.clamp(_t.round(b_eff * grid), q.q_min, q.q_max) / grid
    # write back through the same effective relation (invert the fold)
    if isinstance(p.normalization, _nn.Identity):
        p.layer.weight.data[:] = qw * s
        if p.layer.bias is not None:
            p.layer.bias.data[:] = qb * s.reshape(-1)
    else:
        u, beta, mean = pt._get_u_beta_mean(p.normalization)
        p.layer.weight.data[:] = (qw * s) / u.unsqueeze(-1)
        if p.layer.bias is not None:
            p.layer.bias.data[:] = (((qb * s.reshape(-1)) - beta) / u) + mean


def quant_aware_ft(flow, xtr, ytr, S, base, teacher, bits, *, steps, seed,
                   w_lr=5e-4, bs=256, alpha=0.3):
    """Quant-aware FT: hard STE through the production quantizer each step.

    forward weights == quantized (deploy grid); backward via the FP weights
    (straight-through). The optimiser updates FP weights; we re-snap before each
    genuine forward so training sees the deployed grid."""
    device = xtr.device
    params = [p for p in flow.parameters() if p.requires_grad]
    if not params:
        for p in flow.get_perceptrons():
            p.layer.weight.requires_grad_(True)
            if p.layer.bias is not None:
                p.layer.bias.requires_grad_(True)
        params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=w_lr)
    for x, y in batches(xtr, ytr, bs, steps, seed):
        fp = {p: (p.layer.weight.data.clone(),
                  None if p.layer.bias is None else p.layer.bias.data.clone())
              for p in flow.get_perceptrons()}
        quantize_flow(flow, bits, device)            # snap to deploy grid (forward)
        logits = genuine_logits(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=alpha)
        opt.zero_grad(); loss.backward()
        for p, (w, b) in fp.items():                 # STE: restore FP, then step
            p.layer.weight.data = w
            if b is not None:
                p.layer.bias.data = b
        opt.step()
    quantize_flow(flow, bits, device)                # final deploy grid
    return flow


def run_staircase(S, seed=0):
    """Cold converted flow: weight-quant loss on the analytical staircase ceiling."""
    flow, _xtr, _ytr, xte, yte, cont, _t, _b = build(DEPTH, S, seed=seed)
    stair_fp = ttfs_staircase_acc(flow, xte, yte, S)
    rows = []
    for W in WBITS:
        fq = quantize_flow(copy.deepcopy(flow), W, xte.device)
        acc = ttfs_staircase_acc(fq, xte, yte, S)
        rows.append((W, acc))
    return cont, stair_fp, rows


def run_genuine(S, seed=0):
    """Genuine-healthy FT'd flow: post-hoc weight-quant loss on the deploy path,
    plus the quant-aware-FT mitigation at the production bit width."""
    flow, xtr, ytr, xte, yte, cont, teacher, base = build(DEPTH, S, seed=seed)
    combo.train(flow, xtr, ytr, xte, yte, S, base, teacher, steps=FT_STEPS, seed=seed)
    gen_fp = ttfs_genuine_acc(flow, xte, yte, S)
    rows = []
    for W in WBITS:
        fq = quantize_flow(copy.deepcopy(flow), W, xte.device)
        rows.append((W, ttfs_genuine_acc(fq, xte, yte, S)))
    qaft = quant_aware_ft(copy.deepcopy(flow), xtr, ytr, S, base, teacher,
                          PROD_BITS, steps=200, seed=seed + 1)
    gen_qaft = ttfs_genuine_acc(qaft, xte, yte, S)
    return cont, gen_fp, rows, gen_qaft


def main():
    t0 = time.time()
    print(f"=== A2 weight-quant loss (depth={DEPTH}, prod bits={PROD_BITS}) ===\n")

    print("--- STAIRCASE (analytical ceiling, cold convert, no FT) ---")
    print("  loss reported as pp DELTA from cont; quant-only = vs FP staircase")
    print(f"{'S':>3} {'cont':>7} {'stairFP':>8} | " +
          " ".join(f"W{w:>2}" for w in WBITS))
    stair_data = {}
    for S in (16, 32):
        cont, stair_fp, rows = run_staircase(S)
        stair_data[S] = (cont, stair_fp, dict(rows))
        accs = " ".join(f"{a:>5.3f}" for _w, a in rows)
        print(f"{S:>3} {cont:>7.4f} {stair_fp:>8.4f} | {accs}")
    print("  delta vs cont (pp):")
    for S in (16, 32):
        cont, stair_fp, d = stair_data[S]
        deltas = " ".join(f"{(d[w]-cont)*100:>5.1f}" for w in WBITS)
        print(f"{S:>3} {'':>7} {'':>8} | {deltas}")
    print("  quant-only delta vs FP staircase (pp):")
    for S in (16, 32):
        cont, stair_fp, d = stair_data[S]
        deltas = " ".join(f"{(d[w]-stair_fp)*100:>5.1f}" for w in WBITS)
        print(f"{S:>3} {'':>7} {'':>8} | {deltas}")

    print("\n--- GENUINE (deployed single-spike cascade, combo-FT'd) ---")
    print("  loss reported as pp DELTA from FP genuine (post-hoc weight quant)")
    print(f"{'S':>3} {'cont':>7} {'genFP':>7} | " +
          " ".join(f"W{w:>2}" for w in WBITS) + f" | QAFT@{PROD_BITS}")
    for S in (16, 32):
        cont, gen_fp, rows, gen_qaft = run_genuine(S)
        d = dict(rows)
        accs = " ".join(f"{a:>5.3f}" for _w, a in rows)
        print(f"{S:>3} {cont:>7.4f} {gen_fp:>7.4f} | {accs} | {gen_qaft:>6.4f}")
        deltas = " ".join(f"{(d[w]-gen_fp)*100:>5.1f}" for w in WBITS)
        print(f"{S:>3} {'':>7} {'delta':>7} | {deltas} | "
              f"{(gen_qaft - d[PROD_BITS]) * 100:>+5.1f}pp recover")

    print(f"\n[total {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
