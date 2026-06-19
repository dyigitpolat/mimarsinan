"""PHASE 2 — STACK the levers where depth > d_max(S) (rescale alone cannot fix).

Phase 1 result (artifact 20): the cascaded single-spike TTFS "death cascade" is a
CORRECTABLE per-layer decode gain-distortion. Three deployable levers were found:

  G  GAIN-CORRECTION (theta trim) — a per-layer/per-channel multiplicative trim of
     ``activation_scale`` (theta), derived from CALIBRATION stats (posmean->target),
     that inverts the ramp's R(tau)~(T-tau)^2 down-weighting of late/deep spikes.
     Deployable category: CALIBRATION (sets the per-neuron threshold; bit-exact decode).
     => reuses precomp.py (DIRECTION B).
  F  GENUINE-FORWARD STE FINE-TUNING — fine-tune the weights THROUGH the genuine
     single-spike cascade (cascade_forward(grad=True, surrogate_temp=...)). Phase 1
     showed the genuine STE is the well-conditioned gradient ONCE the cascade is alive
     (3x at S=16), and the crude analytical/timing proxy is REFUTED. Deployable
     category: TRAINING-TIME (it only moves trained weights/biases).
  K  INPUT->DECODE CONCAT SKIP — every core ALSO reads the raw input via torch.cat
     (InputSkipAllMLP from depth_reduction.py / DIRECTION E). The input spike fires
     earliest (long ramp, faithful), so routing it to every core keeps a faithful,
     depth-independent reference current alive => effective depth ~1 for the surviving
     path; extends d_max. Deployable category: ROUTING (ConcatMapper, in-segment, NO
     extra spike, NO host op) + the trained skip weights.

THE QUESTION (this file): at depth in {4,6}, S in {8,16} (so depth > d_max ~ 0.56*sqrt(S),
i.e. d_max in {1.6 @S8, 2.2 @S16}), where a PURE rescale provably fails (depth-6 S=8 stays
at chance), does STACKING the levers reach continuous-level? Full ablation:
baseline / G / F / K / G+F / G+K / F+K / G+F+K, multi-seed, vs the architecture's OWN
continuous teacher AND the oracle per-depth theta-scale upper bound.

A CONFOUND we control explicitly: a plain deep MLP underfits digits (cont ~0.45 at
depth 6); the skip architecture trains HIGHER (cont ~0.56). So we report (a) absolute
gen_acc, (b) the architecture's own continuous ceiling, and (c) RETENTION = gen/cont, to
separate "the cascade survived" from "the net trained better". The oracle theta-scale
(grid search, NOT deployable as-is — it is an UPPER BOUND reference) bounds what any
static per-depth gain trim can reach.

Run:  source env/bin/activate
      python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/stack.py
      python docs/.../experiments/stack.py quick    # smaller grid, 2 seeds (smoke)
"""

from __future__ import annotations

import itertools
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

# cascade_lab.py is the research-artifacts root (one dir up).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cascade_lab as L  # sets sys.path for cascade_fixtures + src
from cascade_fixtures import _calibrate_scales, cascade_forward, install_ttfs_nodes
from depth_reduction import InputSkipAllMLP, PlainMLP, acc, segment_count
from mimarsinan.torch_mapping.converter import convert_torch_model

CHANCE = 0.10  # 10-class digits


# --------------------------------------------------------------------------- #
# Calibration-derived GAIN CORRECTION (lever G). Reuses the precomp.py recipe
# (DIRECTION B): per-CHANNEL posmean->target theta trim, a pure function of the
# pre-install ReLU calibration stats. Deployable as a per-neuron threshold trim.
# --------------------------------------------------------------------------- #
def _per_channel_relu_stats(flow, x):
    """Per-perceptron channel-flattened ReLU activation on calib (theta-independent).

    MUST run BEFORE install_ttfs_nodes (which swaps p.activation for the spike node).
    """
    acts: dict = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, out, k=k):
            acts[k] = out.detach().reshape(-1, out.shape[-1]).clamp(min=0).double()
        handles.append(p.activation.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return acts


def posmean_target_thetas(stats, base_scales, *, target=0.5):
    """theta_d[c] = mean over the FIRED (>0) population per channel / target.

    Centres each channel's typical active drive at ``target`` -> mean fire-time
    tau ~ S(1-target), inverting the death cascade's late-firing. Per-channel is
    just as deployable as scalar (the chip threshold is per-neuron). Identical
    recipe to precomp.thetas_posmean_target (the winning closed form, B)."""
    out = []
    for k in range(len(base_scales)):
        a = stats[k]
        mask = (a > 1e-9).double()
        denom = mask.sum(0).clamp(min=1.0)
        pm = (a.sum(0) / denom).clamp(min=1e-6) / target  # (C,)
        out.append(pm.to(torch.float64))
    return out


def apply_gain_correction(flow, calib, *, target=0.5):
    """Set per-channel theta from the pre-install calib ReLU posmean (lever G).

    Returns the chosen thetas (so a re-install after FT can re-apply them). Call
    BEFORE install_ttfs_nodes."""
    base_scales = [p.activation_scale.clone() for p in flow.get_perceptrons()]
    stats = _per_channel_relu_stats(flow, calib)
    thetas = posmean_target_thetas(stats, base_scales, target=target)
    for p, th in zip(flow.get_perceptrons(), thetas):
        p.set_activation_scale(th)
    return thetas


# --------------------------------------------------------------------------- #
# GENUINE-FORWARD STE FINE-TUNING (lever F). Fine-tune all weights through the
# genuine single-spike cascade. surrogate_temp enables the offload-boundary STE
# so the backward flows through every segment (Phase 1: the well-conditioned grad).
# --------------------------------------------------------------------------- #
def finetune_genuine(flow, xtr, ytr, S, *, ft_epochs=40, ft_lr=2e-3, surrogate_temp=0.5):
    """In-place genuine-cascade STE fine-tune of the flow's trainable params.

    Re-installs spike nodes first so theta/bias edits (lever G) are picked up; the
    install keeps the current activation_scale on each perceptron."""
    install_ttfs_nodes(flow, S)
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=ft_lr)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    for _ in range(ft_epochs):
        opt.zero_grad()
        logits = cascade_forward(flow, xtr, S, grad=True, surrogate_temp=surrogate_temp)
        lossf(logits, ytr).backward()
        opt.step()


# --------------------------------------------------------------------------- #
# Build + measure. One trained continuous backbone per (arch, depth, seed); the
# cheap conversion + theta-trim + FT are then applied per lever-combo on a FRESH
# converted copy (so combos don't contaminate each other).
# --------------------------------------------------------------------------- #
def _fresh_flow(base, xtr, *, in_dim=64, n_classes=10):
    """Convert + calibrate a fresh flow from the (already-trained) continuous base.

    Asserts the whole net is ONE spike segment (concat skips are transparent
    routing; a value-add would cut it -> not a cascade)."""
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    if len(flow.get_perceptrons()) == 0 or segment_count(flow) != 1:
        raise AssertionError(
            f"non-cascade conversion: {len(flow.get_perceptrons())} percs, "
            f"{segment_count(flow)} segs")
    _calibrate_scales(flow, xtr[:256])
    return flow


def eval_combo(base, xtr, ytr, xte, yte, S, *, gain, ft, target=0.5,
               ft_epochs=40, ft_lr=2e-3):
    """Evaluate one (gain on/off, ft on/off) combo on a fresh converted copy.

    The SKIP lever is the architecture (``base`` is plain or skip-all), so the
    K dimension is selected by the caller's ctor; gain/ft toggle here. Returns
    genuine test accuracy of the deployed cascade."""
    flow = _fresh_flow(base, xtr)
    if gain:
        apply_gain_correction(flow, xtr[:256], target=target)
    install_ttfs_nodes(flow, S)
    if ft:
        finetune_genuine(flow, xtr, ytr, S, ft_epochs=ft_epochs, ft_lr=ft_lr)
    with torch.no_grad():
        return acc(cascade_forward(flow, xte, S), yte)


def oracle_theta_scale(base, xtr, xte, yte, S, *, grid=(1.0, 0.7, 0.5, 0.35, 0.25, 0.15),
                       seed=0):
    """UPPER-BOUND reference: best per-depth theta-scale found on the eval set. NOT
    deployable (it peeks at the metric); it bounds what any STATIC per-depth GAIN
    trim could reach. The per-layer optimal gains are NON-separable (a layer's best
    gain depends on the others' fire-times), so a greedy coordinate search badly
    UNDER-estimates it -> we use the FULL product grid for depth<=4 (grid^4<=1296)
    and a large RANDOM search seeded by the law-implied monotone-decreasing trims
    for deeper nets (full product is infeasible: grid^6 ~ 46656 * seeds)."""
    flow = _fresh_flow(base, xtr)
    percs = flow.get_perceptrons()
    base_scales = [p.activation_scale.clone() for p in percs]
    depth = len(percs)

    def acc_for(gammas):
        for k, p in enumerate(percs):
            p.set_activation_scale(base_scales[k] * gammas[k])
        install_ttfs_nodes(flow, S)
        return acc(cascade_forward(flow, xte, S), yte)

    if depth <= 4:
        candidates = list(itertools.product(grid, repeat=depth))
    else:
        rng = np.random.default_rng(seed)
        # monotone-decreasing trims (the death-cascade law predicts deeper layers
        # need a smaller theta) + uniform random; ~1500 samples bound it well.
        mono = []
        for _ in range(900):
            vals = sorted(rng.choice(grid, size=depth), reverse=True)
            mono.append(tuple(vals))
        rand = [tuple(rng.choice(grid, size=depth)) for _ in range(600)]
        candidates = [tuple([1.0] * depth)] + mono + rand
    best = ([1.0] * depth, acc_for([1.0] * depth))
    for g in candidates:
        a = acc_for(list(g))
        if a > best[1]:
            best = (list(g), a)
    return float(best[1]), best[0]


# --------------------------------------------------------------------------- #
# Ablation grid. K is the architecture axis (plain vs skip-all); G, F toggle.
# --------------------------------------------------------------------------- #
LEVER_COMBOS = [
    ("baseline", dict(gain=False, ft=False)),
    ("G", dict(gain=True, ft=False)),
    ("F", dict(gain=False, ft=True)),
    ("G+F", dict(gain=True, ft=True)),
]


_BACKBONE_CACHE: dict = {}


def _get_backbone(name, ctor, depth, seed):
    """Trained continuous backbone (cached by arch/depth/seed); reused across S."""
    key = (name, depth, seed)
    if key not in _BACKBONE_CACHE:
        torch.manual_seed(seed)
        xtr, ytr, xte, yte = L.digits_task(seed=seed + 1)
        b = ctor(depth)
        L.train_continuous(b, xtr, ytr, epochs=120)
        with torch.no_grad():
            c = acc(b(xte.float()), yte)
        _BACKBONE_CACHE[key] = (b, xtr, ytr, xte, yte, c)
    return _BACKBONE_CACHE[key]


def law_target(depth, S):
    """Calibration-only gain target that rises with depth (NOT metric-tuned).

    The death-cascade law: the local fire-cycle drifts later with depth, toward the
    window S. To keep the DEEPEST layer firing inside its window, the per-layer
    target normalized drive must rise with depth (higher target -> earlier firing
    -> longer ramp). A monotone schedule in depth, capped at 0.8 (above that the
    fire-time hits 0 and the timing code saturates). A function of (depth, S) only --
    a calibration choice, never the eval metric."""
    return float(min(0.5 + 0.06 * depth, 0.8))


def run_ablation(depth, S, seeds, *, ft_epochs=40, ft_lr=2e-3, target=None,
                 do_oracle=True):
    """Full stacked-lever ablation at one (depth, S), averaged over seeds.

    Returns a dict keyed by lever-label -> {plain: ..., skip: ...} with mean gen,
    per-seed gens, the arch's continuous ceiling, and (plain) oracle bound."""
    if target is None:
        target = law_target(depth, S)
    archs = [("plain", PlainMLP), ("skip", InputSkipAllMLP)]
    # Train one continuous backbone per (arch, depth, seed); the 120-epoch train is
    # the cost, so cache and reuse across S (conversion + trim + FT are cheap).
    backbones = {}
    cont = {}
    oracle = {}
    for name, ctor in archs:
        cont[name] = []
        oracle[name] = []
        for s in seeds:
            b, xtr, ytr, xte, yte, c = _get_backbone(name, ctor, depth, s)
            cont[name].append(c)
            backbones[(name, s)] = (b, xtr, ytr, xte, yte)
            if do_oracle:
                oa, _ = oracle_theta_scale(b, xtr, xte, yte, S, seed=s)
                oracle[name].append(oa)

    results = {}
    for label, kw in LEVER_COMBOS:
        results[label] = {}
        for name, _ in archs:
            gens = []
            for s in seeds:
                b, xtr, ytr, xte, yte = backbones[(name, s)]
                gens.append(eval_combo(b, xtr, ytr, xte, yte, S, target=target,
                                       ft_epochs=ft_epochs, ft_lr=ft_lr, **kw))
            results[label][name] = gens
    return {
        "depth": depth, "S": S, "seeds": list(seeds), "target": target,
        "cont": {n: cont[n] for n, _ in archs},
        "oracle": {n: oracle[n] for n, _ in archs} if do_oracle else None,
        "results": results,
    }


def _fmt(v):
    return f"{np.mean(v):.3f}" if v else "  -  "


def print_ablation(r):
    depth, S = r["depth"], r["S"]
    print(f"\n{'='*86}")
    print(f"depth={depth}  S={S}  (d_max~{0.56*S**0.5:.1f})  seeds={r['seeds']}  "
          f"chance={CHANCE}  gain_target={r['target']:.2f}")
    print(f"{'='*86}")
    cont_p = np.mean(r["cont"]["plain"])
    cont_s = np.mean(r["cont"]["skip"])
    print(f"continuous ceiling:   plain={cont_p:.3f}   skip(=K)={cont_s:.3f}")
    if r["oracle"] is not None:
        print(f"oracle theta-scale:   plain={np.mean(r['oracle']['plain']):.3f}   "
              f"skip={np.mean(r['oracle']['skip']):.3f}   (UPPER BOUND, not deployable)")
    print(f"\n{'lever':<10}{'plain gen':>12}{'+skip(K) gen':>16}    "
          f"{'plain reten':>13}{'skip reten':>12}")
    print("-" * 86)
    for label, _ in LEVER_COMBOS:
        gp = r["results"][label]["plain"]
        gs = r["results"][label]["skip"]
        mp, ms = np.mean(gp), np.mean(gs)
        rp = mp / cont_p if cont_p > 1e-9 else 0.0
        rs = ms / cont_s if cont_s > 1e-9 else 0.0
        # plain row = the lever WITHOUT skip; +skip column = same levers WITH skip
        # (so the four rows x two columns ARE the full 8-cell ablation incl. triple).
        print(f"{label:<10}{mp:>12.3f}{ms:>16.3f}    {rp:>13.3f}{rs:>12.3f}")
    # Spell out the 8-cell mapping for the writeup.
    print("\n  cells: 'plain' col = {baseline,G,F,G+F} (no K);  '+skip' col = those")
    print("         + K. So '+skip G+F' IS the TRIPLE (G+F+K).")


def main(quick=False):
    seeds = (0, 1) if quick else (0, 1, 2)
    ft_epochs = 25 if quick else 40
    combos = [(4, 8), (4, 16), (6, 8), (6, 16)]
    if quick:
        combos = [(4, 8), (6, 8)]
    print("STACKED-LEVER ABLATION — cascaded single-spike TTFS, depth > d_max regime")
    print(f"levers: G=gain-correction(theta trim, CALIB)  F=genuine-STE-FT(TRAIN)  "
          f"K=input->decode concat skip(ROUTING)")
    print(f"seeds={seeds}  ft_epochs={ft_epochs}  target=0.5")
    all_r = []
    for depth, S in combos:
        r = run_ablation(depth, S, seeds, ft_epochs=ft_epochs, do_oracle=not quick)
        print_ablation(r)
        all_r.append(r)
    return all_r


if __name__ == "__main__":
    main(quick=(len(sys.argv) > 1 and sys.argv[1] == "quick"))
