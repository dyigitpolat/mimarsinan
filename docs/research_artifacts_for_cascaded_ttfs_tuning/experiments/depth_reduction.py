"""DIRECTION E — Effective-depth reduction via skip / residual routing (H5).

The cascaded single-spike TTFS death-cascade is exponential in DEPTH: each hop
attenuates (deep layers fire late -> short ramp window -> attenuated value) and
the attenuation compounds multiplicatively, so deep layers starve to ~0. H5: if
a deep layer ALSO receives a copy of an EARLY layer's spike, that early spike
fired with low latency (long ramp window -> faithful value), so it survives the
cascade and bypasses the attenuating chain. Effective depth (for the surviving
signal) drops -> the depth budget d_max ~ T is extended.

HARDWARE FAITHFULNESS (the load-bearing question for this direction)
-------------------------------------------------------------------
A skip is realised by ``torch.cat([deep_hidden, early_hidden])`` feeding the next
linear layer. The converter maps ``torch.cat`` to a ``ConcatMapper`` — TRANSPARENT
routing inside ONE cascade segment (verified: segment_count stays 1, no host
ComputeOp cut). So the deployed cost of a skip is:

  * an EXTRA FAN-IN (extra synapses / a wider weight block) into the consumer
    core, fed by routing the EARLY core's already-emitted spike to a second
    destination. NO extra spike is emitted (the early neuron still fires once);
    the same spike is fanned out to an additional core.
  * within standard CMOS/RRAM crossbar capability (extra columns / extra axons),
    subject only to the per-core max-axon (fan-in) budget — exactly the kind of
    routing the mapper already coalesces. It is NOT a host op, NOT a per-layer
    sync, NOT a second spike. => DEPLOYABLE as a trained-weights / architecture
    change (category: architecture + trained params).

Contrast: a value-domain residual ``b + a`` maps to a ``ComputeOpMapper`` (host
op) that CUTS the segment, forcing a decode/host-add/re-encode boundary — that is
NOT what we want (it adds a sync point and a re-encode). We therefore use CONCAT
skips (in-segment routing), not value-domain adds.

We run on the shared fast harness only (cascade_lab + cascade_fixtures + src);
this file is a prototype and does not edit any shared file.

Experiments
-----------
E1  controlled graft: a good no-skip teacher, then a concat skip whose skip
    sub-weights start at ZERO (continuous output identical), fine-tuned through
    the GENUINE cascade — isolates the pure cascade-revival effect from training.
E2  architecture comparison at matched continuous quality: no-skip vs several
    skip topologies, reporting cont/gen/retention + attenuation, seeds x S x depth.
E3  depth-budget extension: d_max(S) (largest depth with gen_acc above chance+margin)
    for skip vs no-skip.
"""

from __future__ import annotations

import os
import sys

# Keep CPU contention low on a shared box (parallel agents): cap thread fan-out
# BEFORE importing torch so the limit takes effect.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

# cascade_lab.py lives one directory up (the research-artifacts root).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cascade_lab as L  # sets sys.path for cascade_fixtures + src
from cascade_fixtures import (
    _calibrate_scales,
    cascade_forward,
    install_ttfs_nodes,
    segment_count,
)
from mimarsinan.torch_mapping.converter import convert_torch_model

CHANCE = 0.10  # 10-class digits


def acc(logits, y):
    return float((logits.argmax(-1) == y).double().mean())


# --------------------------------------------------------------------------- #
# Architectures. Every layer (incl. classifier) is an nn.Linear + an nn.ReLU
# MODULE INSTANCE -> a perceptron in the converted flow. (The converter pattern-
# matches nn.ReLU module instances; functional torch.relu is NOT recognised and
# silently turns the whole net into host ComputeOps == no cascade. We assert
# segment_count==1 / perceptron count downstream to guard against that trap.)
# Skips use torch.cat (-> transparent ConcatMapper, in-segment routing), never a
# value-domain ``+`` (-> host ComputeOp that cuts the segment).
# --------------------------------------------------------------------------- #
def _lin_relu(a, b):
    return nn.Sequential(nn.Linear(a, b), nn.ReLU())


class PlainMLP(nn.Module):
    """depth stacked Linear+ReLU (== cascade_fixtures._SingleSegmentMLP)."""

    def __init__(self, depth, width=64, in_dim=64, n_classes=10):
        super().__init__()
        dims = [in_dim] + [width] * (depth - 1) + [n_classes]
        self.stages = nn.ModuleList(_lin_relu(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        return x


class InputSkipMLP(nn.Module):
    """Every hidden layer ALSO reads the raw input x (concat). Classifier reads
    the last hidden only (avoids the negative-logit ReLU-saturation pathology of
    skipping straight into the final ReLU). Skip = input fanned into deep cores."""

    def __init__(self, depth, width=64, in_dim=64, n_classes=10):
        super().__init__()
        self.in_dim = in_dim
        self.stages = nn.ModuleList()
        self.stages.append(_lin_relu(in_dim, width))
        for _ in range(depth - 2):
            self.stages.append(_lin_relu(width + in_dim, width))
        self.clf = _lin_relu(width, n_classes)

    def forward(self, x):
        h = self.stages[0](x)
        for k in range(1, len(self.stages)):
            h = self.stages[k](torch.cat([h, x], -1))
        return self.clf(h)


class InputSkipAllMLP(nn.Module):
    """STRONGEST variant: EVERY layer (hidden AND classifier) also reads the raw
    input x via concat. The input (encoding-layer) spike fires earliest -> fully
    faithful -> routing it to every core keeps a depth-independent faithful
    reference current alive at all depths => effective depth ~ 1 for the surviving
    path; no layer starves to 0. Deployment cost: fan the input axons out to every
    core (extra fan-in per core; no extra spike, no host op)."""

    def __init__(self, depth, width=64, in_dim=64, n_classes=10):
        super().__init__()
        self.in_dim = in_dim
        self.stages = nn.ModuleList()
        self.stages.append(_lin_relu(in_dim, width))
        for _ in range(depth - 2):
            self.stages.append(_lin_relu(width + in_dim, width))
        self.clf = _lin_relu(width + in_dim, n_classes)  # classifier reads input too

    def forward(self, x):
        h = self.stages[0](x)
        for k in range(1, len(self.stages)):
            h = self.stages[k](torch.cat([h, x], -1))
        return self.clf(torch.cat([h, x], -1))


class DenseSkipMLP(nn.Module):
    """DenseNet-style: hidden layer k reads concat(input, all earlier hiddens).
    Classifier reads the last hidden only. Maximal in-segment routing."""

    def __init__(self, depth, width=64, in_dim=64, n_classes=10):
        super().__init__()
        self.n_hidden = depth - 1
        self.stages = nn.ModuleList()
        for k in range(self.n_hidden):
            self.stages.append(_lin_relu(in_dim + k * width, width))
        self.clf = _lin_relu(width, n_classes)

    def forward(self, x):
        feats = [x]
        for k in range(self.n_hidden):
            feats.append(self.stages[k](torch.cat(feats, -1)))
        return self.clf(feats[-1])


class GraftSkipMLP(nn.Module):
    """A PlainMLP whose classifier ALSO reads layer-0's hidden via concat. The
    skip sub-block of the classifier weight starts at ZERO so the continuous
    output is identical to the plain net; only the skip weights are then trained
    (E1, controlled graft). Used to isolate the pure cascade-revival effect."""

    def __init__(self, depth, width=64, in_dim=64, n_classes=10):
        super().__init__()
        dims = [in_dim] + [width] * (depth - 1)
        self.stages = nn.ModuleList(_lin_relu(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.clf = _lin_relu(width + width, n_classes)
        with torch.no_grad():
            self.clf[0].weight[:, width:].zero_()  # skip sub-block = 0
        self.width = width

    def forward(self, x):
        feats = []
        for s in self.stages:
            x = s(x)
            feats.append(x)
        early = feats[0]
        return self.clf(torch.cat([feats[-1], early], -1))


# --------------------------------------------------------------------------- #
# Conversion + measurement
# --------------------------------------------------------------------------- #
def _convert_and_eval(base, xtr, xte, yte, S, *, in_dim=64, n_classes=10,
                      expect_percs=None):
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    segs = segment_count(flow)
    n_percs = len(flow.get_perceptrons())
    # Guard: a skip built from functional torch.relu silently degrades to host
    # ComputeOps (0 perceptrons, segs=0) => NOT a cascade. Every Linear+ReLU
    # stage must become a perceptron and the whole net must be ONE spike segment
    # (concat is transparent routing; a value-add would cut it).
    if n_percs == 0 or segs != 1:
        raise AssertionError(
            f"non-cascade conversion: {n_percs} perceptrons, {segs} segments "
            f"(skips must be torch.cat with nn.ReLU module instances)")
    if expect_percs is not None and n_percs != expect_percs:
        raise AssertionError(f"expected {expect_percs} perceptrons, got {n_percs}")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    gen = acc(cascade_forward(flow, xte, S), yte)
    return gen, segs, flow


def _atten_profile(flow, xte, S):
    teacher = L._capture_activation_means(flow, xte)
    cascade = L._cascade_decoded_means(flow, xte, S)
    out = []
    for k in range(len(flow.get_perceptrons())):
        t, c = teacher.get(k), cascade.get(k)
        if t is None or c is None:
            out.append(None)
            continue
        n = min(t.numel(), c.numel())
        tm = float(t[:n].clamp(min=0).mean())
        cm = float(c[:n].mean())
        out.append(round(cm / tm, 3) if tm > 1e-9 else None)
    return out


def train_arch_once(ctor, depth, seed, *, epochs=80, lr=3e-3):
    """Train continuous ONCE; return (base, xtr, xte, yte, cont_acc). The same
    trained net is then converted at multiple S (conversion is cheap)."""
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = L.digits_task(seed=seed + 1)
    base = ctor(depth)
    L.train_continuous(base, xtr, ytr, epochs=epochs, lr=lr)
    with torch.no_grad():
        cont = acc(base(xte.float()), yte)
    return base, xtr, xte, yte, round(cont, 4)


def eval_arch_at_S(base, xtr, xte, yte, cont, S):
    gen, segs, flow = _convert_and_eval(base, xtr, xte, yte, S)
    return {
        "cont": cont,
        "gen": round(gen, 4),
        "gap": round(cont - gen, 4),
        "retention": round(gen / cont, 3) if cont > 1e-9 else None,
        "segs": segs,
        "atten": _atten_profile(flow, xte, S),
    }


def run_arch(ctor, depth, S, seed, *, epochs=120, lr=3e-3):
    base, xtr, xte, yte, cont = train_arch_once(ctor, depth, seed, epochs=epochs, lr=lr)
    return eval_arch_at_S(base, xtr, xte, yte, cont, S)


def _mean_runs_multiS(ctor, depth, Ss, seeds, **kw):
    """Train once per seed, eval at every S. Returns {S: aggregated dict}."""
    per_seed = [train_arch_once(ctor, depth, s, **kw) for s in seeds]
    out = {}
    for S in Ss:
        rs = [eval_arch_at_S(b, xtr, xte, yte, c, S) for (b, xtr, xte, yte, c) in per_seed]
        out[S] = {
            "cont": float(np.mean([r["cont"] for r in rs])),
            "gen": float(np.mean([r["gen"] for r in rs])),
            "retention": float(np.mean([r["retention"] for r in rs if r["retention"] is not None])),
            "segs": rs[0]["segs"],
            "atten_example": rs[0]["atten"],
            "gen_seeds": [r["gen"] for r in rs],
            "cont_seeds": [r["cont"] for r in rs],
        }
    return out


# --------------------------------------------------------------------------- #
# E1 — controlled graft: train ONLY the zero-initialised skip weights through the
# GENUINE cascade, leaving the plain backbone fixed. Pure cascade-revival test.
# --------------------------------------------------------------------------- #
def experiment_graft(depth=3, S=8, seed=0, epochs=80, graft_epochs=60, lr=3e-3):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = L.digits_task(seed=seed + 1)

    # 1. train a plain backbone well (continuous).
    base = GraftSkipMLP(depth)
    # train the backbone (skip block still 0 => identical to plain net).
    L.train_continuous(base, xtr, ytr, epochs=epochs, lr=lr)
    with torch.no_grad():
        cont_before = acc(base(xte.float()), yte)
    gen_before, segs, _ = _convert_and_eval(base, xtr, xte, yte, S)

    # 2. unfreeze ONLY the skip sub-block of the classifier; train it through the
    #    GENUINE single-spike cascade (boundary STE on) so the gradient is the
    #    deployed dynamics, not the staircase.
    flow = convert_torch_model(base, (64,), 10, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    percs = flow.get_perceptrons()
    clf_layer = percs[-1].layer
    width = base.width
    for p in flow.parameters():
        p.requires_grad_(False)
    # only the skip columns of the classifier weight train.
    skip_mask = torch.zeros_like(clf_layer.weight)
    skip_mask[:, width:] = 1.0
    clf_layer.weight.requires_grad_(True)
    opt = torch.optim.Adam([clf_layer.weight], lr=1e-2)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    for _ in range(graft_epochs):
        opt.zero_grad()
        logits = cascade_forward(flow, xtr, S, grad=True, surrogate_temp=0.5)
        loss = lossf(logits, ytr)
        loss.backward()
        if clf_layer.weight.grad is not None:
            clf_layer.weight.grad.mul_(skip_mask)  # keep backbone columns frozen
        opt.step()
    with torch.no_grad():
        gen_after = acc(cascade_forward(flow, xte, S), yte)
    return {
        "cont_before": round(cont_before, 4),
        "gen_before": round(gen_before, 4),
        "gen_after": round(gen_after, 4),
        "lift": round(gen_after - gen_before, 4),
        "segs": segs,
    }


# --------------------------------------------------------------------------- #
# E4 — genuine-cascade-aware fine-tuning. Cold conversion only shows the raw
# representation limit; the deployable lever is to FINE-TUNE end-to-end through
# the genuine single-spike cascade (boundary STE on) so the optimizer learns to
# route the discriminative computation through the surviving short-effective-depth
# (skip) path. Tests whether the skip architecture has a reachable deployed basin
# that the plain architecture lacks.
# --------------------------------------------------------------------------- #
def experiment_finetune(ctor, depth=3, S=8, seed=0, epochs=80, ft_epochs=40,
                        lr=3e-3, ft_lr=2e-3):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = L.digits_task(seed=seed + 1)
    base = ctor(depth)
    L.train_continuous(base, xtr, ytr, epochs=epochs, lr=lr)
    with torch.no_grad():
        cont = acc(base(xte.float()), yte)

    flow = convert_torch_model(base, (64,), 10, device="cpu")
    if len(flow.get_perceptrons()) == 0 or segment_count(flow) != 1:
        raise AssertionError("non-cascade conversion")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    with torch.no_grad():
        gen_cold = acc(cascade_forward(flow, xte, S), yte)

    # fine-tune ALL weights through the genuine cascade (STE backward).
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=ft_lr)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    for _ in range(ft_epochs):
        opt.zero_grad()
        logits = cascade_forward(flow, xtr, S, grad=True, surrogate_temp=0.5)
        lossf(logits, ytr).backward()
        opt.step()
    with torch.no_grad():
        gen_ft = acc(cascade_forward(flow, xte, S), yte)
    return {"cont": round(cont, 4), "gen_cold": round(gen_cold, 4),
            "gen_ft": round(gen_ft, 4), "ft_lift": round(gen_ft - gen_cold, 4)}


def report_e4(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("E4  Genuine-cascade fine-tuning (deployable lever): cold-convert then")
    print("    fine-tune end-to-end THROUGH the cascade. plain vs skip, depth=3.")
    print("=" * 78)
    archs = [("plain", PlainMLP), ("input-skip", InputSkipMLP),
             ("input-skip-ALL", InputSkipAllMLP), ("dense-skip", DenseSkipMLP)]
    for S in (8, 16):
        print(f"\n--- S = {S} (mean over seeds {seeds}) ---")
        print(f"{'arch':<14}{'cont':>8}{'gen_cold':>10}{'gen_ft':>9}{'ft_lift':>9}")
        for name, ctor in archs:
            rs = [experiment_finetune(ctor, 3, S, s) for s in seeds]
            c = np.mean([r["cont"] for r in rs])
            gc = np.mean([r["gen_cold"] for r in rs])
            gf = np.mean([r["gen_ft"] for r in rs])
            print(f"{name:<14}{c:>8.3f}{gc:>10.3f}{gf:>9.3f}{gf - gc:>+9.3f}")


# --------------------------------------------------------------------------- #
# E1-control — disambiguation: is the E1 lift from the SKIP, or just from
# retraining the classifier through the cascade? Same protocol as E1 but with NO
# skip (a plain backbone, retrain the classifier weights+bias through the genuine
# cascade). The E1-minus-control gap is the skip's specific contribution.
# --------------------------------------------------------------------------- #
def experiment_graft_control(depth=3, S=8, seed=0, epochs=80, graft_epochs=60, lr=3e-3):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = L.digits_task(seed=seed + 1)
    base = PlainMLP(depth)
    L.train_continuous(base, xtr, ytr, epochs=epochs, lr=lr)
    flow = convert_torch_model(base, (64,), 10, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    with torch.no_grad():
        gen_before = acc(cascade_forward(flow, xte, S), yte)
    clf = flow.get_perceptrons()[-1].layer
    for p in flow.parameters():
        p.requires_grad_(False)
    train_params = [clf.weight]
    clf.weight.requires_grad_(True)
    if clf.bias is not None:
        clf.bias.requires_grad_(True)
        train_params.append(clf.bias)
    opt = torch.optim.Adam(train_params, lr=1e-2)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    for _ in range(graft_epochs):
        opt.zero_grad()
        lossf(cascade_forward(flow, xtr, S, grad=True, surrogate_temp=0.5), ytr).backward()
        opt.step()
    with torch.no_grad():
        gen_after = acc(cascade_forward(flow, xte, S), yte)
    return {"gen_before": round(gen_before, 4), "gen_after": round(gen_after, 4),
            "lift": round(gen_after - gen_before, 4)}


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def report_e2(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("E2  Architecture comparison (mean over seeds %s)" % (seeds,))
    print("    PlainMLP (no skip) vs InputSkip / DenseSkip — depth x S")
    print("=" * 78)
    archs = [("plain", PlainMLP), ("input-skip", InputSkipMLP),
             ("input-skip-ALL", InputSkipAllMLP), ("dense-skip", DenseSkipMLP)]
    Ss = (8, 16, 32)
    for depth in (3, 4, 5):
        print(f"\n--- depth = {depth} ---")
        header = f"{'arch':<12}{'S':>4}{'cont':>8}{'gen':>8}{'reten':>8}  atten(example seed0)"
        print(header)
        for name, ctor in archs:
            byS = _mean_runs_multiS(ctor, depth, Ss, seeds)
            for S in Ss:
                m = byS[S]
                print(f"{name:<12}{S:>4}{m['cont']:>8.3f}{m['gen']:>8.3f}"
                      f"{m['retention']:>8.3f}  {m['atten_example']}")


def report_e3(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("E3  Depth-budget extension: gen_acc vs depth at S=8 and S=16")
    print("=" * 78)
    archs = [("plain", PlainMLP), ("input-skip", InputSkipMLP),
             ("input-skip-ALL", InputSkipAllMLP), ("dense-skip", DenseSkipMLP)]
    depths = (2, 3, 4, 5, 6)
    # train each (arch, depth, seed) once, eval at both S.
    cache = {}
    for name, ctor in archs:
        for d in depths:
            cache[(name, d)] = _mean_runs_multiS(ctor, d, (8, 16), seeds)
    for S in (8, 16):
        print(f"\n--- S = {S} (gen_acc; chance={CHANCE}) ---")
        print(f"{'arch':<12}" + "".join(f"{'d'+str(d):>9}" for d in depths))
        for name, ctor in archs:
            row = f"{name:<12}"
            for d in depths:
                row += f"{cache[(name, d)][S]['gen']:>9.3f}"
            print(row)
        print(f"{'(cont)':<12}" + "".join(
            f"{cache[('plain', d)][S]['cont']:>9.3f}" for d in depths))


def report_e1(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("E1  Controlled graft (backbone FROZEN): train ONLY the zero-init skip")
    print("    weights through the GENUINE cascade. CONTROL = same, no skip, retrain")
    print("    the classifier weights. (E1 - control) = skip's specific contribution.")
    print("=" * 78)
    for S in (8, 16):
        print(f"\n--- S = {S} (depth=3) ---")
        print(f"{'seed':<6}{'gen_before':>11}{'skip_after':>11}{'ctrl_after':>11}"
              f"{'skip_lift':>10}{'ctrl_lift':>10}")
        sk, ct = [], []
        for s in seeds:
            r = experiment_graft(depth=3, S=S, seed=s)
            c = experiment_graft_control(depth=3, S=S, seed=s)
            sk.append(r["gen_after"])
            ct.append(c["gen_after"])
            print(f"{s:<6}{r['gen_before']:>11.3f}{r['gen_after']:>11.3f}"
                  f"{c['gen_after']:>11.3f}{r['lift']:>+10.3f}{c['lift']:>+10.3f}")
        print(f"mean S={S}: skip_after={np.mean(sk):.3f} ctrl_after={np.mean(ct):.3f} "
              f"=> skip-specific gain {np.mean(sk) - np.mean(ct):+.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    seeds = (0, 1, 2)
    if which in ("e2", "all"):
        report_e2(seeds)
    if which in ("e3", "all"):
        report_e3(seeds)
    if which in ("e4", "all"):
        report_e4(seeds)
    if which in ("e1", "all"):
        report_e1(seeds)
