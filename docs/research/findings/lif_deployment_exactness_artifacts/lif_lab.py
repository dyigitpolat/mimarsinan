"""LIF deployment exactness lab.

Drives the REPO kernels (lif_fire_and_reset / lif_core_contribute_and_fire /
to_uniform_spikes) end-to-end on quick-trained MNIST chains, measures every
commutation violation, and prototypes the analytical corrections.

Wire conventions mirror the deployed hybrid executor
(src/mimarsinan/models/spiking/hybrid/lif_step.py):
  - wire weights  W~ = W * theta_in / theta_out   (perceptron_transformer.py:110)
  - wire bias     b~ = b / theta_out              (perceptron_transformer.py:122)
  - threshold 1.0 (wire units), hw_bias added EVERY active cycle
  - entry encode: r = clamp(v/theta, 0, 1); n = round(r*T) Uniform spikes
    (spike_modes.to_uniform_spikes; all live channels pulse at cycle 0)
  - decode: counts / T (segment_boundary.py:173-178); final logits = counts
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_nn

sys.path.insert(0, "/home/yigit/repos/research_stuff/mimarsinan/src")

from mimarsinan.models.nn.lif_kernels import lif_fire_and_reset
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire
from mimarsinan.chip_simulation.recording.spike_modes import to_uniform_spikes

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  # COMPUTE_DTYPE (models/spiking/spiking_config.py)
SEED = 0
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/yigit/repos/research_stuff/mimarsinan/datasets"

THRESHOLDING_MODE = "<"  # t0_01 lif config; "<=" checked in twin tests


# --------------------------------------------------------------------------
# closed-form count function F (the staircase the deployed kernel realizes)
# --------------------------------------------------------------------------

def count_fn(x: torch.Tensor, mode: str) -> torch.Tensor:
    """Terminal spike count for total charge x (theta=1), timing-ideal.

    "<=" : count of integers k>=1 with k <= x  -> floor(x)
    "<"  : count of integers k>=1 with k <  x  -> ceil(x) - 1 (x - 1 if integer)
    """
    if mode == "<=":
        return torch.floor(x)
    return torch.ceil(x - 1.0)


# --------------------------------------------------------------------------
# Part A: twin checks against the repo kernels
# --------------------------------------------------------------------------

def twin_checks() -> dict:
    torch.manual_seed(1234)
    res = {}

    # A1: to_uniform_spikes delivers exactly round(r*T) spikes, first at cycle 0
    for T in (4, 8, 16, 32):
        r = torch.rand(512, dtype=DTYPE)
        train = torch.stack([to_uniform_spikes(r, t, T) for t in range(T)]).to(DTYPE)
        n = torch.round(r * T)
        assert torch.equal(train.sum(0), n), f"uniform count mismatch T={T}"
        live = n >= 1
        assert torch.all(train[0][live] == 1.0), "live channels must pulse at cycle 0"
    res["uniform_encode_exact_count"] = True
    res["uniform_encode_all_live_pulse_cycle0"] = True

    # A2: single neuron, constant drive: kernel count == F(Q_T) for both modes
    for mode in ("<", "<="):
        for T in (4, 8, 16, 32):
            d = torch.linspace(-0.3, 1.3, 257, dtype=DTYPE).unsqueeze(1)
            memb = torch.zeros_like(d)
            c = torch.zeros_like(d)
            th = torch.tensor(1.0, dtype=DTYPE)
            for _ in range(T):
                memb = memb + d
                f = lif_fire_and_reset(memb, th, thresholding_mode=mode,
                                       firing_mode="Default", output_dtype=DTYPE)
                c = c + f
            ref = torch.clamp(count_fn(d.squeeze(1) * T, mode), 0, T)
            got = c.squeeze(1)
            assert torch.equal(got, ref), f"constant-drive law failed {mode} T={T}"
    res["constant_drive_count_law_exact"] = True

    # A3: full-matrix kernel loop == our vectorized hop executor on random configs
    for mode in ("<", "<="):
        T, B, n_in, n_out = 8, 16, 32, 24
        W = (torch.randn(n_out, n_in, dtype=DTYPE) * 0.3)
        b = torch.randn(n_out, dtype=DTYPE) * 0.05
        r = torch.rand(B, n_in, dtype=DTYPE)
        train = torch.stack([to_uniform_spikes(r, t, T) for t in range(T)]).to(DTYPE)
        # repo kernel path
        memb = torch.zeros(B, n_out, dtype=DTYPE)
        counts_k = torch.zeros(B, n_out, dtype=DTYPE)
        th = torch.tensor(1.0, dtype=DTYPE)
        for t in range(T):
            f = lif_core_contribute_and_fire(
                memb, W, train[t], th, hw_bias=b,
                thresholding_mode=mode, firing_mode="Default", output_dtype=DTYPE)
            counts_k = counts_k + f
        out_train, counts_v, memb_v, diag = run_hop(
            train, W, b, torch.ones(n_out, dtype=DTYPE), mode)
        assert torch.equal(counts_k, counts_v), "hop executor != repo kernel"
        assert torch.allclose(memb, memb_v), "membrane mismatch"
    res["hop_executor_bit_equal_repo_kernel"] = True
    return res


# --------------------------------------------------------------------------
# deployed hop executor (repo semantics, vectorized, with diagnostics)
# --------------------------------------------------------------------------

def run_hop(train: torch.Tensor, W: torch.Tensor, b: torch.Tensor,
            theta: torch.Tensor, mode: str):
    """One spiking hop. train: (T,B,n_in) -> out_train (T,B,n_out).

    Exactly memb += W@s + b; fire per comparator; subtractive reset by theta
    (lif_core_step.py:22-32 + lif_kernels.py:25-37, Default reset).
    theta: per-neuron threshold vector (per-core scalar on chip; per-neuron
    realizable exactly by row scaling).
    Returns (out_train, counts, final_memb, diagnostics).
    """
    T, B, _ = train.shape
    n_out = W.shape[0]
    memb = torch.zeros(B, n_out, dtype=DTYPE, device=train.device)
    counts = torch.zeros(B, n_out, dtype=DTYPE, device=train.device)
    out_train = torch.zeros(T, B, n_out, dtype=DTYPE, device=train.device)
    Q = torch.zeros(B, n_out, dtype=DTYPE, device=train.device)
    Qmax = torch.full((B, n_out), -1e30, dtype=DTYPE, device=train.device)
    for t in range(T):
        # bit-faithful to lif_core_step.py:22-24 (same matmul orientation)
        contrib = torch.matmul(W, train[t].T).T + b
        Q = Q + contrib
        Qmax = torch.maximum(Qmax, Q)
        memb = memb + contrib
        if mode == "<":
            fired = (memb > theta).to(DTYPE)
        else:
            fired = (memb >= theta).to(DTYPE)
        memb = memb - fired * theta
        counts = counts + fired
        out_train[t] = fired
    diag = {"Q_T": Q, "Q_max": Qmax}
    return out_train, counts, memb, diag


# --------------------------------------------------------------------------
# Part B: models
# --------------------------------------------------------------------------

class Chain(nn.Module):
    def __init__(self, widths):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1))

    def forward(self, x):
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F_nn.relu(x)
        return x

    def activations(self, x):
        acts = []
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F_nn.relu(x)
            acts.append(x)
        return acts


def load_mnist():
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))])
    tr = datasets.MNIST(DATA_DIR, train=True, download=False, transform=tf)
    te = datasets.MNIST(DATA_DIR, train=False, download=False, transform=tf)
    return tr, te


def train_model(widths, epochs=4, lr=3e-3, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, te = load_mnist()
    model = Chain(widths).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
    loader = torch.utils.data.DataLoader(tr, batch_size=128, shuffle=True,
                                         num_workers=2, drop_last=True,
                                         generator=torch.Generator().manual_seed(seed))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(loader))
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.view(xb.shape[0], -1).to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            loss = F_nn.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
    model.eval()
    return model


def tensor_split(seed=SEED, n_test=2000, n_calib=4000):
    tr, te = load_mnist()
    g = torch.Generator().manual_seed(seed)
    te_idx = torch.randperm(len(te), generator=g)[:n_test]
    tr_idx = torch.randperm(len(tr), generator=g)[:n_calib]
    def grab(ds, idx):
        xs = torch.stack([ds[i][0] for i in idx.tolist()])
        ys = torch.tensor([ds[i][1] for i in idx.tolist()])
        return xs.view(xs.shape[0], -1).to(DEVICE), ys.to(DEVICE)
    return grab(te, te_idx), grab(tr, tr_idx)


# --------------------------------------------------------------------------
# Part C: conversion + deployment arms
# --------------------------------------------------------------------------

def count_quantile(acts: torch.Tensor, q=0.99) -> float:
    """Pipeline count-quantile theta (activation_scale_policy.py: positive mass)."""
    pos = acts.reshape(-1)
    pos = pos[pos > 1e-9].float()
    if pos.numel() == 0:
        return 1e-6
    return max(float(torch.quantile(pos, q)), 1e-6)


class Converted:
    """Wire-domain parameters of the chain: hop 0 is the host encoder."""

    def __init__(self, model: Chain, calib_x: torch.Tensor, q=0.99):
        with torch.no_grad():
            acts = model.activations(calib_x.to(DEVICE))
        self.thetas = [count_quantile(a, q) for a in acts]
        self.model = model
        Ws, bs = [], []
        for i, l in enumerate(model.layers):
            W = l.weight.data.to(DTYPE)
            b = l.bias.data.to(DTYPE)
            if i == 0:
                continue  # encoder hop stays on host (subsume placement)
            Ws.append(W * self.thetas[i - 1] / self.thetas[i])
            bs.append(b / self.thetas[i])
        self.W = [w.to(DEVICE) for w in Ws]
        self.b = [x.to(DEVICE) for x in bs]
        self.n_hops = len(Ws)

    def encoder_rates(self, x):
        with torch.no_grad():
            v = F_nn.relu(self.model.layers[0](x.to(DEVICE)))
        return torch.clamp(v.to(DTYPE) / self.thetas[0], 0.0, 1.0)

    def float_logits(self, x):
        with torch.no_grad():
            return self.model(x.to(DEVICE))

    def ann_reference_rates(self, x, T, half_step: bool, mode=THRESHOLDING_MODE):
        """Composed-nearest(-or-floor) staircase: the commutation target."""
        r = self.encoder_rates(x)
        rates = [r]
        hs = 0.5 if half_step else 0.0
        for k in range(self.n_hops):
            z = r @ self.W[k].T + self.b[k]
            c = torch.clamp(count_fn(T * z + hs, mode), 0.0, float(T))
            r = c / T
            rates.append(r)
        return rates


def deploy(conv: Converted, x: torch.Tensor, T: int, *,
           half_step=True, guard=None, col_gain=None, col_bias=None,
           memb_readout=False, mode=THRESHOLDING_MODE, collect_diag=False):
    """Run the deployed chain through the kernel executor.

    guard:    list of per-neuron h vectors (theta -> 1+h, hw_bias += h/T,
              downstream decompensation folded into consumer W/b).
    col_gain: per-hop per-INPUT-channel gain a (folded into consumer columns).
    col_bias: per-hop per-INPUT-channel shift c (folded into consumer bias).
    Returns (logits_value, per-hop rates list, diagnostics list).
    """
    r0 = conv.encoder_rates(x)
    B = r0.shape[0]
    train = torch.stack([to_uniform_spikes(r0, t, T) for t in range(T)]).to(DTYPE)
    rates = [r0]
    diags = []
    counts = None
    memb = None
    theta_last = None
    for k in range(conv.n_hops):
        W = conv.W[k].clone()
        b = conv.b[k].clone()
        n_out = W.shape[0]
        # fold input-channel affine correction r' = a*r + c (consumer-side):
        # z = W~(a*r + c) + b~  =>  W <- W~*a,  b <- b~ + W~ @ c
        if col_bias is not None and col_bias[k] is not None:
            b = b + conv.W[k] @ col_bias[k]
        if col_gain is not None and col_gain[k] is not None:
            W = W * col_gain[k].unsqueeze(0)
        # guard decompensation for the PRODUCER hop's guard (input side)
        if guard is not None and k > 0 and guard[k - 1] is not None:
            h_in = guard[k - 1]
            W = W * (1.0 + h_in).unsqueeze(0)
            b = b - conv.W[k] @ (h_in / T)
        h = guard[k] if guard is not None else None
        theta = torch.ones(n_out, dtype=DTYPE, device=DEVICE)
        if h is not None:
            theta = theta + h
            b = b + h / T
        if half_step:
            b = b + theta / (2.0 * T)
        out_train, c, m, diag = run_hop(train, W, b, theta, mode)
        if collect_diag:
            hs_charge = theta / 2.0 if half_step else torch.zeros_like(theta)
            Q = diag["Q_T"]
            c_star = torch.clamp(count_fn(Q / theta, mode), 0.0, float(T))
            over = torch.clamp(c - c_star, min=0)
            under = torch.clamp(c_star - c, min=0)
            diags.append({
                "hop": k,
                "overcount_rate": float(over.mean() / T),
                "deficit_rate": float(under.mean() / T),
                "overcount_frac": float((over > 0).to(DTYPE).mean()),
                "deficit_frac": float((under > 0).to(DTYPE).mean()),
                "mean_overshoot": float(torch.clamp(
                    diag["Q_max"] - torch.clamp(diag["Q_T"], min=0), min=0).mean()),
            })
        train = out_train
        counts, memb, theta_last = c, m, theta
        rates.append(c / T)
    # readout decode
    if memb_readout:
        Qhat = counts * theta_last + memb  # exact charge identity
        if guard is not None and guard[-1] is not None:
            Qhat = Qhat - guard[-1] / T * T  # remove the guard's own +h charge
        if half_step:
            Qhat = Qhat - theta_last / 2.0
        logits = conv.thetas[-1] * Qhat / T
    else:
        c = counts
        if guard is not None and guard[-1] is not None:
            c = c * (1.0 + guard[-1]) - guard[-1]  # host-side decompensation
        logits = conv.thetas[-1] * c / T
    return logits, rates, diags


# --------------------------------------------------------------------------
# calibration: sequential residual-mean / LS-affine folds, guard hazard
# --------------------------------------------------------------------------

def calibrate_affine(conv: Converted, calib_x, T, *, half_step, guard=None,
                     fit_gain: bool, mode=THRESHOLDING_MODE):
    """Sequential per-channel affine (a, c): deployed rate -> float-ANN rate.

    Layer-by-layer: run the deployed chain with corrections applied upstream,
    fit hop k's OUTPUT channels against the float-ANN clamped rate target,
    fold into the CONSUMER (returned as col_gain/col_bias lists indexed by the
    consuming hop's input side). Readout hop gets a host-side per-class affine.
    """
    with torch.no_grad():
        acts = conv.model.activations(calib_x.to(DEVICE))
    targets = []
    for k in range(conv.n_hops):
        v = acts[k + 1].to(DTYPE)
        targets.append(torch.clamp(v / conv.thetas[k + 1], 0.0, 1.0))

    col_gain = [None] * conv.n_hops
    col_bias = [None] * conv.n_hops
    host_affine = None
    for k in range(conv.n_hops):
        logits, rates, _ = deploy(conv, calib_x, T, half_step=half_step,
                                  guard=guard, col_gain=col_gain,
                                  col_bias=col_bias, mode=mode)
        r_dep = rates[k + 1]
        r_tgt = targets[k]
        if fit_gain:
            var = r_dep.var(dim=0, unbiased=False)
            cov = ((r_dep - r_dep.mean(0)) * (r_tgt - r_tgt.mean(0))).mean(0)
            a = torch.where(var > 1e-10, cov / torch.clamp(var, min=1e-10),
                            torch.ones_like(var))
            a = torch.clamp(a, 0.25, 4.0)
        else:
            a = torch.ones_like(r_dep.mean(0))
        c = r_tgt.mean(0) - a * r_dep.mean(0)
        if k < conv.n_hops - 1:
            col_gain[k + 1] = a
            col_bias[k + 1] = c
        else:
            host_affine = (a, c)
    return col_gain, col_bias, host_affine


def calibrate_guard(conv: Converted, calib_x, T, *, half_step, q=0.99,
                    mode=THRESHOLDING_MODE):
    """Per-neuron overshoot hazard h_j = q-quantile of (max_t Q_t - max(Q_T,0))+.

    Sequential: hop k's hazard is measured on the GUARDED upstream stream
    (with the exact downstream decompensation folded in), then hop k is
    re-run guarded to produce the next hop's arrivals.
    """
    guards: list = []
    r0 = conv.encoder_rates(calib_x)
    train = torch.stack([to_uniform_spikes(r0, t, T) for t in range(T)]).to(DTYPE)
    for k in range(conv.n_hops):
        W = conv.W[k].clone()
        b = conv.b[k].clone()
        if k > 0 and guards[k - 1] is not None:
            h_in = guards[k - 1]
            W = W * (1.0 + h_in).unsqueeze(0)
            b = b - conv.W[k] @ (h_in / T)
        n_out = W.shape[0]
        # unguarded probe to measure the hazard
        theta1 = torch.ones(n_out, dtype=DTYPE, device=DEVICE)
        b1 = b + (theta1 / (2.0 * T) if half_step else 0.0)
        _, _, _, diag = run_hop(train, W, b1, theta1, mode)
        os_ = torch.clamp(diag["Q_max"] - torch.clamp(diag["Q_T"], min=0.0), min=0.0)
        h = torch.quantile(os_, q, dim=0).to(DTYPE)
        guards.append(h)
        # guarded re-run to feed the next hop
        theta_g = 1.0 + h
        bg = b + h / T
        if half_step:
            bg = bg + theta_g / (2.0 * T)
        out_train, _, _, _ = run_hop(train, W, bg, theta_g, mode)
        train = out_train
    return guards


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def accuracy(logits, y):
    return float((logits.argmax(dim=1) == y).to(torch.float32).mean())


def hop_agreement(conv, rates_dep, rates_ref):
    out = []
    for k in range(1, len(rates_dep)):
        d = (rates_dep[k] - rates_ref[k]).abs()
        out.append({"hop": k, "mad_rate": float(d.mean()),
                    "p99_rate": float(torch.quantile(d.float(), 0.99))})
    return out


# --------------------------------------------------------------------------
# Part D: unequal-depth join head-loss / stale-buffer demo (gated executor)
# --------------------------------------------------------------------------

def join_demo(T=8, seed=7):
    """3-deep chain + direct shallow edge joining at C; gated window semantics
    exactly as lif_step.py:117-159 (fills read prev-cycle buffers; a core's
    buffer holds its LAST spike after its window ends)."""
    torch.manual_seed(seed)
    n = 16
    Wa = torch.randn(n, n, dtype=DTYPE) * 0.35
    Wb = [torch.randn(n, n, dtype=DTYPE) * 0.35 for _ in range(3)]
    Wc_deep = torch.randn(n, n, dtype=DTYPE) * 0.3
    Wc_shallow = torch.randn(n, n, dtype=DTYPE) * 0.3
    r_in = torch.rand(64, n, dtype=DTYPE)
    train_in = torch.stack([to_uniform_spikes(r_in, t, T) for t in range(T)]).to(DTYPE)

    def gated_run(latencies, relay_depth):
        # cores: A(lat 0), B1..B3 (1..3), optional relays R1..R? on shallow path,
        # C (lat max+1). Global loop faithful to the executor.
        cores = {"A": {"lat": 0, "W": Wa, "in": "input"}}
        prev = "A"
        for i in range(3):
            cores[f"B{i+1}"] = {"lat": i + 1, "W": Wb[i], "in": prev}
            prev = f"B{i+1}"
        shallow_src = "A"
        for i in range(relay_depth):
            # 1+eps: under the strict "<" comparator a unit charge NEVER fires
            # (memb == theta), so an exact-identity relay is dead. Itself a
            # violated-precondition finding for integer-arithmetic deployments.
            cores[f"R{i+1}"] = {"lat": i + 1,
                                "W": torch.eye(n, dtype=DTYPE) * (1.0 + 1e-9),
                                "in": shallow_src}
            shallow_src = f"R{i+1}"
        c_lat = max(cores["B3"]["lat"], cores[shallow_src]["lat"]) + 1
        cores["C"] = {"lat": c_lat, "W": None, "in": ("B3", shallow_src)}
        total = c_lat + T
        B = r_in.shape[0]
        buf = {k: torch.zeros(B, n, dtype=DTYPE) for k in cores}
        memb = {k: torch.zeros(B, n, dtype=DTYPE) for k in cores}
        counts_c = torch.zeros(B, n, dtype=DTYPE)
        th = torch.tensor(1.0, dtype=DTYPE)
        for cyc in range(total):
            fills = {}
            for name, core in cores.items():
                L = core["lat"]
                if not (L <= cyc < L + T):
                    continue
                if core["in"] == "input":
                    lc = cyc - L
                    fills[name] = train_in[lc] if 0 <= lc < T else torch.zeros(B, n, dtype=DTYPE)
                elif isinstance(core["in"], tuple):
                    fills[name] = (buf[core["in"][0]], buf[core["in"][1]])
                else:
                    fills[name] = buf[core["in"]]
            for name, core in cores.items():
                L = core["lat"]
                if not (L <= cyc < L + T):
                    continue
                if name == "C":
                    d, s = fills[name]
                    contrib = d @ Wc_deep.T + s @ Wc_shallow.T
                else:
                    contrib = fills[name] @ core["W"].T
                memb[name] = memb[name] + contrib
                f = lif_fire_and_reset(memb[name], th, thresholding_mode="<",
                                       firing_mode="Default", output_dtype=DTYPE)
                buf[name] = f
                if name == "C":
                    counts_c = counts_c + f
        return counts_c

    c_join = gated_run(None, relay_depth=0)          # shallow gap = 4
    c_balanced = gated_run(None, relay_depth=3)      # relays equalize depth
    mad = float((c_join - c_balanced).abs().mean()) / T
    frac = float(((c_join - c_balanced).abs() > 0).to(DTYPE).mean())
    return {"T": T, "mad_rate_join_vs_balanced": mad,
            "frac_neurons_differ": frac,
            "mean_count_join": float(c_join.mean()),
            "mean_count_balanced": float(c_balanced.mean())}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    t0 = time.time()
    results = {"seed": SEED, "thresholding_mode": THRESHOLDING_MODE}
    print("== twin checks ==")
    results["twin"] = twin_checks()
    print(json.dumps(results["twin"], indent=1))

    print("== training ==")
    vehicles = {
        "mlp3": [784, 128, 128, 10],
        "chain9": [784, 128, 128, 128, 128, 128, 128, 128, 128, 10],
    }
    (test_x, test_y), (calib_x, _) = tensor_split()
    results["vehicles"] = {}
    for name, widths in vehicles.items():
        model = train_model(widths)
        conv = Converted(model, calib_x)
        with torch.no_grad():
            fl = conv.float_logits(test_x)
        acc_f = accuracy(fl, test_y)
        vres = {"widths": widths, "float_acc": acc_f,
                "thetas": [round(t, 4) for t in conv.thetas]}
        print(f"{name}: float acc {acc_f:.4f} thetas {vres['thetas']}")

        for T in (4, 8, 16, 32):
            tv = {}
            # references
            ann_floor = conv.ann_reference_rates(test_x, T, half_step=False)
            ann_near = conv.ann_reference_rates(test_x, T, half_step=True)
            tv["ann_floor_acc"] = accuracy(conv.thetas[-1] * ann_floor[-1], test_y)
            tv["ann_nearest_acc"] = accuracy(conv.thetas[-1] * ann_near[-1], test_y)

            # arm 0: floor (no half-step)
            lg, rates, diags = deploy(conv, test_x, T, half_step=False,
                                      collect_diag=True)
            tv["a0_floor"] = {"acc": accuracy(lg, test_y),
                              "hops": hop_agreement(conv, rates, ann_floor),
                              "diag": diags}
            # arm 1: half-step (shipped default)
            lg, rates, diags = deploy(conv, test_x, T, half_step=True,
                                      collect_diag=True)
            tv["a1_halfstep"] = {"acc": accuracy(lg, test_y),
                                 "hops": hop_agreement(conv, rates, ann_near),
                                 "diag": diags}
            # arm 2: + residual-mean bias fold
            cg, cb, ha = calibrate_affine(conv, calib_x, T, half_step=True,
                                          fit_gain=False)
            lg, rates, _ = deploy(conv, test_x, T, half_step=True,
                                  col_gain=cg, col_bias=cb)
            if ha is not None:
                lg = conv.thetas[-1] * (ha[0] * (lg / conv.thetas[-1]) + ha[1])
            tv["a2_biasfold"] = {"acc": accuracy(lg, test_y),
                                 "hops": hop_agreement(conv, rates, ann_near)}
            # arm 3: + LS gain
            cg, cb, ha = calibrate_affine(conv, calib_x, T, half_step=True,
                                          fit_gain=True)
            lg, rates, _ = deploy(conv, test_x, T, half_step=True,
                                  col_gain=cg, col_bias=cb)
            if ha is not None:
                lg = conv.thetas[-1] * (ha[0] * (lg / conv.thetas[-1]) + ha[1])
            tv["a3_affine"] = {"acc": accuracy(lg, test_y),
                               "hops": hop_agreement(conv, rates, ann_near)}
            # arm 4: guard (on top of half-step only)
            guards = calibrate_guard(conv, calib_x, T, half_step=True)
            lg, rates, diags = deploy(conv, test_x, T, half_step=True,
                                      guard=guards, collect_diag=True)
            tv["a4_guard"] = {"acc": accuracy(lg, test_y),
                              "mean_h": [float(h.mean()) for h in guards],
                              "diag": diags}
            # arm 5: membrane readout on top of a1 and a3
            lg, _, _ = deploy(conv, test_x, T, half_step=True, memb_readout=True)
            tv["a5_membread_on_a1"] = {"acc": accuracy(lg, test_y)}
            cg, cb, ha = calibrate_affine(conv, calib_x, T, half_step=True,
                                          fit_gain=True)
            lg, _, _ = deploy(conv, test_x, T, half_step=True, col_gain=cg,
                              col_bias=cb, memb_readout=True)
            tv["a5_membread_on_a3"] = {"acc": accuracy(lg, test_y)}
            vres[f"T{T}"] = tv
            print(f"  T={T}: floorANN {tv['ann_floor_acc']:.4f} "
                  f"nearANN {tv['ann_nearest_acc']:.4f} | "
                  f"a0 {tv['a0_floor']['acc']:.4f} a1 {tv['a1_halfstep']['acc']:.4f} "
                  f"a2 {tv['a2_biasfold']['acc']:.4f} a3 {tv['a3_affine']['acc']:.4f} "
                  f"a4 {tv['a4_guard']['acc']:.4f} "
                  f"a5(a1) {tv['a5_membread_on_a1']['acc']:.4f} "
                  f"a5(a3) {tv['a5_membread_on_a3']['acc']:.4f}")
        results["vehicles"][name] = vres

    print("== join demo ==")
    results["join_demo"] = {f"T{t}": join_demo(T=t) for t in (4, 8, 16)}
    print(json.dumps(results["join_demo"], indent=1))

    results["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"done in {results['wall_seconds']}s -> results.json")


if __name__ == "__main__":
    main()
