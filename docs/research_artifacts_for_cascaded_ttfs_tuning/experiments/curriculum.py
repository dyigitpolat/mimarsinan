"""Direction D (H4) — multi-spike -> single-spike relaxation curriculum.

A multi-spike (rate-like) code is depth-robust: the decoded value is a SPIKE
COUNT, independent of *when* in the window the spikes land, so the cascade's
window-shortening-with-depth (the death cascade) does not attenuate it. The
single-spike TIMING code is the opposite: value = a ramp over ``[tau, T)`` whose
length shrinks with latency, so deep layers starve.

H4: train the continuous teacher into the deployed single-spike basin by
ANNEALING the code, not the resolution S. Every cascade neuron gets a knob
``k`` = the number of spikes it may emit (``k=1`` is the deployed single-spike
timing code; large ``k`` approaches a depth-robust rate code), and we anneal
``k -> 1``, recovering at each step. The DEPLOYED forward stays the bit-exact
single-spike cascade; ``k`` exists only at training time (a TRAINING-time
curriculum).

Everything here is self-contained: a cycle-accurate cascade simulator built from
the converted flow's perceptron params (W, b, activation_scale, input_scale,
latency-by-depth), VALIDATED to bit-match the lab's genuine ``cascade_forward``
at ``k=1`` (single segment). That simulator carries the ``k`` knob and the
gradient; the lab's genuine cascade_forward measures the deployed (k=1) accuracy
of the resulting weights, so a "win" is a real deployed win, not a proxy.

This file is a research prototype (Direction D). It imports cascade_lab + the
tested fixtures; it does not edit them.
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LAB = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "tests"), _LAB, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cascade_fixtures import (  # noqa: E402
    _SingleSegmentMLP,
    _calibrate_scales,
    cascade_forward,
    install_ttfs_nodes,
)
from cascade_lab import _accuracy, digits_task  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


# --------------------------------------------------------------------------- #
# Hard single-spike cascade reference (bit-exact with the deployed policy)
# --------------------------------------------------------------------------- #
#
# Genuine single-spike cascade (k=1), single segment, per the deployed policy
# (segment_policy_ttfs.TtfsSegmentPolicy.run_segment + TTFSActivation):
#
#   * The segment runs n_cycles = S + max_depth cycles.
#   * Layer 0 is the segment ENTRY (encoding=True). It takes the IDEAL real
#     pre-activation V0 = W0 @ x + b0, normalizes v0 = clamp(relu(V0)/scale0,0,1)
#     and emits a single TTFS spike at tau0 = round(S(1-v0)), held high after
#     (latch). This is the value->spike host encoder; it does NOT ramp-integrate
#     an input spike train.
#   * Layer d>=1 is a cascade neuron at latency d. In its window [d, d+S) it reads
#     the upstream perceptron's spike output from the PREVIOUS cycle (1-cycle
#     core delay), and ramp-integrates: ramp += weighted; mem += ramp + bias_norm;
#     fires once when mem >= 1 (thresholding "<=" => pre = mem-1 >= 0).
#     "weighted" = (in_scale/scale) * (W @ upstream_spike); bias_norm = b/scale.
#   * The OUTPUT decode = accum/S * scale, accum = sum over the window [d,d+S) of
#     the latched (max-so-far) single spike => ((S - tau_local)/S) * scale.
#
# The DELAY chain matters: layer 1 sees layer 0's spikes shifted by 1 cycle, etc.
# That late arrival is what shortens deep layers' effective ramp window -> the
# death cascade. The hard reference below reproduces this exactly; the soft
# k-spike model (KSpikeCascade) reduces to it at k=1.


def _hard_entry_spikes(v0_norm, S):
    """Single-spike train (S cycles) of an ENCODING entry value.

    The encoding TTFSActivation fires at the first cycle ``k`` with
    ``v_norm - 1/S + (k+1)/S >= 1`` => ``k_fire = ceil(S(1 - v_norm))`` (the
    membrane crosses the threshold), not ``round``. v<=0 never fires. Returns
    (S, B, F): a single spike at k_fire."""
    v = v0_norm.clamp(0, 1)
    kf = torch.ceil(S * (1.0 - v)).long()  # (B,F)
    kf = torch.where(v <= 0, torch.full_like(kf, S), kf)
    cyc = torch.arange(S, device=v0_norm.device).view(S, 1, 1)
    spike = (cyc == kf.unsqueeze(0)) & (kf.unsqueeze(0) < S)
    return spike.to(v0_norm.dtype)


def hard_cascade(kcas, x_norm, latency=True):
    """Bit-exact hard single-spike cascade forward (the deployed k=1 reference).

    Uses kcas.W / kcas.b / scales. Returns the decoded output (B, out)."""
    S = kcas.S
    B = x_norm.shape[0]
    depth = kcas.depth
    max_lat = (depth - 1) if latency else 0
    n_cycles = S + max_lat

    # Per-layer spike output buffer over all cycles.
    spikes = [None] * depth  # spikes[d]: (n_cycles, B, out_d)
    states = []
    for d in range(depth):
        W = kcas.W[d].detach()
        out_d = W.shape[0]
        states.append(dict(
            ramp=torch.zeros(B, out_d, dtype=x_norm.dtype),
            mem=torch.zeros(B, out_d, dtype=x_norm.dtype),
            fired=torch.zeros(B, out_d, dtype=x_norm.dtype),
            latched=torch.zeros(B, out_d, dtype=x_norm.dtype),
            accum=torch.zeros(B, out_d, dtype=x_norm.dtype),
            train=torch.zeros(n_cycles, B, out_d, dtype=x_norm.dtype),
        ))

    # entry value
    W0, b0 = kcas.W[0].detach(), kcas.b[0].detach()
    sc0 = max(kcas.scales[0], 1e-12)
    V0 = x_norm @ W0.T + b0
    v0n = (torch.relu(V0) / sc0).clamp(0, 1)
    entry = _hard_entry_spikes(v0n, S)  # (S, B, out0)

    for t in range(n_cycles):
        for d in range(depth):
            lat = d if latency else 0
            if t < lat or t >= lat + S:
                continue
            st = states[d]
            sc = max(kcas.scales[d], 1e-12)
            insc = max(kcas.in_scales[d], 1e-12)
            if d == 0:
                spike = entry[t - lat] if (t - lat) < S else torch.zeros_like(st["mem"])
            else:
                up = spikes[d - 1]
                src = up[t - 1] if t - 1 >= 0 else torch.zeros(B, kcas.W[d - 1].shape[0], dtype=x_norm.dtype)
                weighted = (src @ kcas.W[d].detach().T) * (insc / sc)
                bias_norm = (kcas.b[d].detach() / sc)
                st["ramp"] = st["ramp"] + weighted
                st["mem"] = st["mem"] + st["ramp"] + bias_norm
                fire = ((st["mem"] >= 1.0) & (st["fired"] < 1.0)).to(x_norm.dtype)
                st["fired"] = (st["fired"] + fire).clamp(max=1.0)
                spike = fire
            st["train"][t] = spike
            st["latched"] = torch.maximum(st["latched"], spike)
            st["accum"] = st["accum"] + st["latched"]
        for d in range(depth):
            spikes[d] = states[d]["train"]

    sc = max(kcas.scales[depth - 1], 1e-12)
    return (states[depth - 1]["accum"] / S) * sc


# --------------------------------------------------------------------------- #
# Soft k-spike cascade (differentiable; reduces to hard_cascade at k=1)
# --------------------------------------------------------------------------- #
class KSpikeCascade(torch.nn.Module):
    """A differentiable k-spike cascade built from a converted flow's params.

    Reproduces the deployed single-spike ramp-integrate cascade at ``k=1`` and
    relaxes toward a depth-robust multi-spike rate code as ``k`` grows. ``W``/``b``
    are trainable nn.Parameters initialized from the converted flow.

    Forward uses a STRAIGHT-THROUGH spike: the forward value is the hard
    (deployed) spike, and a soft surrogate carries the gradient. At ``k=1`` the
    forward is therefore byte-identical to ``hard_cascade``.
    """

    def __init__(self, flow, S: int, surrogate_alpha: float = 4.0):
        super().__init__()
        self.S = int(S)
        self.alpha = float(surrogate_alpha)
        Ws, bs, scales, in_scales, enc = [], [], [], [], []
        for p in flow.get_perceptrons():
            Ws.append(torch.nn.Parameter(p.layer.weight.detach().double().clone()))
            bs.append(torch.nn.Parameter(p.layer.bias.detach().double().clone()))
            scales.append(float(p.activation_scale))
            in_scales.append(float(p.input_activation_scale))
            enc.append(bool(getattr(p, "is_encoding_layer", False)))
        self.W = torch.nn.ParameterList(Ws)
        self.b = torch.nn.ParameterList(bs)
        self.scales = scales
        self.in_scales = in_scales
        self.enc = enc
        self.depth = len(Ws)

    def forward(self, x_norm, k: float = 1.0, *, latency=True):
        """x_norm: (B, in_dim) in [0,1]. Returns decoded output (B, out)."""
        S = self.S
        B = x_norm.shape[0]
        depth = self.depth
        max_lat = (depth - 1) if latency else 0
        n_cycles = S + max_lat
        a = self.alpha

        # entry value -> spike train (k-relaxed).
        W0, b0 = self.W[0], self.b[0]
        sc0 = max(self.scales[0], 1e-12)
        V0 = x_norm @ W0.T + b0
        v0n = (torch.relu(V0) / sc0).clamp(0, 1)
        entry = _soft_entry_train(v0n, S, k, a)  # (S, B, out0)

        out_dims = [self.W[d].shape[0] for d in range(depth)]
        zeros = [torch.zeros(B, od, dtype=x_norm.dtype, device=x_norm.device) for od in out_dims]
        states = [dict(ramp=zeros[d].clone(), mem=zeros[d].clone(), count=zeros[d].clone(),
                       latched=zeros[d].clone(), accum=zeros[d].clone()) for d in range(depth)]
        # per-layer per-cycle spike output (list, no in-place tensor writes).
        trains = [[zeros[d]] * n_cycles for d in range(depth)]

        for t in range(n_cycles):
            new_spikes = [None] * depth
            for d in range(depth):
                lat = d if latency else 0
                if t < lat or t >= lat + S:
                    new_spikes[d] = zeros[d]
                    continue
                st = states[d]
                sc = max(self.scales[d], 1e-12)
                insc = max(self.in_scales[d], 1e-12)
                if d == 0:
                    spike = entry[t - lat] if (t - lat) < S else zeros[d]
                else:
                    src = trains[d - 1][t - 1] if t - 1 >= 0 else zeros[d - 1]
                    weighted = (src @ self.W[d].T) * (insc / sc)
                    bias_norm = self.b[d] / sc
                    st["ramp"] = st["ramp"] + weighted
                    st["mem"] = st["mem"] + st["ramp"] + bias_norm
                    spike = _st_kspike(st["mem"], st["count"], k, a)
                    st["mem"] = st["mem"] - spike  # soft reset (count>1 needs it)
                    st["count"] = st["count"] + spike
                new_spikes[d] = spike
                st["latched"] = _smooth_max(st["latched"], spike)
                st["accum"] = st["accum"] + st["latched"]
            for d in range(depth):
                trains[d][t] = new_spikes[d]

        sc = max(self.scales[depth - 1], 1e-12)
        return (states[depth - 1]["accum"] / S) * sc


# --------------------------------------------------------------------------- #
# Straight-through / soft primitives
# --------------------------------------------------------------------------- #
def _heaviside_ste(pre, alpha):
    """Forward: 1.0*(pre>=0); backward: sigmoid'(alpha*pre)."""
    hard = (pre >= 0).to(pre.dtype)
    soft = torch.sigmoid(alpha * pre)
    return hard.detach() + (soft - soft.detach())


def _st_kspike(mem, count, k, alpha):
    """Straight-through k-budgeted fire: forward = hard(mem>=1 & count<k), grad
    through soft membrane and soft budget. At k=1 forward == single-spike latch."""
    may = _heaviside_ste(mem - 1.0, alpha)
    # budget gate: 1 while count <= k-1 (hard), soft grad.
    bhard = (count <= (k - 1.0 + 1e-9)).to(mem.dtype)
    bsoft = torch.sigmoid(alpha * (k - 0.5 - count))
    budget = bhard.detach() + (bsoft - bsoft.detach())
    return may * budget


def _smooth_max(a, b):
    """Forward = hard max; grad to both (straight-through to the argmax)."""
    return torch.maximum(a, b)


def _soft_entry_train(v_norm, S, k, alpha):
    """Entry spike train interpolating single-spike(k=1) -> rate(k=S), STE.

    k=1: single latched TTFS spike at tau=round(S(1-v)) (the deployed entry).
    k>1: a multi-spike train whose count/S tracks v (rate code), evenly spread.
    Convex blend by a smooth gate of (k-1); forward is the hard single spike at
    k=1 so the entry matches the deployed encode exactly there.
    """
    shape = v_norm.shape
    cyc = torch.arange(S, device=v_norm.device, dtype=v_norm.dtype).view((S,) + (1,) * len(shape))
    vb = v_norm.unsqueeze(0).clamp(0, 1)
    # hard single-spike (timing): encoding membrane crossing at k_fire=ceil(S(1-v)).
    kf = torch.ceil(S * (1.0 - vb))
    single_hard = ((cyc == kf) & (kf < S)).to(v_norm.dtype)
    # soft single-spike for grad (peak near the firing cycle, slides with v).
    tau_soft = S * (1.0 - vb)
    single_soft = torch.softmax(-alpha * (cyc - tau_soft).abs(), dim=0)
    single = single_hard.detach() + (single_soft - single_soft.detach())

    # rate (count): place n = round(S*v) spikes evenly; count/S = v exactly.
    n = torch.round(S * vb)
    idx = cyc  # 0..S-1
    # spike at cycle c iff floor(c * n / S) increments -> evenly spaced.
    step = torch.floor((idx + 1) * n / S) - torch.floor(idx * n / S)
    rate_hard = (step >= 1).to(v_norm.dtype)
    rate_soft = vb.expand((S,) + shape)  # grad surrogate: each cycle ~ v
    rate = rate_hard.detach() + (rate_soft - rate_soft.detach())

    # Blend gate: EXACTLY 0 at k<=1 (pure single-spike == deployed single spike),
    # linearly to 1 at k>=2. Continuous in k so the curriculum can be annealed.
    w_rate = min(max(float(k) - 1.0, 0.0), 1.0)
    return (1.0 - w_rate) * single + w_rate * rate


# --------------------------------------------------------------------------- #
# Lab plumbing
# --------------------------------------------------------------------------- #
def build_flow(depth, width, in_dim, n_classes, S, seed):
    """Build a converted flow + calibrated scales for the digits cascade."""
    torch.manual_seed(seed)
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    install_ttfs_nodes(flow, S)
    return base, flow, (xtr.double(), ytr, xte.double(), yte)


def _base_from_kcas(kcas, depth, width, in_dim, n_classes):
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    linears = [m for m in base.modules() if isinstance(m, torch.nn.Linear)]
    for lin, W, b in zip(linears, kcas.W, kcas.b):
        lin.weight.data = W.detach().float().clone()
        lin.bias.data = b.detach().float().clone()
    return base


def recalibrate_scales(kcas, depth, width, in_dim, n_classes, S, x_calib):
    """Re-derive per-layer activation scales from the CURRENT kcas weights via the
    converter's calibration (the deployment scales), and write them into kcas so
    the training forward uses the scales it will actually deploy with."""
    base = _base_from_kcas(kcas, depth, width, in_dim, n_classes)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, x_calib.double())
    install_ttfs_nodes(flow, S)
    kcas.scales = [float(p.activation_scale) for p in flow.get_perceptrons()]
    kcas.in_scales = [float(p.input_activation_scale) for p in flow.get_perceptrons()]
    return kcas


def reconvert_and_deploy(kcas, depth, width, in_dim, n_classes, S, seed, data):
    """Load kcas weights into a fresh ReLU MLP, convert, re-calibrate scales, and
    deploy through the LAB's genuine single-spike cascade. Returns (cont, gen)."""
    xtr, ytr, xte, yte = data
    base = _base_from_kcas(kcas, depth, width, in_dim, n_classes)
    with torch.no_grad():
        cont = _accuracy(base(xte.float()), yte)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256].double())
    install_ttfs_nodes(flow, S)
    gen = _accuracy(cascade_forward(flow, xte, S), yte)
    return float(cont), float(gen)


# --------------------------------------------------------------------------- #
# Curriculum training (the H4 experiment)
# --------------------------------------------------------------------------- #
def normalize_input(x):
    """digits pixels are already in [0,1]; the entry encoder clamps anyway."""
    return x.clamp(0, 1).double()


def train_kcascade(kcas, data, k, *, epochs, lr, depth, width, in_dim, n_classes, S,
                   recal_every=0, batch=256):
    """Train the k-spike cascade at a fixed code level ``k`` via CE on its output.

    ``recal_every`` > 0 re-derives deployment scales every that-many epochs (so
    training tracks the scales it will deploy with)."""
    xtr, ytr, xte, yte = data
    opt = torch.optim.Adam(kcas.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    xtr_n = normalize_input(xtr)
    n = xtr_n.shape[0]
    for ep in range(epochs):
        if recal_every and ep > 0 and ep % recal_every == 0:
            recalibrate_scales(kcas, depth, width, in_dim, n_classes, S, xtr[:256])
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = kcas(xtr_n[idx], k=k)
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
    return kcas


def make_teacher_kcascade(depth, width, in_dim, n_classes, S, seed, *,
                          init_train=100, dtype=torch.float32):
    """Train the continuous teacher once, convert+calibrate, and return a fresh
    KSpikeCascade snapshot of it (the shared init for every training arm)."""
    base, flow, data = build_flow(depth, width, in_dim, n_classes, S, seed)
    xtr, ytr, xte, yte = data
    from cascade_lab import train_continuous
    train_continuous(base, xtr, ytr, epochs=init_train)
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256].double())
    install_ttfs_nodes(flow, S)
    kcas = KSpikeCascade(flow, S).to(dtype)
    return kcas, data


def _clone_kcas(kcas):
    c = KSpikeCascade.__new__(KSpikeCascade)
    torch.nn.Module.__init__(c)
    c.S, c.alpha, c.depth = kcas.S, kcas.alpha, kcas.depth
    c.scales, c.in_scales, c.enc = list(kcas.scales), list(kcas.in_scales), list(kcas.enc)
    c.W = torch.nn.ParameterList([torch.nn.Parameter(W.detach().clone()) for W in kcas.W])
    c.b = torch.nn.ParameterList([torch.nn.Parameter(b.detach().clone()) for b in kcas.b])
    return c


def train_arm(kcas, data, schedule, *, depth, width, in_dim, n_classes, S,
              epochs_per_k, lr, subset, recal_each_stage=True, dtype=torch.float32):
    """Train one arm following ``schedule`` (list of k). Returns per-stage deploy.

    Total gradient steps = len(schedule) * epochs_per_k, identical across arms so
    the curriculum vs direct comparison is matched on compute."""
    xtr, ytr, xte, yte = data
    xn = normalize_input(xtr).to(dtype)[:subset]
    yt = ytr[:subset]
    lossf = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(kcas.parameters(), lr=lr)
    log = []
    for k in schedule:
        for _ in range(epochs_per_k):
            opt.zero_grad()
            out = kcas(xn, k=float(k))
            lossf(out, yt).backward()
            opt.step()
        if recal_each_stage:
            recalibrate_scales(kcas, depth, width, in_dim, n_classes, S, xtr[:256])
        cont, gen = reconvert_and_deploy(kcas, depth, width, in_dim, n_classes, S, 0, data)
        log.append((f"k={k}", round(cont, 4), round(gen, 4)))
    return log


def compare_arms(*, depth=3, width=48, in_dim=64, n_classes=10, S=8, seed=0,
                 epochs_per_k=12, lr=2e-3, subset=512, init_train=100,
                 curriculum=(4.0, 3.0, 2.0, 1.5, 1.0)):
    """Matched-compute comparison of the k-annealing curriculum vs direct k=1.

    Both arms start from the SAME teacher snapshot and run the SAME number of
    gradient steps (len(schedule)*epochs_per_k); only the k-schedule differs.
    Returns dict with init deploy + final deployed (k=1 genuine) acc per arm."""
    teacher, data = make_teacher_kcascade(depth, width, in_dim, n_classes, S, seed,
                                          init_train=init_train)
    init = reconvert_and_deploy(teacher, depth, width, in_dim, n_classes, S, 0, data)
    n_stages = len(curriculum)
    direct = train_arm(_clone_kcas(teacher), data, [1.0] * n_stages,
                       depth=depth, width=width, in_dim=in_dim, n_classes=n_classes,
                       S=S, epochs_per_k=epochs_per_k, lr=lr, subset=subset)
    curr = train_arm(_clone_kcas(teacher), data, list(curriculum),
                     depth=depth, width=width, in_dim=in_dim, n_classes=n_classes,
                     S=S, epochs_per_k=epochs_per_k, lr=lr, subset=subset)
    return {
        "seed": seed, "init": (round(init[0], 4), round(init[1], 4)),
        "direct_final": direct[-1], "direct_log": direct,
        "curriculum_final": curr[-1], "curriculum_log": curr,
    }


if __name__ == "__main__":
    import time
    print("=== sanity: k=1 forward bit-exact with deployed cascade ===")
    base, flow, data = build_flow(3, 64, 64, 10, 8, seed=0)
    kc = KSpikeCascade(flow, 8).double()
    x = data[2][:16]
    print("max diff soft(k=1) vs genuine:",
          float((cascade_forward(flow, x, 8) - kc(x, k=1.0)).abs().max()))
    print("\n=== curriculum vs direct-genuine, depth=3 S=8, seeds 0,1,2 ===")
    for seed in (0, 1, 2):
        t = time.time()
        r = compare_arms(seed=seed)
        print(f"seed={seed}  init(cont,gen)={r['init']}  "
              f"direct k=1={r['direct_final']}  curriculum={r['curriculum_final']}  "
              f"({time.time()-t:.0f}s)", flush=True)
