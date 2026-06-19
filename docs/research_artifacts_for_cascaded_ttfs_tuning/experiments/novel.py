"""DIRECTION G (Phase 2) — NON-OBVIOUS mechanisms targeting the gain-distortion root.

Phase 1 (artifacts 10-15, 20_phase1_synthesis) proved the death cascade is a
CORRECTABLE per-layer decode GAIN-DISTORTION, NOT a capacity limit:

  A spike arriving at consumer-local cycle ``a`` deposits a latched ramp; the
  consumer double-integrates it, so by end of window its membrane contribution is
        C(a) = (S-a)(S-a+1)/2  ~  (T-tau)^2 / 2        [QUADRATIC]
  whereas a faithful linear decode wants
        L(a) = (S-a)            ~  (T-tau)              [LINEAR]
  so the per-spike GAIN  R(a) = C/L = (S-a+1)/2  falls 4.5 -> 1.0 across the
  window (S=8): EARLY (large-value) spikes are over-weighted ~4.5x vs LATE
  (small-value, deep-layer) spikes.  Deep layers fire late -> under-decoded ->
  shrink -> next layer fires later -> geometric death (d_max ~ 0.56*sqrt(S)).

The Phase-1 deployable fix is a per-layer multiplicative threshold trim
(``mean_target`` in precomp.py): ~0.69 mean on PRIMARY (depth-3 S=8, seeds 0-2),
oracle upper bound ~0.91, baseline ~0.077.  THIS FILE asks: are there NON-obvious
mechanisms that BEAT or COMPLEMENT the simple theta trim by attacking the
quadratic gain MORE DIRECTLY?

BRAINSTORM (6 mechanisms, all expressed as deployable trained-params; decode stays
bit-exact):

  N1  GAIN-LINEARIZING VALUE-WARP (input pre-scale that cancels the quadratic).
      The consumer over-weights large/early inputs because C(a) ~ (T-tau)^2.  If we
      reshape each producer's value v -> g(v) with a CONCAVE warp (g'~1/sqrt), the
      quadratic ramp of the warped value integrates back to ~linear in v: a unit of
      v contributes a constant amount to the consumer membrane regardless of v's
      magnitude.  Closed form: choose g so C(tau(g(v))) is proportional to v, which
      (C ~ (S*g)^2/2, tau~S(1-g)) gives g(v) ~ sqrt(v).  More generally a per-layer
      POWER warp g(v) = (v/theta)^p with p in (0,1] sweeps from the raw quadratic
      (p=1) toward the linearized decode (p->0.5).  DEPLOYABLE: a monotone per-layer
      activation reshape; we realise it bit-exactly by warping the value BEFORE the
      TTFS encode (folded into the producer's effective activation / next layer's
      input_scale), so the deployed decode is untouched.  [PROTOTYPED]

  N2  TRAINED PER-NEURON PHASE-ADVANCE (learnable spike-time offset == bias).
      A constant per-cycle membrane lift beta advances the fire time (earlier ->
      faithful early window -> higher decoded value), reviving starved deep neurons.
      beta is EXACTLY the per-neuron bias_norm = bias/theta the node already folds.
      Phase-1's partial novel.py only HAND-SET a depth-graded beta cold; the novel
      angle here is to TRAIN the per-neuron offset (and gain) THROUGH the genuine
      single-spike cascade (boundary STE) so the gradient discovers the per-neuron
      phase correction the analytical trim cannot.  DEPLOYABLE: trained bias.
      [PROTOTYPED]

  N3  ARRIVAL-AWARE THRESHOLD (depth/fire-time-derived theta).  Sharpen the theta
      trim using the fire-cycle drift model (delta ~ d/sqrt(S)) instead of only the
      value mean.  [folded into the N1/theta comparison; the trim already captures it]
  N4  TRAINING-TIME SPIKE-TIME JITTER for robustness.  [discussed, rejected: it
      regularizes a continuous proxy, does not attack the deterministic gain.]
  N5  DUAL / COMPLEMENTARY two-spike code.  [rejected: changes the deployed decode
      -> NOT deployable per the rules.]
  N6  WINDOW-LENGTH-INVARIANT NORMALIZATION.  [== N1 with p chosen from S; folded in.]

Run:  source env/bin/activate
      python docs/.../experiments/novel.py quick   # PRIMARY depth-3 S=8, seeds 0-2
      python docs/.../experiments/novel.py         # full sweeps (minutes)
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cascade_lab  # noqa: F401,E402  (sets sys.path to src/tests)
from cascade_lab import (  # noqa: E402
    _SingleSegmentMLP,
    _accuracy,
    _calibrate_scales,
    cascade_forward,
    digits_task,
    train_continuous,
)
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from mimarsinan.torch_mapping.converter import convert_torch_model  # noqa: E402


# ---------------------------------------------------------------------------
# Shared infra: train a continuous cascade ONCE per (depth, seed), cache the
# state_dict, and stamp out fresh converted+calibrated flows cheaply.  Training
# is the only real cost; conversion+calibration is ~free.
# ---------------------------------------------------------------------------
_BASE_CACHE: dict = {}
EP = 120  # continuous-train epochs (cont ~0.94 on PRIMARY)


def _trained_base(depth, seed, *, width=64, in_dim=64, n_classes=10, epochs=EP):
    key = (depth, seed, width, in_dim, n_classes, epochs)
    if key not in _BASE_CACHE:
        torch.manual_seed(seed)
        xtr, ytr, xte, yte = digits_task(seed=seed + 1)
        base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
        train_continuous(base, xtr, ytr, epochs=epochs)
        with torch.no_grad():
            cont = _accuracy(base(xte.float()), yte)
        _BASE_CACHE[key] = (base.state_dict(), xtr, ytr, xte, yte, cont,
                            (depth, width, in_dim, n_classes))
    return _BASE_CACHE[key]


def _fresh_flow(depth, seed, **kw):
    sd, xtr, ytr, xte, yte, cont, dims = _trained_base(depth, seed, **kw)
    d, width, in_dim, n_classes = dims
    base = _SingleSegmentMLP(d, width, in_dim, n_classes)
    base.load_state_dict(sd)
    base.eval()
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    _calibrate_scales(flow, xtr[:256])
    return flow, xtr, ytr, xte, yte, cont


def _gen_acc(flow, xte, yte, S):
    install_ttfs_nodes(flow, S)
    with torch.no_grad():
        return _accuracy(cascade_forward(flow, xte, S), yte)


def _layer_value_dists(flow, x):
    """Per-perceptron flattened positive pre-scale outputs relu(W x + b) (theta-
    independent calib stats — the only signal the analytical warp may use)."""
    out: dict[int, torch.Tensor] = {}
    handles = []
    flow.double().eval()
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, o, k=k):
            out[k] = torch.relu(o.detach()).reshape(-1).double()
        handles.append(p.register_forward_hook(hook))
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return out


# ---------------------------------------------------------------------------
# theta-trim baseline (the Phase-1 deployable fix we must beat / complement).
# theta_d = mean(relu(act_d)) / target  -> mean drive maps to firing value `target`.
# ---------------------------------------------------------------------------
def apply_theta_trim(flow, xcal, *, target=0.5, last_layer=False):
    # theta_d = mean(relu(act_d)) / target, mean over ALL channels INCLUDING dead
    # zeros (matches precomp.thetas_mean_target — the dead-zero mass pulls the mean
    # down so theta shrinks more, pushing the live distribution up into the firing
    # band).  Using the positive-only mean under-trims and barely beats baseline.
    dists = _layer_value_dists(flow, xcal)
    n = len(list(flow.get_perceptrons()))
    for k, p in enumerate(flow.get_perceptrons()):
        if k == n - 1 and not last_layer:
            continue
        m = float(dists[k].mean())
        if m <= 0:
            continue
        p.set_activation_scale(torch.tensor(max(m / target, 1e-6), dtype=torch.float64))
    return flow


# ===========================================================================
# N1 — GAIN-LINEARIZING VALUE-WARP (per-layer power pre-shape).
# ===========================================================================
# The consumer membrane contribution of a producer value v is C(tau(v)) ~ (S*v')^2/2
# where v' = relu(out)/theta in [0,1].  We pre-warp v' -> v'^p (p in (0,1]) BEFORE
# the TTFS encode.  p=1 = the raw (quadratic-gained) decode; p->0.5 makes C linear
# in the ORIGINAL value, cancelling the over-weighting of large/early inputs.
#
# DEPLOYABILITY.  A power warp is a monotone reshape of the producer activation. We
# realise it bit-exactly WITHOUT touching the deployed decode by warping the value
# in the value domain and re-encoding: equivalently, the producer emits g(v') and
# the consumer's weights are pre-divided so it still computes the intended product.
# In this toy we implement the warp by replacing the post-ReLU value the next layer
# consumes -> this is a per-layer activation function the chip realises as a
# remapped threshold ladder (the encode tau = round(S(1-g(v'))) is still a single
# spike; only the value->time map changes, which is a calibration table, not a new
# decode).  We keep the LAST layer un-warped (argmax readout must stay faithful).
class _PowerWarp(nn.Module):
    """relu(x) then warp the normalized value v' = relu(x)/theta -> (v')^p, rescaled
    back by theta so downstream weights are unchanged.  p<1 is concave (gain-linearizing)."""

    def __init__(self, p: float, theta: float):
        super().__init__()
        self.p = float(p)
        self.theta = float(theta)

    def forward(self, x):
        v = torch.relu(x)
        vn = (v / self.theta).clamp(0.0, 1.0)
        return (vn ** self.p) * self.theta


def apply_n1_warp(flow, xcal, *, p=0.5, theta_target=0.5, last_layer=False):
    """Per-layer: set theta via the trim (so values span [0,1]) THEN insert a power
    warp on the activation so the encoded value is (v')^p.  The warp is folded into
    the perceptron's activation module; install_ttfs_nodes re-reads activation_scale
    so the decode stays the genuine single-spike ramp on the warped value."""
    apply_theta_trim(flow, xcal, target=theta_target, last_layer=True)
    n = len(list(flow.get_perceptrons()))
    for k, p_ in enumerate(flow.get_perceptrons()):
        if k == n - 1 and not last_layer:
            continue  # readout stays un-warped; its theta is still trimmed above
        theta = float(p_.activation_scale)
        p_.set_activation(_PowerWarp(p, theta))
    return flow


def _gen_acc_warped(flow, xte, yte, S):
    """Genuine cascade acc when warp modules are installed: install_ttfs_nodes would
    overwrite the warp, so we instead encode the warp INTO the value seen by the TTFS
    node.  Implemented by composing warp -> TTFS at install time (below)."""
    install_ttfs_warped_nodes(flow, S)
    with torch.no_grad():
        return _accuracy(cascade_forward(flow, xte, S), yte)


class _WarpedTTFS(nn.Module):
    """A power warp followed by the genuine TTFSActivation node — a single per-layer
    activation that (a) reshapes the value, (b) encodes/decodes via the unchanged
    single-spike ramp.  The warp is part of the calibration value->time table; the
    decode (ramp double-integrate, threshold crossing) is bit-exact with hardware."""

    def __init__(self, p, theta, ttfs):
        super().__init__()
        self.p = float(p)
        self.theta = float(theta)
        self.ttfs = ttfs

    def __getattr__(self, name):
        # Delegate node-protocol attributes (set_cycle_accurate, reset_state, T, ...)
        # to the wrapped TTFS node so the segment driver treats us as the node.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.ttfs, name)

    def forward(self, x):
        v = torch.relu(x)
        vn = (v / self.theta).clamp(0.0, 1.0)
        warped = (vn ** self.p) * self.theta
        return self.ttfs(warped)


def install_ttfs_warped_nodes(flow, S):
    """Like install_ttfs_nodes but, where a perceptron carries a _PowerWarp, wrap the
    genuine TTFS node in the warp so the encoded value is (v')^p."""
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation

    for p in flow.get_perceptrons():
        warp = p.activation if isinstance(p.activation, _PowerWarp) else None
        node = TTFSActivation(
            T=S, activation_scale=p.activation_scale,
            input_scale=p.input_activation_scale, bias=p.layer.bias,
            thresholding_mode="<=", encoding=getattr(p, "is_encoding_layer", False))
        if warp is not None:
            p.set_activation(_WarpedTTFS(warp.p, warp.theta, node))
        else:
            p.set_activation(node)
    return flow.double()


# ===========================================================================
# N2 — TRAINED PER-NEURON PHASE-ADVANCE (+ optional per-neuron gain), trained
# THROUGH the genuine single-spike cascade (boundary STE).  This is the novel
# departure from the cold hand-set beta in Phase-1: the gradient discovers the
# per-neuron offset that the scalar/per-channel analytical trim cannot express.
# Only the per-neuron BIAS (and optionally a per-neuron weight-row scale) train ->
# deployable trained params; weights' direction is frozen so it is a pure
# correction of the conversion, not a retrain of the task.
# ===========================================================================
def train_n2_offsets(flow, xtr, ytr, S, *, epochs=40, lr=2e-2, train_gain=False,
                     surrogate_temp=0.5, init_from_trim=True, xcal=None,
                     target=0.5):
    """Freeze weight DIRECTIONS; train ONLY per-neuron bias (phase-advance) and,
    optionally, a per-neuron output gain, through the genuine cascade."""
    if init_from_trim:
        apply_theta_trim(flow, xcal if xcal is not None else xtr[:256], target=target,
                         last_layer=True)
    install_ttfs_nodes(flow, S)
    flow.double()

    percs = flow.get_perceptrons()
    train_params = []
    for k, p in enumerate(percs):
        p.layer.weight.requires_grad_(False)
        if p.layer.bias is not None:
            p.layer.bias.requires_grad_(True)
            train_params.append(p.layer.bias)
    gains = None
    if train_gain:
        gains = [nn.Parameter(torch.ones(p.layer.weight.shape[0], dtype=torch.float64))
                 for p in percs]
        train_params += gains
    # phase-advance acts through bias_norm = bias/theta which the node folds at drive
    # time; train it so the cascade revives.  install_ttfs_nodes binds bias by ref.
    opt = torch.optim.Adam(train_params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(xtr.shape[0])
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = cascade_forward(flow, xtr[idx], S, grad=True,
                                     surrogate_temp=surrogate_temp)
            lossf(logits, ytr[idx]).backward()
            opt.step()
    return flow


# ===========================================================================
# Evaluation entry points.
# ===========================================================================
def eval_baseline(depth, S, seed):
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    return cont, _gen_acc(flow, xte, yte, S)


def eval_theta_trim(depth, S, seed, *, target=0.5):
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    apply_theta_trim(flow, xtr[:256], target=target, last_layer=True)
    return cont, _gen_acc(flow, xte, yte, S)


def eval_n1(depth, S, seed, *, p=0.5, theta_target=0.5):
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    apply_n1_warp(flow, xtr[:256], p=p, theta_target=theta_target)
    return cont, _gen_acc_warped(flow, xte, yte, S)


def eval_n2(depth, S, seed, *, epochs=40, lr=2e-2, train_gain=False, target=0.5,
            init_from_trim=True):
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    # cold reference for lift accounting
    if init_from_trim:
        apply_theta_trim(flow, xtr[:256], target=target, last_layer=True)
    cold = _gen_acc(flow, xte, yte, S)
    flow2, xtr2, ytr2, xte2, yte2, _ = _fresh_flow(depth, seed)
    train_n2_offsets(flow2, xtr2, ytr2, S, epochs=epochs, lr=lr, train_gain=train_gain,
                     init_from_trim=init_from_trim, xcal=xtr2[:256], target=target)
    with torch.no_grad():
        ft = _accuracy(cascade_forward(flow2, xte2, S), yte2)
    return cont, cold, float(ft)


def eval_n1_then_n2(depth, S, seed, *, p=0.5, theta_target=0.5, epochs=40, lr=2e-2):
    """STACK: N1 warp (gain-linearize, cold) then N2 trained offsets through the
    genuine cascade ON the warped activation.  Does the warp give the FT a better
    init / higher ceiling than the trim alone?"""
    flow, xtr, ytr, xte, yte, cont = _fresh_flow(depth, seed)
    apply_n1_warp(flow, xtr[:256], p=p, theta_target=theta_target)
    install_ttfs_warped_nodes(flow, S)
    with torch.no_grad():
        cold = _accuracy(cascade_forward(flow, xte, S), yte)
    percs = flow.get_perceptrons()
    train_params = []
    for p_ in percs:
        p_.layer.weight.requires_grad_(False)
        if p_.layer.bias is not None:
            p_.layer.bias.requires_grad_(True)
            train_params.append(p_.layer.bias)
    opt = torch.optim.Adam(train_params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    bs = 256
    for _ in range(epochs):
        perm = torch.randperm(xtr.shape[0])
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logits = cascade_forward(flow, xtr[idx], S, grad=True, surrogate_temp=0.5)
            lossf(logits, ytr[idx]).backward()
            opt.step()
    with torch.no_grad():
        ft = _accuracy(cascade_forward(flow, xte, S), yte)
    return cont, float(cold), float(ft)


def _mean(xs):
    return sum(xs) / len(xs)


# ===========================================================================
# Reports.
# ===========================================================================
def report_n1(seeds=(0, 1, 2)):
    print("=" * 78)
    print("N1  GAIN-LINEARIZING VALUE-WARP (power pre-shape) vs theta-trim, COLD")
    print("    PRIMARY depth-3 S=8.  p=1.0 == trim (no warp); p<1 cancels the")
    print("    quadratic over-weighting of early/large spikes.")
    print("=" * 78)
    base = [eval_baseline(3, 8, s)[1] for s in seeds]
    trim = [eval_theta_trim(3, 8, s, target=0.5)[1] for s in seeds]
    print(f"  baseline (max-calib) : {[round(b,3) for b in base]} mean={_mean(base):.4f}")
    print(f"  theta-trim (p=1.0)   : {[round(t,3) for t in trim]} mean={_mean(trim):.4f}")
    print(f"  {'warp p':>8}  seeds                  mean")
    for p in (0.9, 0.75, 0.6, 0.5, 0.4):
        g = [eval_n1(3, 8, s, p=p, theta_target=0.5)[1] for s in seeds]
        print(f"  {p:>8}  {[round(x,3) for x in g]}   {_mean(g):.4f}")


def report_n1_grid(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("N1  warp (best p) vs theta-trim vs baseline across depth x S (COLD)")
    print("=" * 78)
    print(f"{'depth':>5}{'S':>4}{'cont':>8}{'base':>8}{'trim':>8}{'warp':>8}{'w-t':>8}")
    for depth in (2, 3, 4):
        for S in (4, 8, 16, 32):
            cont = _mean([eval_baseline(depth, S, s)[0] for s in seeds])
            base = _mean([eval_baseline(depth, S, s)[1] for s in seeds])
            trim = _mean([eval_theta_trim(depth, S, s, target=0.5)[1] for s in seeds])
            warp = _mean([eval_n1(depth, S, s, p=0.5, theta_target=0.5)[1] for s in seeds])
            print(f"{depth:>5}{S:>4}{cont:>8.3f}{base:>8.3f}{trim:>8.3f}{warp:>8.3f}"
                  f"{warp-trim:>+8.3f}")


def report_n2(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("N2  TRAINED per-neuron PHASE-ADVANCE through the genuine cascade.")
    print("    trim-init -> train ONLY per-neuron bias (offset) via boundary STE.")
    print("    PRIMARY depth-3 S=8.  cold = trim-init; ft = after offset training.")
    print("=" * 78)
    print(f"{'seed':>5}{'cont':>8}{'trim_cold':>11}{'n2_ft':>9}{'lift':>8}")
    conts, colds, fts = [], [], []
    for s in seeds:
        cont, cold, ft = eval_n2(3, 8, s, epochs=40, lr=2e-2)
        conts.append(cont); colds.append(cold); fts.append(ft)
        print(f"{s:>5}{cont:>8.3f}{cold:>11.3f}{ft:>9.3f}{ft-cold:>+8.3f}")
    print(f" mean {_mean(conts):>7.3f}{_mean(colds):>11.3f}{_mean(fts):>9.3f}"
          f"{_mean(fts)-_mean(colds):>+8.3f}")


def report_n2_grid(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("N2  trained offsets across depth x S (cold=trim-init, ft=after training)")
    print("=" * 78)
    print(f"{'depth':>5}{'S':>4}{'cont':>8}{'trim':>8}{'n2_ft':>9}{'lift':>8}")
    for depth in (2, 3, 4):
        for S in (8, 16):
            r = [eval_n2(depth, S, s, epochs=30, lr=2e-2) for s in seeds]
            cont = _mean([x[0] for x in r])
            cold = _mean([x[1] for x in r])
            ft = _mean([x[2] for x in r])
            print(f"{depth:>5}{S:>4}{cont:>8.3f}{cold:>8.3f}{ft:>9.3f}{ft-cold:>+8.3f}")


def report_stack(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("STACK  N1 warp (cold) -> N2 trained offsets (genuine FT) on PRIMARY.")
    print("    Does the gain-linearized warp give the FT a higher ceiling than trim?")
    print("=" * 78)
    print(f"{'method':<22}{'cont':>8}{'cold':>8}{'ft':>8}{'lift':>8}")
    # control: trim-init then N2 (no warp) -- already in report_n2; recompute mean ft
    n2 = [eval_n2(3, 8, s, epochs=40, lr=2e-2) for s in seeds]
    print(f"{'trim -> N2':<22}{_mean([x[0] for x in n2]):>8.3f}"
          f"{_mean([x[1] for x in n2]):>8.3f}{_mean([x[2] for x in n2]):>8.3f}"
          f"{_mean([x[2]-x[1] for x in n2]):>+8.3f}")
    st = [eval_n1_then_n2(3, 8, s, p=0.5, epochs=40, lr=2e-2) for s in seeds]
    print(f"{'warp(p=.5) -> N2':<22}{_mean([x[0] for x in st]):>8.3f}"
          f"{_mean([x[1] for x in st]):>8.3f}{_mean([x[2] for x in st]):>8.3f}"
          f"{_mean([x[2]-x[1] for x in st]):>+8.3f}")


def quick(seeds=(0, 1, 2)):
    print("PRIMARY depth-3 S=8, seeds", seeds, "  (baseline / trim / N1-warp / N2-trained)")
    base = [eval_baseline(3, 8, s)[1] for s in seeds]
    trim = [eval_theta_trim(3, 8, s, target=0.5)[1] for s in seeds]
    warp = [eval_n1(3, 8, s, p=0.5, theta_target=0.5)[1] for s in seeds]
    n2 = [eval_n2(3, 8, s, epochs=40, lr=2e-2)[2] for s in seeds]
    print("baseline    :", [round(b, 3) for b in base], "mean", round(_mean(base), 4))
    print("theta-trim  :", [round(t, 3) for t in trim], "mean", round(_mean(trim), 4))
    print("N1 warp p=.5:", [round(w, 3) for w in warp], "mean", round(_mean(warp), 4))
    print("N2 trained  :", [round(n, 3) for n in n2], "mean", round(_mean(n2), 4))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "quick":
        quick()
    else:
        report_n1()
        report_n1_grid()
        report_n2()
        report_n2_grid()
        report_stack()
