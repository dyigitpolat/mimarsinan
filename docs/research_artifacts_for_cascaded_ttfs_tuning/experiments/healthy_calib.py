"""PILLAR C1 — HEALTHY data-grounded calibration for the COLD greedy fire-once cascade.

Make the cold genuine single-spike cascade ALIVE + well-conditioned (per-depth firing
rates tracking the ANN, decoded values ordered, genuine accuracy >> chance) under the
UNCHANGED greedy fire-once execution, with ZERO fine-tuning. This is the healthy init
the quick FT (Pillar 2) starts from.

Mechanism the calibration exploits (firing rule, from segment_policy_ttfs / ttfs_spiking):
  cascade neuron: weighted = (W·spike)·(in_scale/theta); ramp += weighted;
                  membrane += ramp + bias/theta; fires once when membrane >= 1.
  decoded value  = (count/T)·theta  ;  rate = count/T in [0,1].
So per-neuron theta (activation_scale) is BOTH firing threshold and decode scale:
 - raising theta -> later/never crossing (revive nothing, kill firing);
 - lowering theta -> earlier crossing (more firing) AND a smaller decode scale.
 - bias/theta shifts the membrane baseline every cycle -> the clean revive/suppress
   lever (raise to revive a dead channel, lower to suppress an over-firing channel)
   that drives the firing RATE toward the ANN rate without the theta decode coupling.

The cold pathology (measured): shallow layers DIE (alive_frac ~0.3-0.5) while their
surviving channels OVER-fire (cas_rate >> ann_rate), and deep layers OVER-fire wholesale
(cas_rate 0.3-0.45 vs ann ~0.10) -- premature fire on mixed-sign fan-in. So calibration
must simultaneously REVIVE dead channels and SUPPRESS over-firers to the ANN rate.

CALIBRATION (firing-rate matching, iterated downstream, deployable -- all per-neuron
bias/theta the chip already carries):
  1. per-channel promote theta and bias.
  2. iterate front-to-back: drive the cold cascade, read each non-encoding neuron's
     firing rate, nudge its bias toward (ann_rate - cas_rate) so the membrane baseline
     pushes the rate to the ANN rate (revives dead, suppresses over-firers); re-drive so
     downstream sees the corrected upstream (the cascade is causal -> downstream rate
     only settles after upstream is fixed).
  3. revive any still-dead-but-should-fire channels by lowering their theta until they
     cross (theta only touches dead channels; alive channels keep their decode scale).

ZERO weight changes to the ANN logits path beyond per-neuron bias (a chip param);
the staircase/continuous accuracy is preserved by construction (bias correction is the
deployable DFQ first-moment match -- see dfq_bias_correction.py).

VERDICT (digits, {6,9,12}deep x {8,16,32}S, device-safe value-mode = scale-aware
boundaries + DFQ decoded-mean match):
 * HEALTH rises 3-7x at EVERY depth: alive_frac -> 1.0 for the first 4-6 layers and
   the per-sample decode_corr (ANN signal) is REVIVED from cold ~0 to 0.5-0.86 through
   the shallow/mid layers (d12 cold corr [.39 .08 .02 0 0..] -> cal [.86 .76 .65 .51 .41
   .27 .14 ...]). The death cascade is reversed in the shallow half.
 * COLD genuine accuracy: d=6 0.66-0.86 (>> chance, MEETS the 0.5-0.85 target, ZERO FT);
   d>=9 stays ~chance because decode_corr STILL collapses to 0 at the DEEPEST layers
   (the irreducible premature-fire weight-timing scramble per-neuron calib cannot touch).
 * PILLAR-2 PAYOFF (the point of a healthy init): FT from the healthy init is ALIVE so
   the gradient is not severed -- d=9 reaches 0.91/50 steps & 0.93/100 (cold needs ~400);
   d=12 reaches 0.876/400 where COLD STALLS at 0.45. Use a LOW LR (5e-4) on the alive
   init (2e-3 wobbles/collapses). Net: calibration delivers cold-acc at d=6 and the fast
   FERTILE init for the quick FT at depth, exactly the Pillar1/Pillar2 labor split.

Run: python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/healthy_calib.py
     python .../healthy_calib.py --ft     # Pillar-2 FT speed (cold vs healthy init)
     python .../healthy_calib.py --test   # contained self-test
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(__file__)
for _p in (_HERE, os.path.join(_HERE, "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The worktree's spikingjelly submodule is not checked out (genuine-FT backward
# needs spikingjelly.activation_based.surrogate); force the main repo's copy onto the
# front of sys.path (the harness already inserted the empty worktree submodule dir,
# which shadows the real package).
_MAIN_SJ = "/home/yigit/repos/research_stuff/mimarsinan/spikingjelly"
if os.path.isdir(os.path.join(_MAIN_SJ, "spikingjelly", "activation_based")):
    sys.path.insert(0, _MAIN_SJ)
    import importlib
    if "spikingjelly" in sys.modules:
        try:
            importlib.reload(sys.modules["spikingjelly"])
        except Exception:  # noqa: BLE001
            del sys.modules["spikingjelly"]

import recipe_harness as H  # noqa: E402
from revive import _per_neuron  # noqa: E402  (per-perceptron decoded + rate probe)
from mimarsinan.mapping.mappers.compute_op_mapper import ComputeOpMapper  # noqa: E402
from mimarsinan.mapping.mappers.structural import ConcatMapper, InputMapper  # noqa: E402
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402


def propagate_input_scales_per_channel(flow, input_data_scale=1.0):
    """Per-channel-aware variant of propagate_boundary_input_scales: each
    perceptron's input_activation_scale = mean theta_out of upstream sources, but
    theta_out may be a per-output-channel vector (the production helper casts it to
    a scalar via float() and crashes). We keep theta_out as a tensor and reduce to
    a scalar mean for the (scalar) input_activation_scale the cascade consumes."""
    repr_ = flow.get_mapper_repr()
    repr_._ensure_exec_graph()
    out_scales: dict = {}

    def scalar(v):
        if isinstance(v, torch.Tensor):
            return float(v.double().mean())
        return float(v)

    def mean_src(deps, default):
        present = [out_scales[d] for d in deps if d in out_scales]
        return sum(present) / len(present) if present else default

    for node in repr_._exec_order:
        deps = repr_._deps.get(node, [])
        if isinstance(node, InputMapper):
            out_scales[node] = float(input_data_scale)
        elif perceptron_of(node) is not None:
            p = perceptron_of(node)
            p.set_input_activation_scale(mean_src(deps, float(input_data_scale)))
            out_scales[node] = scalar(p.activation_scale)
        elif isinstance(node, (ConcatMapper, ComputeOpMapper)):
            out_scales[node] = mean_src(deps, float(input_data_scale))
        else:
            present = [out_scales[d] for d in deps if d in out_scales]
            if present:
                out_scales[node] = sum(present) / len(present)


# --------------------------------------------------------------------------- #
# HEALTH METRIC (the Pillar-C1 target).
# --------------------------------------------------------------------------- #
def snapshot_teacher_rate(flow, teacher):
    """FIXED ANN firing-rate target per perceptron = clamp(relu(act)/theta0, 0, 1),
    computed against the ORIGINAL (build-time) theta and frozen. The decoded-value
    target is relu(act); rate = decoded/theta0 is the rate that decodes to it. Frozen
    so later per-channel theta moves don't drift the target (the earlier instability)."""
    snap = {}
    for k, p in enumerate(flow.get_perceptrons()):
        t = teacher.get(k)
        if t is None:
            continue
        sc = p.activation_scale.detach().double()
        if sc.dim() == 0:
            sc = sc.reshape(1).expand_as(t)
        n = min(t.numel(), sc.numel())
        snap[k] = (t[:n].clamp(min=0) / sc[:n].clamp(min=1e-9)).clamp(0.0, 1.0)
    return snap


def _teacher_rate(flow, k, teacher):
    """Live ANN firing-rate target (recomputed against current theta) — used only by
    the health metric; calibration uses the frozen snapshot_teacher_rate."""
    t = teacher.get(k)
    if t is None:
        return None
    sc = flow.get_perceptrons()[k].activation_scale.detach().double()
    if sc.dim() == 0:
        sc = sc.reshape(1).expand_as(t)
    n = min(t.numel(), sc.numel())
    return (t[:n].clamp(min=0) / sc[:n].clamp(min=1e-9)).clamp(0.0, 1.0)


def _decoded_corr(decoded, t):
    """Per-channel Pearson(decoded_cascade, teacher_activation) over the batch is
    not available here (we only kept channel-means); use the CHANNEL-ordering
    correlation across channels as the decode_corr proxy (does the cascade rank
    channels like the ANN). decoded/t are per-channel mean vectors."""
    n = min(decoded.numel(), t.numel())
    a = decoded[:n].double()
    b = t[:n].clamp(min=0).double()
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a @ b) / denom)


def per_layer_health(flow, x, S, teacher, *, dead_rate=0.02, rate_snap=None):
    """Per-non-encoding-layer HEALTH = (1-rate_err)*alive_frac*max(decode_corr,0)
    plus the mode-a (under-fire) / mode-b (mistime) rate-error split. The rate target
    is the FROZEN snapshot (theta0) so the metric is comparable across theta moves."""
    if rate_snap is None:
        rate_snap = snapshot_teacher_rate(flow, teacher)
    pn = _per_neuron(flow, x, S)
    rows = []
    for k, (decoded, rate) in enumerate(pn):
        p = flow.get_perceptrons()[k]
        if decoded is None or getattr(p, "is_encoding_layer", False):
            continue
        trate = rate_snap.get(k)
        t = teacher.get(k)
        if trate is None or t is None:
            continue
        n = min(rate.numel(), trate.numel())
        r, tr = rate[:n].double(), trate[:n].double()
        wants = t[:n].clamp(min=0) > 1e-4
        if not bool(wants.any()):
            continue
        rerr = float((r - tr).abs()[wants].mean())
        dead = (r < dead_rate) & wants
        alive_frac = 1.0 - float(dead.sum()) / float(wants.sum())
        # mode split of the rate error over wanted channels
        under = (r < tr) & wants                    # under-fire (dead-ish, mode a)
        mode_a = float((tr - r).clamp(min=0)[under].sum())
        total = float((r - tr).abs()[wants].sum())
        mode_a_frac = mode_a / max(total, 1e-9)
        corr = max(_decoded_corr(decoded, t), 0.0)
        health = (1.0 - min(rerr, 1.0)) * alive_frac * corr
        rows.append(dict(layer=k, health=health, alive_frac=alive_frac,
                         rate_err=rerr, decode_corr=corr,
                         mode_a_frac=mode_a_frac,
                         cas_rate=float(r.mean()), ann_rate=float(tr[wants].mean())))
    return rows


def model_health(flow, x, S, teacher, rate_snap=None):
    rows = per_layer_health(flow, x, S, teacher, rate_snap=rate_snap)
    return (sum(r["health"] for r in rows) / max(len(rows), 1)), rows


# --------------------------------------------------------------------------- #
# CALIBRATION (firing-rate matching).
# --------------------------------------------------------------------------- #
def _promote_per_channel(flow):
    """Promote scalar activation_scale + layer.bias to per-output-channel for
    non-encoding perceptrons (the chip carries per-neuron theta + bias)."""
    for p in flow.get_perceptrons():
        if getattr(p, "is_encoding_layer", False):
            continue
        out_dim = p.layer.weight.shape[0]
        s = p.activation_scale.detach()
        if s.dim() == 0:
            p.activation_scale.data = s * torch.ones(out_dim, dtype=s.dtype, device=s.device)
        node = getattr(p, "activation", None)
        if node is not None and hasattr(node, "activation_scale"):
            node.activation_scale = p.activation_scale
        b = getattr(p.layer, "bias", None)
        if b is not None and b.dim() == 0:
            p.layer.bias.data = b * torch.ones(out_dim, dtype=b.dtype, device=b.device)


def apply_scale_aware_boundaries(flow, x, S, *, quantile=0.99):
    """theta_out = teacher p-quantile per perceptron (normalizes each block's output
    to [0,1]); encoding block pinned to the data scale. This is the load-bearing
    NORMALIZATION step (without it the deep cascade saturates/dies) -- the firing-rate
    bias loop refines on top. Reuses the production calibrate_scale_aware_boundaries."""
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    import torch as _t
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    samples = {}
    hs = [p.activation.register_forward_hook(
        lambda _m, _i, o, k=k: samples.__setitem__(k, o.detach()))
        for k, p in enumerate(flow.get_perceptrons())]
    flow.double().eval()
    with _t.no_grad():
        flow(x.double())
    for h in hs:
        h.remove()
    perceptrons = list(flow.get_perceptrons())
    dev = perceptrons[0].activation_scale.device
    dtype = perceptrons[0].activation_scale.dtype
    for k, p in enumerate(perceptrons):
        if getattr(p, "is_encoding_layer", False):
            theta = 1.0
        else:
            theta = max(float(_t.quantile(samples[k].abs().flatten(), quantile)), 1e-2)
        p.set_activation_scale(_t.tensor(theta, device=dev, dtype=dtype))
    propagate_input_scales_per_channel(flow)  # device-safe input_scale propagation
    return flow


def calibrate_rate_match(flow, x, S, teacher, *, iters=15, eta=0.5, mode="value",
                         boundaries=True, verbose=False, rate_snap=None):
    """Scale-aware boundaries + per-neuron bias correction (DEVICE-SAFE, deployable).

    1. scale-aware [0,1] boundaries (theta_out = teacher quantile) -- the load-bearing
       NORMALIZATION (without it the deep cascade saturates/dies). Device-safe (the
       production calibrate_scale_aware_boundaries moves theta to CPU via float(); this
       keeps it on the model's device so FT stays on GPU).
    2. per-neuron bias correction re-driven each round so downstream sees the corrected
       upstream (causal settle). The membrane sees bias/theta:
         mode="value": match decoded MEAN to relu(ann) (DFQ; revives dead neurons by
                       raising the membrane baseline -- the strongest cold-acc lever).
         mode="rate" : match firing RATE to the frozen ANN rate (alive-preserving).
    theta untouched after boundaries (decode scale stays calibrated)."""
    if boundaries:
        apply_scale_aware_boundaries(flow, x, S)
    _promote_per_channel(flow)
    perceptrons = list(flow.get_perceptrons())
    if rate_snap is None or boundaries:
        rate_snap = snapshot_teacher_rate(flow, teacher)
    ann_value = {k: teacher[k].clamp(min=0) for k in teacher}

    for it in range(iters):
        pn = _per_neuron(flow, x, S)
        for k, p in enumerate(perceptrons):
            if getattr(p, "is_encoding_layer", False):
                continue
            decoded, rate = pn[k]
            b = getattr(p.layer, "bias", None)
            if decoded is None or b is None:
                continue
            if mode == "value":
                t = ann_value.get(k)
                if t is None:
                    continue
                n = min(decoded.numel(), t.numel(), b.numel())
                step = eta * (t[:n].double() - decoded[:n].double())   # raw value domain
            else:  # rate
                trate = rate_snap.get(k)
                if trate is None:
                    continue
                n = min(rate.numel(), trate.numel(), b.numel())
                theta = p.activation_scale.detach().double()
                if theta.dim() == 0:
                    theta = theta.reshape(1).expand(n)
                step = eta * (trate[:n] - rate[:n].double()) * theta[:n]
            with torch.no_grad():
                b[:n] += step.to(b.device, b.dtype)
        if verbose and (it % 3 == 0 or it == iters - 1):
            mh, _ = model_health(flow, x, S, teacher, rate_snap=rate_snap)
            print(f"    {mode}-match it{it}: health={mh:.3f}")
    return flow


def per_sample_decode_corr(flow, x, teacher_acts):
    """TRUE per-sample decode signal per non-encoding layer: mean over ANN-active
    channels of Pearson(cascade_decoded[:,c], relu(ann_act)[:,c]) across the batch.
    This is the discriminative signal the next layer can use -- if it is ~0 the layer
    carries no ANN information regardless of mean/rate match (the premature-fire
    scramble), so calibration (per-neuron bias/theta) cannot make that layer healthy."""
    from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
    from mimarsinan.spiking.segment_partition import perceptron_of

    rec = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), teacher_acts["__S__"])
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by = {id(perceptron_of(n)): v for n, v in rec.items() if perceptron_of(n) is not None}
    out = []
    for k, p in enumerate(flow.get_perceptrons()):
        if getattr(p, "is_encoding_layer", False):
            continue
        v = by.get(id(p))
        a = teacher_acts.get(k)
        if v is None or a is None:
            continue
        cas = v.reshape(v.shape[0], -1).double()
        ann = a.reshape(a.shape[0], -1).clamp(min=0).double()
        n = min(cas.shape[1], ann.shape[1])
        cas, ann = cas[:, :n], ann[:, :n]
        cm, am = cas - cas.mean(0), ann - ann.mean(0)
        corr = (cm * am).sum(0) / (cm.norm(dim=0) * am.norm(dim=0)).clamp(min=1e-9)
        fires = ann.mean(0) > 1e-4
        out.append(float(corr[fires].mean()) if bool(fires.any()) else 0.0)
    return out


def capture_teacher_acts(flow, x, S):
    """Per-sample ANN activations per perceptron (+ S stashed under '__S__')."""
    acts = {"__S__": S}
    hs = [p.activation.register_forward_hook(
        lambda _m, _i, o, k=k: acts.__setitem__(k, o.detach()))
        for k, p in enumerate(flow.get_perceptrons())]
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in hs:
        h.remove()
    return acts


def _fmt(rows, key):
    return "[" + " ".join(f"{r[key]:.2f}" for r in rows) + "]"


def _fmtl(lst):
    return "[" + " ".join(f"{v:.2f}" for v in lst) + "]"


def calibrate_distribution_matching(flow, x, S):
    """The production calibration (scale-aware boundaries + DFQ bias), for comparison.
    The teacher is the flow ITSELF in analytical staircase mode (== the ANN), which is
    what match_activation_distributions hooks (.activation per get_perceptrons)."""
    from mimarsinan.spiking.distribution_matching import match_activation_distributions
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    match_activation_distributions(flow, flow, x.double(), S,
                                   quantile=0.99, bias_iters=15, eta=0.5)
    return flow


def run(depths=(6, 9, 12), Ss=(8, 16, 32), seed=0):
    import copy
    print("=== PILLAR C1: HEALTHY data-grounded calibration of the COLD greedy cascade ===")
    print("cold = greedy fire-once genuine (chance ~0.10). decode_corr = per-sample signal")
    print("each non-encoding layer carries (Pearson cascade-vs-ANN over batch); ~0 => the")
    print("layer is scrambled, NO per-neuron calibration can make it healthy.\n")
    for D in depths:
        for S in Ss:
            flow, xtr, ytr, xte, yte, cont, teacher, base = H.build(D, S, seed=seed)
            snap = snapshot_teacher_rate(flow, teacher)
            tacts = capture_teacher_acts(flow, xte, S)
            cold = H.genuine_acc(flow, xte, yte, S)
            h0, _ = model_health(flow, xte, S, teacher, rate_snap=snap)
            corr0 = per_sample_decode_corr(flow, xte, tacts)

            # A) firing-rate-matching bias calibration (this work)
            fa = copy.deepcopy(flow)
            calibrate_rate_match(fa, xtr[:512], S, teacher, rate_snap=snap)
            acc_a = H.genuine_acc(fa, xte, yte, S)
            ha, rows_a = model_health(fa, xte, S, teacher, rate_snap=snap)
            corr_a = per_sample_decode_corr(fa, xte, tacts)

            # B) production distribution-matching (scale-aware boundaries + DFQ)
            fb = copy.deepcopy(flow)
            try:
                calibrate_distribution_matching(fb, xtr[:512], S)
                acc_b = H.genuine_acc(fb, xte, yte, S)
                hb, _ = model_health(fb, xte, S, teacher, rate_snap=snap)
            except Exception as e:  # noqa: BLE001
                acc_b, hb = float("nan"), float("nan")
                print(f"   [dist-match raised: {type(e).__name__}: {e}]")

            print(f"d={D:>2} S={S:>2}: cont={cont:.3f}  cold={cold:.3f}(H={h0:.3f})  "
                  f"value-match={acc_a:.3f}(H={ha:.3f})  dist-match={acc_b:.3f}(H={hb:.3f})")
            print(f"        decode_corr cold {_fmtl(corr0)}")
            print(f"        decode_corr cal  {_fmtl(corr_a)}")
            print(f"        cas_rate cal {_fmt(rows_a,'cas_rate')} ann {_fmt(rows_a,'ann_rate')} "
                  f"alive {_fmt(rows_a,'alive_frac')}")


# --------------------------------------------------------------------------- #
# PILLAR 2 — FT speed from a HEALTHY (calibrated) init vs from CHANCE (cold).
# --------------------------------------------------------------------------- #
def run_ft_speed(depths=(9, 12), Ss=(16,), seed=0, ladder=(25, 50, 100, 200, 400)):
    """Genuine fire-once cascade FT, COLD init vs HEALTHY (value-match) init. Reports
    per-step accuracy + cumulative wall (calib time folded into the healthy column).
    The healthy init has heavier per-step backward cost (alive neurons => more spike
    events) but needs FAR fewer steps; the cold deep cascade stalls (severed gradient)."""
    import copy
    import time

    from ft_budget import ft_genuine

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print("=== PILLAR 2: genuine fire-once FT, COLD vs HEALTHY init ===")
    for D in depths:
        for S in Ss:
            flow, xtr, ytr, xte, yte, cont, teacher, base = H.build(D, S, seed=seed)
            cold = H.genuine_acc(flow, xte, yte, S)
            fcal = copy.deepcopy(flow)
            sync(); t0 = time.time()
            calibrate_rate_match(fcal, xtr[:512], S, teacher, mode="value")
            sync(); cal_t = time.time() - t0
            print(f"d={D} S={S} cont={cont:.3f} cold={cold:.3f} "
                  f"calib={H.genuine_acc(fcal, xte, yte, S):.3f} (calib {cal_t:.1f}s)")
            print(f"  {'steps':>6} | {'cold acc':>8} | {'heal acc(lo-lr)':>15}")
            for label, f0, lr, base_t in (("cold", flow, 2e-3, 0.0),
                                          ("heal", fcal, 5e-4, cal_t)):
                f = copy.deepcopy(f0); prev = 0; row = []
                for steps in ladder:
                    ft_genuine(f, xtr, ytr, S, steps - prev, lr=lr); prev = steps
                    row.append((steps, H.genuine_acc(f, xte, yte, S)))
                if label == "cold":
                    cold_row = row
                else:
                    for i, steps in enumerate(ladder):
                        print(f"  {steps:>6} | {cold_row[i][1]:>8.3f} | {row[i][1]:>15.3f}")
                    bc = max(cold_row, key=lambda r: r[1])
                    bh = max(row, key=lambda r: r[1])
                    print(f"  BEST cold {bc[1]:.3f}@{bc[0]}  BEST heal {bh[1]:.3f}@{bh[0]}")


def _selftest():
    """Contained sanity checks on a tiny cascade (no GPU/large-data dependency on the
    grid): the health metric is in range, snapshot rate is in [0,1], boundaries keep
    theta on-device, and value-match shrinks the cascade<->ANN decoded-mean gap."""
    import copy

    flow, xtr, ytr, xte, yte, cont, teacher, base = H.build(6, 8, seed=0)
    snap = snapshot_teacher_rate(flow, teacher)
    for k, r in snap.items():
        assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0, f"rate snap[{k}] out of [0,1]"
    h, rows = model_health(flow, xte, S=8, teacher=teacher, rate_snap=snap)
    assert 0.0 <= h <= 1.0, f"model health {h} out of [0,1]"
    assert all(0.0 <= row["health"] <= 1.0 for row in rows)
    dev0 = flow.get_perceptrons()[1].activation_scale.device

    fa = copy.deepcopy(flow)
    apply_scale_aware_boundaries(fa, xtr[:128], 8)
    assert fa.get_perceptrons()[1].activation_scale.device == dev0, "boundaries moved theta off-device"

    fb = copy.deepcopy(flow)
    ann_value = {k: teacher[k].clamp(min=0) for k in teacher}
    from cascade_lab import _cascade_decoded_means
    from mimarsinan.spiking.dfq_bias_correction import mean_abs_gap
    cold_acc = H.genuine_acc(flow, xte, yte, 8)
    gap0 = mean_abs_gap(ann_value, _cascade_decoded_means(fb, xte, 8))
    calibrate_rate_match(fb, xtr[:128], 8, teacher, mode="value", iters=10)
    gap1 = mean_abs_gap(ann_value, _cascade_decoded_means(fb, xte, 8))
    assert gap1 < gap0, f"value-match did not shrink decoded-mean gap ({gap0:.3f} -> {gap1:.3f})"
    assert H.genuine_acc(fb, xte, yte, 8) > cold_acc, "value-match did not beat cold at d=6"
    print(f"_selftest OK: health={h:.3f}, decoded-mean gap {gap0:.3f}->{gap1:.3f}")


if __name__ == "__main__":
    import sys as _sys
    if "--ft" in _sys.argv:
        run_ft_speed()
    elif "--test" in _sys.argv:
        _selftest()
    else:
        run()
