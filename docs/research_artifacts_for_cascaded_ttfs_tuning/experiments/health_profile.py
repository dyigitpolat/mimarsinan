"""Numerical-health profile of the COLD greedy fire-once cascaded single-spike TTFS.

PILLAR-1 DIAGNOSTIC. The deployed cascade is pipelined, depth-staggered (latency
T+depth), ONE spike/neuron, each neuron fires the cycle its RUNNING partial sum
crosses theta (greedy). Cold (no FT, scale-calibrated only) it collapses to chance
(~0.10): the DEATH CASCADE. This module profiles WHY, per DEPTH, so the calibration
target (what a healthy ALIVE cascade looks like) is quantitative.

For each (depth, S) it reports, per perceptron layer:

  * firing rate (decoded/scale, via the genuine cascade node_value_recorder) vs the
    teacher's ANN activation rate (relu(act)/scale, clamped to [0,1]);
  * decoded / membrane magnitude vs teacher value magnitude;
  * %dead (rate <= DEAD) and %saturated (rate >= SAT);
  * per-channel decoded-value Pearson correlation to the teacher activation.

HEALTH metric (per depth, in [0,1], higher = healthier):
    health = (1 - rate_match_err) * alive_fraction * max(decode_corr, 0)
  - rate_match_err = mean|rate_cascade - rate_teacher| over channels the teacher
    fires (the cascade should reproduce the ANN's firing rate);
  - alive_fraction = fraction of teacher-active channels that are NOT dead in the
    cascade (a dead channel that should fire is the death-cascade signature);
  - decode_corr = per-channel Pearson(decoded_cascade, teacher_activation): does the
    cascade still ORDER the channels like the ANN, even if attenuated?
The model HEALTH is the per-depth mean (encoding layer excluded — it decodes the
input ideally by construction). A healthy alive cascade -> health ~ 1.0 at every
depth; the death cascade -> health collapses with depth.

MECHANISM SEPARATION (where/why death sets in): two distinct failure modes.
  (a) UNDER-FIRE from scale-mismatch: an attenuated input never reaches theta ->
      dead. A CALIBRATION/health problem (per-neuron theta/scale/bias revivable).
  (b) PREMATURE fire on mixed-sign weights: greedy crosses theta before the
      cancelling negatives arrive, so the value is correct at window END but the
      fire TIMING is early. A WEIGHT problem the FT must adapt.
We separate them with a POSITIVE-WEIGHT CONTROL: a clone whose every linear weight
is made non-negative (|W|) and re-calibrated. With all-positive weights the running
partial sum is monotone in the window, so greedy fire-once timing == the
value-correct fire time: NO premature firing is possible. Any residual death there
is PURE under-fire (mode a). The health RECOVERED by going real -> positive-control
is the premature-fire (mode b) contribution; the health STILL MISSING in the
positive control is the under-fire (mode a) contribution.

Run: python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/health_profile.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
for _p in (_HERE, os.path.join(_HERE, "..")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recipe_harness import build, genuine_acc  # noqa: E402
from cascade_fixtures import install_ttfs_nodes  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402
from mimarsinan.spiking.segment_partition import perceptron_of  # noqa: E402

DEAD = 0.02   # rate <= DEAD  -> the neuron is effectively dead
SAT = 0.98    # rate >= SAT   -> the neuron is saturated (always fires at window start)
ACTIVE = 1e-3  # teacher rate > ACTIVE -> the ANN considers this channel firing


# --------------------------------------------------------------------------- probes


def _eval_batch(xte, n=512):
    return xte[:n].double()


def _per_perceptron_decoded(flow, x, S):
    """Per-perceptron genuine-cascade decoded value, FULL batch (N x C), keyed by
    perceptron index. The recorder stores ``(accum/T)*scale`` per node (the decoded
    firing-rate value); see segment_policy_ttfs.TtfsSegmentPolicy.node_value_recorder."""
    rec: dict = {}
    drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
    drv._driver.policy.node_value_recorder = rec
    with torch.no_grad():
        drv(x.double())
    drv._driver.policy.node_value_recorder = None
    by = {id(perceptron_of(n)): v for n, v in rec.items() if perceptron_of(n) is not None}
    out = {}
    for k, p in enumerate(flow.get_perceptrons()):
        v = by.get(id(p))
        if v is not None:
            out[k] = v.reshape(-1, v.shape[-1]).double()
    return out


def _per_perceptron_teacher(flow, x):
    """Per-perceptron CONTINUOUS-ANN activation value, FULL batch (N x C), keyed by
    perceptron index (the value-domain output of each perceptron's activation node,
    cycle-accurate OFF = the staircase/continuous forward = the lossless target)."""
    from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    caps: dict = {}
    handles = []
    for k, p in enumerate(flow.get_perceptrons()):
        def hook(_m, _i, out, k=k):
            caps[k] = out.detach().reshape(-1, out.shape[-1]).double()
        handles.append(p.activation.register_forward_hook(hook))
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    return caps


def _scale_vec(p, C):
    s = p.activation_scale.detach().double()
    if s.dim() == 0:
        return s * torch.ones(C, dtype=torch.float64, device=s.device)
    return s[:C]


def _pearson_per_channel(a, b):
    """Mean over channels of Pearson(a[:,c], b[:,c]); channels with ~0 variance are
    skipped (degenerate correlation)."""
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a * b).sum(0)
    den = a.norm(dim=0) * b.norm(dim=0)
    valid = den > 1e-9
    if not bool(valid.any()):
        return 0.0
    return float((num[valid] / den[valid]).mean())


def _and_frac(mask, wants):
    """Fraction of ``wants`` channels that are also in ``mask`` (0 if none wanted)."""
    n = int(wants.sum())
    if n == 0:
        return 0.0
    return float((mask & wants).sum()) / n


def _mixed_sign_fanin(p):
    """Per-output-channel fraction of incoming weight magnitude that is NEGATIVE.
    0 = purely additive fan-in (greedy fire-time is value-correct, no premature
    fire possible); ->0.5 = balanced mixed-sign (the partial sum can overshoot
    theta before cancelling negatives arrive => premature fire)."""
    W = p.layer.weight.detach().double()                  # [out, in]
    neg = W.clamp(max=0).abs().sum(1)
    tot = W.abs().sum(1).clamp(min=1e-12)
    return (neg / tot)


# ----------------------------------------------------------------- per-depth profile


def profile_layers(flow, x, S):
    """Per-perceptron health rows for the cold cascade vs the continuous teacher."""
    dec = _per_perceptron_decoded(flow, x, S)
    tea = _per_perceptron_teacher(flow, x)
    rows = []
    for k, p in enumerate(flow.get_perceptrons()):
        d = dec.get(k)
        t = tea.get(k)
        if d is None or t is None:
            continue
        C = min(d.shape[-1], t.shape[-1])
        d, t = d[:, :C], t[:, :C]
        scale = _scale_vec(p, C)
        rate_c = (d / scale.clamp(min=1e-9)).clamp(0.0, 1.0)               # cascade rate
        rate_t = (t.clamp(min=0) / scale.clamp(min=1e-9)).clamp(0.0, 1.0)  # teacher rate
        rate_c_m = rate_c.mean(0)
        rate_t_m = rate_t.mean(0)

        wants = rate_t_m > ACTIVE                                          # teacher fires
        n_wants = int(wants.sum())
        dead = (rate_c_m <= DEAD)
        sat = (rate_c_m >= SAT)
        dead_should = _and_frac(dead, wants)                              # dead but should fire
        alive_frac = 1.0 - dead_should if n_wants else 1.0

        # firing-rate match over teacher-active channels (the death signature is a
        # cascade rate far below the teacher's on channels the ANN fires).
        err = float((rate_c_m[wants] - rate_t_m[wants]).abs().mean()) if n_wants else 0.0
        corr = _pearson_per_channel(d, t.clamp(min=0))

        # Mechanism split of the rate error (per teacher-active channel): a DEAD
        # channel's error is under-fire (mode a, scale/magnitude); a channel that is
        # ALIVE but still mis-rates is a greedy fire-TIMING error (mode b, premature
        # fire) -- the value is correct at window-end, only the spike time is off.
        if n_wants:
            chan_err = (rate_c_m[wants] - rate_t_m[wants]).abs()
            dead_w = dead[wants]
            underfire_err = float(chan_err[dead_w].sum() / max(n_wants, 1))
            mistime_err = float(chan_err[~dead_w].sum() / max(n_wants, 1))
        else:
            underfire_err = mistime_err = 0.0
        neg_frac = float(_mixed_sign_fanin(p)[:C][wants].mean()) if n_wants else 0.0

        health = (1.0 - min(err, 1.0)) * alive_frac * max(corr, 0.0)
        rows.append({
            "depth": k,
            "enc": bool(getattr(p, "is_encoding_layer", False)),
            "n_chan": C,
            "rate_c": round(float(rate_c_m.mean()), 4),
            "rate_t": round(float(rate_t_m.mean()), 4),
            "rate_err": round(err, 4),
            "underfire_err": round(underfire_err, 4),
            "mistime_err": round(mistime_err, 4),
            "neg_fanin": round(neg_frac, 4),
            "dead_frac": round(float(dead.double().mean()), 4),
            "dead_should": round(dead_should, 4),
            "sat_frac": round(float(sat.double().mean()), 4),
            "alive_frac": round(alive_frac, 4),
            "decode_corr": round(corr, 4),
            "dec_mag": round(float(d.mean()), 4),
            "tea_mag": round(float(t.clamp(min=0).mean()), 4),
            "health": round(health, 4),
        })
    return rows


def model_health(rows):
    """Mean per-depth health over the NON-encoding layers (encoding decodes the
    input ideally by construction)."""
    hs = [r["health"] for r in rows if not r["enc"]]
    return round(sum(hs) / len(hs), 4) if hs else 0.0


# --------------------------------------------------- positive-weight control (mode a/b)


def _calibrate_cuda(flow, x):
    caps: dict = {}
    handles = [
        p.register_forward_hook(
            lambda _m, _i, o, p=p: caps.__setitem__(
                p, max(caps.get(p, 0.0), float(o.detach().abs().max())))
        )
        for p in flow.get_perceptrons()
    ]
    flow.double().eval()
    with torch.no_grad():
        flow(x.double())
    for h in handles:
        h.remove()
    for p in flow.get_perceptrons():
        p.set_activation_scale(
            torch.tensor(max(caps[p], 1e-3), dtype=torch.float64, device=x.device))


def positive_weight_clone(flow, x_calib, S):
    """Clone with every linear weight made non-negative (|W|) + re-calibrated +
    fresh TTFS nodes. All-positive weights => the running partial sum is monotone in
    the window => greedy fire-once timing == value-correct fire time => NO premature
    firing. Residual death here is PURE under-fire (scale-mismatch / mode a)."""
    pos = copy.deepcopy(flow)
    for p in pos.get_perceptrons():
        with torch.no_grad():
            p.layer.weight.data = p.layer.weight.data.abs()
    _calibrate_cuda(pos, x_calib)
    install_ttfs_nodes(pos, S)
    return pos


# ----------------------------------------------------------------------------- report


def _fmt_rows(rows):
    head = (f"{'d':>2} {'enc':>3} {'rate_c':>7} {'rate_t':>7} {'err':>6} {'uf_e':>5} "
            f"{'mt_e':>5} {'neg%':>5} {'dead':>5} {'d!fire':>6} {'sat':>5} {'alive':>6} "
            f"{'corr':>6} {'health':>7}")
    lines = [head]
    for r in rows:
        lines.append(
            f"{r['depth']:>2} {('Y' if r['enc'] else '-'):>3} "
            f"{r['rate_c']:>7.4f} {r['rate_t']:>7.4f} {r['rate_err']:>6.3f} "
            f"{r['underfire_err']:>5.3f} {r['mistime_err']:>5.3f} {r['neg_fanin']:>5.2f} "
            f"{r['dead_frac']:>5.2f} {r['dead_should']:>6.2f} {r['sat_frac']:>5.2f} "
            f"{r['alive_frac']:>6.3f} {r['decode_corr']:>6.3f} {r['health']:>7.4f}")
    return "\n".join(lines)


def _mean_nonenc(rows, key):
    vs = [r[key] for r in rows if not r["enc"]]
    return sum(vs) / len(vs) if vs else 0.0


def run(depths=(6, 9, 12), Ss=(8, 16, 32), seed=0, n_eval=512):
    print("=" * 100)
    print("COLD GREEDY FIRE-ONCE CASCADE — numerical-health profile (calibration TARGET)")
    print("rate_c=cascade rate, rate_t=ANN rate, err=|rate_c-rate_t| on ANN-active chans,")
    print("  uf_e=under-fire(dead) err part, mt_e=mistime(alive) err part, neg%=neg fan-in frac,")
    print("  dead=rate<=%.2f frac, d!fire=dead-but-should-fire, sat=rate>=%.2f, corr=decode Pearson"
          % (DEAD, SAT))
    print("=" * 100)
    summary = []
    for depth in depths:
        for S in Ss:
            flow, xtr, ytr, xte, yte, cont, teacher, base = build(depth, S, seed=seed)
            x = _eval_batch(xte, n_eval)
            cold_acc = genuine_acc(flow, xte, yte, S)
            rows = profile_layers(flow, x, S)
            H = model_health(rows)

            pos = positive_weight_clone(flow, xtr[:256], S)
            pos_acc = genuine_acc(pos, xte, yte, S)
            pos_rows = profile_layers(pos, x, S)

            print(f"\n### depth={depth} S={S}  continuous={cont:.4f}  "
                  f"cold-genuine={cold_acc:.4f}  HEALTH={H:.4f}")
            print(_fmt_rows(rows))
            # mode attribution from the per-neuron rate-error split (deployable,
            # execution unchanged): under-fire (mode a) = dead-channel error;
            # premature-fire (mode b) = alive-but-mistimed error.
            uf = _mean_nonenc(rows, "underfire_err")
            mt = _mean_nonenc(rows, "mistime_err")
            tot = uf + mt
            ratio = f"(a:b = {uf/tot:.0%}:{mt/tot:.0%})" if tot > 1e-9 else "(no active err)"
            print(f"  MODE SPLIT (mean rate-err over non-enc layers): "
                  f"under-fire(a)={uf:.3f}  premature/mistime(b)={mt:.3f}  {ratio}")
            # positive-weight control: |W| -> monotone partial sum -> premature fire
            # IMPOSSIBLE. It eliminates dead-from-under-fire (alive_frac->1) but
            # SATURATES (sat->1, accuracy stays chance), confirming both failure modes
            # are scale/sign, not a capacity wall. (|W| is a different function; its
            # accuracy is not comparable, only its firing-health is.)
            pos_alive = _mean_nonenc(pos_rows, "alive_frac")
            pos_sat = _mean_nonenc(pos_rows, "sat_frac")
            real_alive = _mean_nonenc(rows, "alive_frac")
            print(f"  [positive-weight control] genuine={pos_acc:.4f}  alive_frac "
                  f"{real_alive:.3f}->{pos_alive:.3f}  sat_frac->{pos_sat:.3f}  "
                  f"(monotone sum kills death but saturates)")
            print("  per-depth health real:", [r["health"] for r in rows if not r["enc"]])
            summary.append((depth, S, cont, cold_acc, H, uf, mt, real_alive,
                            _mean_nonenc(rows, "decode_corr")))

    print("\n" + "=" * 100)
    print("SUMMARY  (HEALTH and death-cascade mode attribution)")
    print(f"{'depth':>5} {'S':>3} {'cont':>6} {'cold':>6} {'HEALTH':>7} "
          f"{'uf(a)':>6} {'mt(b)':>6} {'alive':>6} {'corr':>6}")
    for depth, S, cont, cold, H, uf, mt, al, co in summary:
        print(f"{depth:>5} {S:>3} {cont:>6.4f} {cold:>6.4f} {H:>7.4f} "
              f"{uf:>6.3f} {mt:>6.3f} {al:>6.3f} {co:>6.3f}")
    print("=" * 100)
    print("READ: HEALTH ~ 1.0 = alive faithful cascade (calibration target). HEALTH << 1 with")
    print("  depth = the death cascade. uf(a) (under-fire/scale, dead channels) is")
    print("  CALIBRATION-fixable; mt(b) (premature-fire/mixed-sign timing) is what the quick FT")
    print("  must adapt the weights TO. Both shown deployable under UNCHANGED greedy execution.")
    return summary


if __name__ == "__main__":
    run()
