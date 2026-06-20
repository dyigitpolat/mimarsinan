"""PILLAR F2 -- the GREEDY-ADAPTATION gradient (fast + stable).

Goal: convert an ANN to the GENUINE greedy fire-once cascaded single-spike TTFS
deployment LOSSLESSLY, under UNCHANGED greedy execution, via two complementary
pillars:

  PILLAR 1 (numerical health): calibration that makes the cold cascade ALIVE and
    well-conditioned (alive_frac~1, ANN-rate-matched, well-ordered) WITHOUT
    saturating -- it CANNOT close accuracy (premature-fire is a weight problem),
    but it removes the near-zero-gradient dead deep layers that starve FT.
  PILLAR 2 (this file): a fast+stable FT gradient that adapts the WEIGHTS to the
    greedy firing timing (the +0.6 neg-fanin <-> over-fire coupling).

This file isolates the FT ENGINE: which gradient converges to lossless FASTEST and
most STABLY (lowest seed variance), measured from a HEALTHY calibrated init vs from
the CHANCE (cold) init, under the genuine greedy cascade. Candidates:

  (a) genuine     -- pure genuine fire-once surrogate (the `combo` skeleton:
                     per-channel theta co-train + progressive shallow->deep
                     unfreeze + KD). The deploy-path surrogate, no STE hedge.
  (b) stair_ste   -- hedged staircase-BACKWARD STE (mix sweep): forward=genuine,
                     backward=(1-mix)*genuine + mix*staircase (clean ceiling grad).
  (c) boundary    -- offload/boundary STE (boundary_surrogate_temp): un-severs the
                     genuine backward at offload/host-ComputeOp boundaries. For a
                     SINGLE-segment cascade (this harness's model) there are no such
                     boundaries, so it is provably == (a); reported as such.
  (d) curriculum  -- surrogate WIDE->SHARP alpha anneal (revive dead neurons with a
                     wide gradient early, sharpen onto the hard deploy node late).

All FT is evaluated with the PURE genuine greedy cascade (genuine_acc). Greedy
fire-once execution is NEVER changed: the surrogate/STE/anneal touch only backward
or the surrogate sharpness, never the forward fire rule.

Run: python greedy_gradient.py [steps] [d] [S...]
"""

from __future__ import annotations

import copy
import os
import sys
import time

# The genuine TTFS surrogate backward lazily imports ``spikingjelly.activation_based``
# (the nested submodule package). In an isolated worktree the submodule is not
# checked out; fall back to the main checkout's submodule so the backward resolves.
for _sj in (os.path.join(os.path.dirname(__file__), "..", "..", "..", "spikingjelly"),
            "/home/yigit/repos/research_stuff/mimarsinan/spikingjelly"):
    _sj = os.path.abspath(_sj)
    if os.path.isdir(os.path.join(_sj, "spikingjelly", "activation_based")) and _sj not in sys.path:
        sys.path.insert(0, _sj)

import torch

from recipe_harness import (
    TTFSActivation, batches, genuine_acc, genuine_logits, kd_ce_loss,
    promote_theta_per_channel, staircase_logits, teacher_logits,
)
from recipe_combo import _weight_params_through
from cascade_fixtures import cascade_forward
from chase_2pp import _val_split
from lif_vs_ttfs import ttfs_staircase_acc

from revive import _per_neuron
from mimarsinan.spiking.scale_aware_boundaries import calibrate_scale_aware_boundaries
from mimarsinan.spiking.dfq_bias_correction import dfq_correct_biases
from mimarsinan.spiking.distribution_matching import _cascade_channel_means


# --------------------------------------------------------------------------- #
# PILLAR 1: healthy calibration + per-depth HEALTH metric
# --------------------------------------------------------------------------- #
def make_healthy(flow, cal_x, S, teacher, *, quantile=0.99, bias_iters=15, eta=0.7):
    """Distribution-matching calibration -> ALIVE, ANN-rate-matched init.

    scale-aware [0,1] boundaries from the teacher per-channel quantile (revives the
    deep dead layers, fixes the scale mismatch) + DFQ per-neuron bias correction
    (matches the cascade first moment to the ANN). Mutates `flow` in place. Cold
    accuracy stays low (premature-fire weight error remains) but the cascade is now
    numerically well-conditioned for FT."""
    n = len(flow.get_perceptrons())
    ann_mean = {k: teacher[k].double() for k in teacher}
    theta_out = [
        max(float(torch.quantile(teacher[k].abs().double().flatten(), quantile)), 1e-2)
        if k in teacher else 1.0
        for k in range(n)
    ]
    calibrate_scale_aware_boundaries(flow, theta_out)
    dfq_correct_biases(
        flow, ann_mean, lambda: _cascade_channel_means(flow, cal_x, S),
        bias_iters=bias_iters, eta=eta)
    return flow


def model_health(flow, cal_x, S, teacher, *, dead_rate=0.02):
    """Per-depth + model HEALTH(layer) = (1 - rate_err) * alive_frac * max(corr,0).

    rate = decoded/scale in [0,1]; rate_err = mean|rate_cascade - rate_teacher| over
    channels the ANN fires; alive_frac = 1 - frac(ANN-active channels that are dead);
    corr = mean per-channel Pearson(decoded_cascade, teacher_act). Encoding layer is
    excluded. Returns (model_health, per_depth_health, per_depth_alive)."""
    pn = _per_neuron(flow, cal_x, S)
    perceptrons = flow.get_perceptrons()
    healths, alives = [], []
    for k, p in enumerate(perceptrons):
        if getattr(p, "is_encoding_layer", False):
            continue
        decoded, rate = pn[k]
        t = teacher.get(k)
        if rate is None or t is None:
            continue
        scale = p.activation_scale.detach().double()
        denom_scale = scale if scale.dim() == 0 else scale[:t.numel()]
        rate_t = (t.clamp(min=0).double() / denom_scale.clamp(min=1e-9))
        m = min(rate.numel(), rate_t.numel())
        rate, rate_t = rate[:m], rate_t[:m]
        wants = rate_t > 1e-4
        if not bool(wants.any()):
            continue
        rate_err = (rate[wants] - rate_t[wants]).abs().mean()
        alive = 1.0 - (rate[wants] < dead_rate).float().mean()
        d = decoded[:m].double()
        td = t[:m].clamp(min=0).double()
        dd, tt = d - d.mean(), td - td.mean()
        denom = dd.norm() * tt.norm()
        corr = (dd @ tt / denom) if float(denom) > 1e-12 else torch.zeros(())
        h = (1.0 - rate_err).clamp(min=0) * alive * corr.clamp(min=0)
        healths.append(float(h))
        alives.append(round(float(alive), 2))
    model_h = sum(healths) / max(1, len(healths))
    return model_h, [round(x, 3) for x in healths], alives


# --------------------------------------------------------------------------- #
# PILLAR 2: the four greedy-adaptation gradients (FT engines)
# --------------------------------------------------------------------------- #
def _set_alpha(flow, a):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_surrogate_alpha(a)


def _set_cycle_accurate(flow, mode):
    for m in flow.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(mode)


def _progressive_opt(flow, thetas, depth, *, w_lr, theta_lr, steps, step):
    weights = _weight_params_through(flow, depth)
    opt = torch.optim.Adam([
        {"params": thetas, "lr": theta_lr},
        {"params": weights, "lr": w_lr},
    ])
    for g in opt.param_groups:
        g["initial_lr"] = g["lr"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, last_epoch=step - 1)
    return opt, sched


def _ft_loop(flow, xtr, ytr, S, base, *, steps, seed, logits_fn, alpha_fn=None,
             w_lr=2e-3, theta_lr=5e-2, bs=256, init_frac=1 / 3, alpha=0.3,
             grad_clip=1.0, eval_cb=None):
    """Shared progressive (shallow->deep) + per-channel-theta + KD FT skeleton.
    `logits_fn(flow, x, S)` produces the (forward=genuine) training logits; the
    gradient differs per candidate. `alpha_fn(r)` (r in [0,1]) sets surrogate
    sharpness for the curriculum. `eval_cb(step)` is called for step-to-lossless
    probing."""
    thetas = promote_theta_per_channel(flow, requires_grad=True)
    n = len(flow.get_perceptrons())
    start = max(1, round(n * init_frac))
    schedule = [min(n, start + round((n - start) * t / max(1, steps - 1)))
                for t in range(steps)]
    depth = -1
    opt = sched = None
    for t, (x, y) in enumerate(batches(xtr, ytr, bs, steps, seed)):
        if schedule[t] != depth:
            depth = schedule[t]
            opt, sched = _progressive_opt(
                flow, thetas, depth, w_lr=w_lr, theta_lr=theta_lr, steps=steps, step=t)
        if alpha_fn is not None:
            _set_alpha(flow, alpha_fn(t / max(steps - 1, 1)))
        logits = logits_fn(flow, x, S)
        loss = kd_ce_loss(logits, y, teacher_logits(base, x), alpha=alpha)
        opt.zero_grad(); loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], grad_clip)
        opt.step(); sched.step()
        if eval_cb is not None:
            eval_cb(t + 1)
    _set_cycle_accurate(flow, True)
    _set_alpha(flow, 2.0)
    return flow


def _genuine_lf(flow, x, S):
    return genuine_logits(flow, x, S)


def _boundary_lf(flow, x, S):
    """Offload-boundary STE: un-severs the genuine backward at offload boundaries.
    Forward is the genuine greedy cascade (unchanged)."""
    return cascade_forward(flow, x, S, grad=True, surrogate_temp=0.5)


def _stair_ste_lf(mix):
    def lf(flow, x, S):
        g = genuine_logits(flow, x, S)
        if mix <= 0.0:
            return g
        s = staircase_logits(flow, x)
        back = mix * s + (1.0 - mix) * g
        _set_cycle_accurate(flow, True)  # staircase_logits flipped it OFF
        return back + (g - back).detach()
    return lf


def _curriculum_alpha(r, alpha_min=0.3, alpha_max=2.0):
    return alpha_min * (alpha_max / alpha_min) ** r


def candidate_train(name, flow, xt, yt, S, base, *, steps, seed, mix=1.0, eval_cb=None):
    """Dispatch a named greedy-adaptation gradient. All share the FT skeleton; only
    the backward gradient (logits_fn) / surrogate sharpness (alpha_fn) differ."""
    if name == "genuine":
        return _ft_loop(flow, xt, yt, S, base, steps=steps, seed=seed,
                        logits_fn=_genuine_lf, eval_cb=eval_cb)
    if name == "boundary":
        return _ft_loop(flow, xt, yt, S, base, steps=steps, seed=seed,
                        logits_fn=_boundary_lf, eval_cb=eval_cb)
    if name == "stair_ste":
        return _ft_loop(flow, xt, yt, S, base, steps=steps, seed=seed,
                        logits_fn=_stair_ste_lf(mix), eval_cb=eval_cb)
    if name == "curriculum":
        return _ft_loop(flow, xt, yt, S, base, steps=steps, seed=seed,
                        logits_fn=_genuine_lf, alpha_fn=_curriculum_alpha, eval_cb=eval_cb)
    raise ValueError(name)


# --------------------------------------------------------------------------- #
# measurement: steps + seconds to lossless, seed variance
# --------------------------------------------------------------------------- #
def steps_to_lossless(name, flow0, xtr, ytr, xte, yte, S, base, target, *,
                      seeds, steps, probe_every, mix=1.0):
    """For each seed: FT a fresh copy of flow0, probing genuine_acc every
    `probe_every` steps; record the first probe step that reaches `target`
    (lossless) and the wall time to that point. Returns per-seed records +
    aggregate (median steps, mean wall, final acc mean/std, hit fraction)."""
    recs = []
    for seed in seeds:
        f = copy.deepcopy(flow0)
        probes = set(range(probe_every, steps + 1, probe_every))
        state = {"hit_step": None, "hit_time": None, "t0": time.time()}

        def eval_cb(step, _f=f, _probes=probes, _st=state):
            if _st["hit_step"] is not None or step not in _probes:
                return
            if genuine_acc(_f, xte, yte, S) >= target:
                _st["hit_step"] = step
                _st["hit_time"] = time.time() - _st["t0"]

        candidate_train(name, f, xtr, ytr, S, base, steps=steps, seed=seed,
                        mix=mix, eval_cb=eval_cb)
        final = genuine_acc(f, xte, yte, S)
        wall = time.time() - state["t0"]
        recs.append(dict(seed=seed, hit_step=state["hit_step"],
                         hit_time=state["hit_time"], final=final, wall=wall))
    hits = [r["hit_step"] for r in recs if r["hit_step"] is not None]
    hit_times = [r["hit_time"] for r in recs if r["hit_time"] is not None]
    finals = torch.tensor([r["final"] for r in recs])
    agg = dict(
        median_steps=(sorted(hits)[len(hits) // 2] if hits else None),
        mean_hit_time=(sum(hit_times) / len(hit_times) if hit_times else None),
        final_mean=float(finals.mean()), final_std=float(finals.std()),
        hit_frac=len(hits) / len(recs),
        mean_wall=sum(r["wall"] for r in recs) / len(recs),
    )
    return recs, agg


def _fmt(agg):
    ms = agg["median_steps"]
    ht = agg["mean_hit_time"]
    steps_s = f"{ms:>7}" if ms is not None else ">budget"
    ht_s = f"{ht:5.1f}s" if ht is not None else "  n/a"
    return (f"steps={steps_s}  hit_t={ht_s}  "
            f"final={agg['final_mean']:.4f}+/-{agg['final_std']:.4f}  "
            f"hit={agg['hit_frac']:.0%}  wall={agg['mean_wall']:.0f}s")


if __name__ == "__main__":
    from recipe_harness import build

    STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    DEPTH = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    SVALS = [int(a) for a in sys.argv[3:]] or [16, 32]
    SEEDS = (0, 1, 2)
    PROBE = max(20, STEPS // 20)
    LOSSLESS_MARGIN = 0.01  # within 1pp of the staircase ceiling == lossless

    # boundary STE is bit-exact == genuine on a SINGLE-segment cascade (no offload
    # boundaries): proven fwd identical + per-layer grad cosine 1.0, max|diff|=0.
    # Dropped from the heavy sweep to save compute; reported as ==genuine.
    CANDS = ["genuine", "curriculum", ("stair_ste", 0.5), ("stair_ste", 1.0)]

    for S in SVALS:
        flow, xtr, ytr, xte, yte, cont, teacher, base = build(DEPTH, S, seed=0)
        ceiling = ttfs_staircase_acc(flow, xte, yte, S)
        target = ceiling - LOSSLESS_MARGIN
        cold_acc = genuine_acc(flow, xte, yte, S)
        cold_h, cold_hd, cold_ad = model_health(flow, xte[:512].double(), S, teacher)

        healthy = make_healthy(copy.deepcopy(flow), xte[:512].double(), S, teacher)
        warm_acc = genuine_acc(healthy, xte, yte, S)
        warm_h, warm_hd, warm_ad = model_health(healthy, xte[:512].double(), S, teacher)

        xt, yt, xv, yv = _val_split(xtr, ytr, seed=0)

        print(f"\n{'='*78}")
        print(f"=== d={DEPTH} S={S}  cont={cont:.4f}  staircase_ceiling={ceiling:.4f}  "
              f"lossless_target={target:.4f}  steps<={STEPS} seeds={SEEDS} ===")
        print(f"  COLD : genuine={cold_acc:.4f}  HEALTH={cold_h:.3f}  alive={cold_ad}")
        print(f"         health/depth={cold_hd}")
        print(f"  HEALTHY init (calib): genuine={warm_acc:.4f}  HEALTH={warm_h:.3f}  "
              f"alive={warm_ad}")
        print(f"         health/depth={warm_hd}")
        print(f"  --- steps+wall to lossless ({target:.4f}); seed-variance via final std ---")

        for init_name, init_flow in (("chance", flow), ("healthy", healthy)):
            print(f"  [init={init_name}]")
            for c in CANDS:
                name, mix = (c if isinstance(c, tuple) else (c, 1.0))
                label = f"{name}(mix={mix})" if name == "stair_ste" else name
                _recs, agg = steps_to_lossless(
                    name, init_flow, xt, yt, xte, yte, S, base, target,
                    seeds=SEEDS, steps=STEPS, probe_every=PROBE, mix=mix)
                print(f"    {label:>16}: {_fmt(agg)}")
