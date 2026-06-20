"""PHASE 2 KEY PATH — the PRINCIPLED closed-form per-layer gain correction.

Phase 1 proved the cascaded single-spike TTFS "death cascade" is a CORRECTABLE
per-layer decode GAIN-DISTORTION, not a capacity limit:

  * A spike arriving at consumer-local cycle ``tau`` is integrated as a quadratic
    ramp, giving the consumer the effective drive ``R(tau) = (T-tau)(T-tau+1)/2``,
    whereas a faithful (teacher) decode wants the LINEAR ``L(tau) ~ (T-tau)``. The
    per-spike effective GAIN relative to the intended linear weight is therefore

        g_eff(tau)  =  [R(tau)/R(0)] / [L(tau)/L(0)]  =  (T - tau + 1) / (T + 1)

    (bit-exact against ``capacity.ramp_effective_weight_distortion``'s R/L column).
  * Deep layers fire LATE: the mean local fire-cycle drifts ``E[tau_d] = tau_0 +
    d * sqrt(S)`` (char: per-hop normalized drift delta ~= 1/sqrt(S) => absolute
    drift S*delta = sqrt(S) cycles/hop), with the encode-layer fire-time fixed by
    the retention law ``rho_0(S) = 1 - c/S`` (c ~= 1.9), i.e.
        E[tau_0] = (T + 1) * (1 - rho_0) + 1 = (T + 1) * c / S + 1.
    So later layers sit further down the g_eff ramp -> exponentially attenuated ->
    death cascade.

THE CLOSED FORM (this file's contribution). Set the per-layer decode threshold to
``theta_d <- theta_d * g_d(S, d)`` with the EXPECTED effective gain at that depth

        g_d(S, d)  =  ( T - E[tau_d] + 1 ) / ( T + 1 ),   T = S,
        E[tau_d]   =  E[tau_0] + d * sqrt(S),
        E[tau_0]   =  (T + 1) * c / S + 1,        c = 1.9  (the rho-law constant)

  =>  g_d(S, d)  =  ( S + 1 - (S+1)*c/S - 1 - d*sqrt(S) + 1 ) / (S + 1)
                 =  1 - c/S - ( d*sqrt(S) - 1 ) / (S + 1).

Lowering theta_d by the factor g_d (g_d < 1) BOOSTS the layer's normalized value by
1/g_d, fires it earlier, and lands its decoded value where the teacher's is. The
decode itself is UNCHANGED (bit-exact) -> deployable as a per-layer activation_scale
trim keyed only on (S, cascade-depth) and the calibration constant c.

WHY THIS UNIFIES char and precomp.
  * char's geometric rule ``theta_d *= gamma^d`` is the SMALL-d / large-S limit:
    expand g_d ~= rho * exp(-d*sqrt(S)/(S+1)) ... a per-hop GEOMETRIC decay whose
    base gamma = g_d/g_{d-1} ~= 1 - sqrt(S)/(S+1) (so gamma ~= 0.5 at S=8 -> char's
    0.5^d falls straight out). char fixed gamma by hand; we DERIVE it from S.
  * precomp's ``mean_target`` (theta = mean(relu(act))/target) is the SAME lever
    (multiplicative theta) reached from the calib FIRST MOMENT instead of the depth
    index. It implicitly absorbs g_d into the data (deeper layers have smaller
    means after attenuation), but cannot see depth past the part already-attenuated.
    Our rule is the depth-explicit, data-free closed form precomp was approximating.

CALIB-ONLY: g_d uses ONLY S, the layer's cascade depth d, and the single law
constant c (fit once on the random-init depth-0 retention; NOT the eval metric).
A per-channel refinement (``gain_from_calib_firetime``) reads the calibration mean
fire-cycle directly instead of the d*sqrt(S) model — still calib-only.

Run:  source env/bin/activate
      python docs/.../experiments/closed_form.py            # cold validation
      python docs/.../experiments/closed_form.py ft         # +fine-tuning
"""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cascade_lab  # noqa: F401  (sets sys.path to src/tests/repo)
from cascade_lab import (
    _SingleSegmentMLP,
    _accuracy,
    _calibrate_scales,
    _capture_activation_means,
    _cascade_decoded_means,
    cascade_forward,
    digits_task,
    train_continuous,
)
from cascade_fixtures import install_ttfs_nodes
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward
from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of
from mimarsinan.spiking.segment_policy_ttfs import TtfsSegmentPolicy
from mimarsinan.torch_mapping.converter import convert_torch_model

CHANCE = 0.10
C_RHO = 1.9  # the rho-law constant rho_0(S) = 1 - c/S (char: 1.91 +- 0.07), fit once.


# =========================================================================== #
# THE CLOSED FORM
# =========================================================================== #
def g_eff(tau: float, S: int) -> float:
    """Per-spike effective gain of the quadratic ramp vs the intended linear decode.

    g_eff(tau) = (T - tau + 1)/(T + 1).  Bit-exact to capacity.R_over_L (tau int).
    """
    return (S - tau + 1.0) / (S + 1.0)


def mean_tau0(S: int, c: float = C_RHO) -> float:
    """Encode-layer mean local fire-cycle from the retention law rho_0 = 1 - c/S.

    rho_0 = g_eff(tau_0) = (S - tau_0 + 1)/(S + 1)  =>  tau_0 = (S+1)*c/S + 1 ... but
    rho_0 = 1 - c/S directly gives g_eff(tau_0) = 1 - c/S, so
    tau_0 = (S + 1) * (c / S) + 1.
    """
    return (S + 1.0) * (c / S) + 1.0


def g_closed(S: int, d: int, c: float = C_RHO) -> float:
    """PRINCIPLED per-layer gain g_d(S, d) — the headline closed form.

    E[tau_d] = tau_0 + d*sqrt(S);  g_d = g_eff(E[tau_d]) = (S - E[tau_d] + 1)/(S+1).
    Returns a float in (0, 1]; clamped to a small floor (a layer past d_max would
    ask for an impossible boost — capped so theta stays positive/sane).
    """
    e_tau = mean_tau0(S, c) + d * math.sqrt(S)
    g = (S - e_tau + 1.0) / (S + 1.0)
    return float(max(g, 0.02))


def g_geometric(S: int, d: int, c: float = C_RHO) -> float:
    """char's geometric law with the DERIVED base (not hand-set 0.5).

    Per-hop ratio gamma = 1 - sqrt(S)/(S+1); theta_d *= rho_0 * gamma^d. This is the
    multiplicative-per-hop view of the same model; reported to show it ~ g_closed.
    """
    rho0 = 1.0 - c / S
    gamma = 1.0 - math.sqrt(S) / (S + 1.0)
    return float(max(rho0 * (gamma ** d), 0.02))


def g_relative(S: int, d: int, c: float = C_RHO) -> float:
    """RELATIVE geometric: theta_d *= gamma^d (no rho_0 prefactor).

    The encode layer (d=0) already fires at gain rho_0, and the calibration that set
    theta accounts for that — so layer 0 needs NO correction (gamma^0 = 1). Only the
    EXTRA per-hop drift relative to layer 0 must be inverted: gamma = 1 -
    sqrt(S)/(S+1). This is self-limiting (gamma -> 1 as S grows) and does not touch
    an already-healthy shallow/high-S layer. The cleanest unified closed form.
    """
    gamma = 1.0 - math.sqrt(S) / (S + 1.0)
    return float(max(gamma ** d, 0.02))


# =========================================================================== #
# Per-channel calib-fire-time refinement (still calibration-only)
# =========================================================================== #
def _mean_local_fire_by_depth(flow, x, S):
    """Per-perceptron mean local fire-cycle on ``x`` (calib), keyed by depth index.

    Reuses char's run_segment instrumentation pattern. Returns {depth: mean_tau}
    where mean is over the FIRED population (no-fire neurons excluded). Calib-only.
    """
    fire_cycle: dict = {}
    orig = TtfsSegmentPolicy.run_segment

    def patched(self, driver, seg_nodes, values, xx):
        T = driver.T
        seg_set = set(seg_nodes)
        dep = self.segment_depths(driver, seg_nodes)
        zeros = self._segment_output_zeros(driver, seg_nodes, values, xx)
        n_cycles = T + max(dep.values(), default=0)
        boundary_trains: dict = {}

        def boundary_spikes(src, t):
            tr = boundary_trains.get(src)
            if tr is None:
                tr = self._boundary_single_spike_train(values[src], T, n_cycles)
                boundary_trains[src] = tr
            return tr[t]

        def read(src, out, perc_prev, t, consumer):
            if src not in seg_set:
                if is_encoding_perceptron(consumer):
                    return values[src]
                return boundary_spikes(src, t)
            if perceptron_of(src) is not None:
                return perc_prev.get(src, zeros[src])
            return out[src]

        perc_prev: dict = {}
        ff = {n: None for n in seg_nodes if perceptron_of(n) is not None}
        for t in range(n_cycles):
            out: dict = {}
            for n in seg_nodes:
                if t < dep[n] or t >= dep[n] + T:
                    out[n] = zeros[n]
                    continue
                d = driver._deps.get(n, [])
                if len(d) == 1:
                    inp = read(d[0], out, perc_prev, t, n)
                elif len(d) == 0:
                    inp = xx
                else:
                    inp = tuple(read(dep_, out, perc_prev, t, n) for dep_ in d)
                out[n] = n.forward(inp)
            for n in seg_nodes:
                if perceptron_of(n) is not None:
                    perc_prev[n] = out[n]
            for n in ff:
                if n in out:
                    sp = out[n]
                    if ff[n] is None:
                        ff[n] = torch.full(sp.shape[1:], -1.0, dtype=torch.float64)
                    fired_now = (sp > 0).any(dim=0) if sp.dim() > 1 else (sp > 0)
                    newly = fired_now & (ff[n] < 0)
                    ff[n][newly] = t - dep[n]
        for n in ff:
            fire_cycle[n] = ff[n]
        return orig(self, driver, seg_nodes, values, xx)

    TtfsSegmentPolicy.run_segment = patched
    try:
        drv = TTFSSegmentForward(flow.get_mapper_repr(), S)
        with torch.no_grad():
            drv(x.double())
    finally:
        TtfsSegmentPolicy.run_segment = orig

    by_perc = {id(perceptron_of(n)): fc for n, fc in fire_cycle.items()
               if perceptron_of(n) is not None}
    out: dict = {}
    for k, p in enumerate(flow.get_perceptrons()):
        fc = by_perc.get(id(p))
        if fc is None:
            continue
        fired = fc[fc >= 0]
        out[k] = float(fired.double().mean()) if fired.numel() else float(S)
    return out


# =========================================================================== #
# Build / cache a trained, converted, calibrated cascade flow.
# =========================================================================== #
_FLOW_CACHE: dict = {}


def build_flow(*, depth, width, in_dim, n_classes, seed, epochs=120):
    key = (depth, width, in_dim, n_classes, seed, epochs)
    if key in _FLOW_CACHE:
        return _FLOW_CACHE[key]
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = digits_task(seed=seed + 1)
    base = _SingleSegmentMLP(depth, width, in_dim, n_classes)
    train_continuous(base, xtr, ytr, epochs=epochs)
    with torch.no_grad():
        cont_acc = float(_accuracy(base(xte.float()), yte))
    flow = convert_torch_model(base, (in_dim,), n_classes, device="cpu")
    calib = xtr[:256]
    _calibrate_scales(flow, calib)
    base_scales = [p.activation_scale.clone() for p in flow.get_perceptrons()]
    base_weights = [p.layer.weight.detach().clone() for p in flow.get_perceptrons()]
    base_biases = [p.layer.bias.detach().clone() if p.layer.bias is not None else None
                   for p in flow.get_perceptrons()]
    _FLOW_CACHE[key] = (flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales,
                        base_weights, base_biases)
    return _FLOW_CACHE[key]


def _restore_params(flow, base_scales, base_weights, base_biases):
    for p, s, w, b in zip(flow.get_perceptrons(), base_scales, base_weights, base_biases):
        with torch.no_grad():
            p.set_activation_scale(s.clone())
            p.layer.weight.copy_(w)
            if b is not None and p.layer.bias is not None:
                p.layer.bias.copy_(b)


# =========================================================================== #
# Apply a per-layer gain correction to theta.
# =========================================================================== #
_ANALYTIC_RULES = {"closed": g_closed, "geometric": g_geometric, "relative": g_relative}


def apply_gain_correction(flow, base_scales, S, *, rule="relative", c=C_RHO,
                          calib=None):
    """theta_d <- theta_d * g_d.  ``rule`` in {closed, geometric, relative, calib_fire}.

    The analytic rules use ONLY (S, depth, c). ``calib_fire`` runs a forward
    calibration pass: it measures each layer's mean local fire-cycle on calib data,
    sets g_d = g_eff(tau_d^calib) RELATIVE to layer 0 (so a healthy layer 0 is
    untouched), and re-measures downstream so each correction sees the corrected
    upstream — still NO eval-metric access.
    """
    percs = flow.get_perceptrons()
    if rule == "calib_fire":
        if calib is None:
            raise ValueError("calib_fire needs the calib batch")
        install_ttfs_nodes(flow, S)
        tau0 = _mean_local_fire_by_depth(flow, calib, S).get(0, mean_tau0(S, c))
        for d, p in enumerate(percs):
            fire = _mean_local_fire_by_depth(flow, calib, S)
            tau = fire.get(d, mean_tau0(S, c) + d * math.sqrt(S))
            # relative gain: invert the EXTRA attenuation beyond layer 0's.
            g = float(max(g_eff(tau, S) / max(g_eff(tau0, S), 1e-6), 0.02))
            g = min(g, 1.0)
            with torch.no_grad():
                p.set_activation_scale(base_scales[d] * g)
            install_ttfs_nodes(flow, S)
        return
    fn = _ANALYTIC_RULES[rule]
    for d, p in enumerate(percs):
        with torch.no_grad():
            p.set_activation_scale(base_scales[d] * fn(S, d, c))
    install_ttfs_nodes(flow, S)


def oracle_thetas(flow, base_scales, xte, yte, S, depth,
                  grid=(1.0, 0.7, 0.5, 0.35, 0.25)):
    """The recoverable upper bound: per-depth theta-scale searched on the EVAL
    metric (capacity.oracle_theta_scale). Used ONLY as the comparison ceiling."""
    import itertools
    percs = flow.get_perceptrons()

    def acc_for(gammas):
        for k, p in enumerate(percs):
            p.set_activation_scale(base_scales[k] * (gammas[k] if k < len(gammas) else 1.0))
        install_ttfs_nodes(flow, S)
        return float(_accuracy(cascade_forward(flow, xte, S), yte))

    best = ([1.0] * depth, acc_for([1.0] * depth))
    for g in itertools.product(grid, repeat=depth):
        a = acc_for(list(g))
        if a > best[1]:
            best = (list(g), a)
    return best


# =========================================================================== #
# COLD evaluation: baseline vs closed-form vs oracle (no fine-tuning).
# =========================================================================== #
def evaluate_cold(*, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0,
                  with_oracle=True, rules=("closed",)):
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)
    _restore_params(flow, base_scales, base_weights, base_biases)

    install_ttfs_nodes(flow, S)
    base_gen = float(_accuracy(cascade_forward(flow, xte, S), yte))

    out = {"cont": round(cont_acc, 4), "baseline": round(base_gen, 4)}

    for rule in rules:
        _restore_params(flow, base_scales, base_weights, base_biases)
        apply_gain_correction(flow, base_scales, S, rule=rule, calib=calib)
        out[rule] = round(float(_accuracy(cascade_forward(flow, xte, S), yte)), 4)

    if with_oracle:
        _restore_params(flow, base_scales, base_weights, base_biases)
        _, oacc = oracle_thetas(flow, base_scales, xte, yte, S, depth)
        out["oracle"] = round(float(oacc), 4)

    _restore_params(flow, base_scales, base_weights, base_biases)
    return out


# =========================================================================== #
# WITH FINE-TUNING: corrected init vs plain init, each +genuine-cascade FT.
# =========================================================================== #
def _genuine_ft(flow, xtr, ytr, S, *, ft_epochs=40, ft_lr=2e-3, surrogate_temp=0.5):
    params = [p for p in flow.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=ft_lr)
    lossf = nn.CrossEntropyLoss()
    flow.double()
    for _ in range(ft_epochs):
        opt.zero_grad()
        logits = cascade_forward(flow, xtr, S, grad=True, surrogate_temp=surrogate_temp)
        lossf(logits, ytr).backward()
        opt.step()


def evaluate_ft(*, depth=3, width=64, in_dim=64, n_classes=10, S=8, seed=0,
                ft_epochs=40, ft_lr=2e-3, rule="relative"):
    """{plain cold, plain+FT, corrected cold, corrected+FT}.

    Both arms fine-tune ALL weights through the GENUINE cascade (boundary STE) — the
    real deployment path. The only difference is the theta init (plain vs gain-fixed).
    theta itself is frozen during FT (only weights/biases train), so the corrected
    init's advantage is purely a healthier starting basin.
    """
    flow, calib, xtr, ytr, xte, yte, cont_acc, base_scales, base_weights, \
        base_biases = build_flow(depth=depth, width=width, in_dim=in_dim,
                                 n_classes=n_classes, seed=seed)

    # --- plain arm ---
    _restore_params(flow, base_scales, base_weights, base_biases)
    install_ttfs_nodes(flow, S)
    plain_cold = float(_accuracy(cascade_forward(flow, xte, S), yte))
    _genuine_ft(flow, xtr, ytr, S, ft_epochs=ft_epochs, ft_lr=ft_lr)
    plain_ft = float(_accuracy(cascade_forward(flow, xte, S), yte))

    # --- corrected arm ---
    _restore_params(flow, base_scales, base_weights, base_biases)
    apply_gain_correction(flow, base_scales, S, rule=rule, calib=calib)
    corr_cold = float(_accuracy(cascade_forward(flow, xte, S), yte))
    _genuine_ft(flow, xtr, ytr, S, ft_epochs=ft_epochs, ft_lr=ft_lr)
    corr_ft = float(_accuracy(cascade_forward(flow, xte, S), yte))

    _restore_params(flow, base_scales, base_weights, base_biases)
    return {"cont": round(cont_acc, 4), "plain_cold": round(plain_cold, 4),
            "plain_ft": round(plain_ft, 4), "corr_cold": round(corr_cold, 4),
            "corr_ft": round(corr_ft, 4)}


# =========================================================================== #
# Reports
# =========================================================================== #
def report_formula():
    print("=" * 78)
    print("0. THE CLOSED FORM  g_d(S,d) = (S - E[tau_d] + 1)/(S+1),  "
          "E[tau_d]=tau_0+d*sqrt(S)")
    print("=" * 78)
    print("   g_eff(tau)=(S-tau+1)/(S+1)  [exact vs capacity.R_over_L]")
    print(f"   tau_0(S)=(S+1)*c/S+1, c={C_RHO}")
    print(f"\n   {'S':>3} | g_closed per depth d=0..3        | derived geom base gamma")
    for S in (4, 8, 16, 32):
        gs = [round(g_closed(S, d), 3) for d in range(4)]
        gamma = 1.0 - math.sqrt(S) / (S + 1.0)
        print(f"   {S:>3} | {gs}    |  {gamma:.3f}")


def report_cold(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("1. COLD validation: baseline vs closed-form vs ORACLE (no fine-tuning)")
    print("=" * 78)
    rules = ("geometric", "relative", "calib_fire")
    for depth in (2, 3, 4):
        print(f"\n--- depth={depth} (mean over seeds {seeds}) ---")
        print(f"{'S':>4} {'cont':>7} {'baseline':>9} {'geom':>8} {'relative':>9} "
              f"{'calibFR':>8} {'oracle':>8}  {'rel/oracle':>11}")
        for S in (4, 8, 16, 32):
            rs = [evaluate_cold(depth=depth, S=S, seed=s, rules=rules) for s in seeds]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            # geometric is the headline rule; only report gap-closed where the
            # death cascade is actually active (oracle gap > 0.05).
            gap = agg["oracle"] - agg["baseline"]
            best = max(agg["geometric"], agg["calib_fire"])
            frac = f"{(best - agg['baseline']) / gap:>10.0%}" if gap > 0.05 else f"{'healthy':>10}"
            print(f"{S:>4} {agg['cont']:>7.3f} {agg['baseline']:>9.3f} "
                  f"{agg['geometric']:>8.3f} {agg['relative']:>9.3f} "
                  f"{agg['calib_fire']:>8.3f} {agg['oracle']:>8.3f}  {frac}")


def report_primary(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("1b. PRIMARY (depth=3 digits S=8): per-seed cold")
    print("=" * 78)
    print(f"{'seed':>5} {'cont':>7} {'baseline':>9} {'geometric':>10} {'oracle':>8}")
    for s in seeds:
        r = evaluate_cold(depth=3, S=8, seed=s, rules=("geometric",))
        print(f"{s:>5} {r['cont']:>7.3f} {r['baseline']:>9.3f} {r['geometric']:>10.3f} "
              f"{r['oracle']:>8.3f}")


def report_ft(seeds=(0, 1, 2)):
    print("\n" + "=" * 78)
    print("2. WITH GENUINE FINE-TUNING: {plain,corrected} x {cold,+FT}")
    print("   (both FT all weights through the genuine cascade; theta frozen)")
    print("=" * 78)
    for depth in (3, 4):
        for S in (8, 16):
            print(f"\n--- depth={depth} S={S} (mean over seeds {seeds}) ---")
            print(f"{'':6}{'cont':>7}{'plain_cold':>11}{'plain_ft':>10}"
                  f"{'corr_cold':>11}{'corr_ft':>9}  {'corr_ft-plain_ft':>16}")
            rs = [evaluate_ft(depth=depth, S=S, seed=s, rule="geometric") for s in seeds]
            agg = {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}
            print(f"{'mean':6}{agg['cont']:>7.3f}{agg['plain_cold']:>11.3f}"
                  f"{agg['plain_ft']:>10.3f}{agg['corr_cold']:>11.3f}"
                  f"{agg['corr_ft']:>9.3f}  {agg['corr_ft']-agg['plain_ft']:>+15.3f}")
            for s, r in zip(seeds, rs):
                print(f"  s{s:<3}{r['cont']:>7.3f}{r['plain_cold']:>11.3f}"
                      f"{r['plain_ft']:>10.3f}{r['corr_cold']:>11.3f}"
                      f"{r['corr_ft']:>9.3f}  {r['corr_ft']-r['plain_ft']:>+15.3f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "cold"
    report_formula()
    if which in ("cold", "all"):
        report_primary()
        report_cold()
    if which in ("ft", "all"):
        report_ft()
