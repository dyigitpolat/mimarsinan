"""Per-cascade-depth gain correction for the deployed single-spike TTFS cascade.

The deployed ramp-integrate decode gives an arriving spike at local time ``tau`` an
effective gain ``g_eff(tau) = (T - tau + 1)/(T + 1)`` — the integral of the
``(T-tau)``-length ramp — instead of the intended linear gain. Deep layers fire
late (latency ~1 cycle/hop; mean fire-cycle drifts ``~ d*sqrt(S)``), so their values
are systematically under-decoded and the error compounds geometrically with depth
(the "death cascade", budget ``d_max(S) ~= 0.56*sqrt(S)``).

This inverts that known per-depth attenuation with a per-layer multiplicative trim of
``activation_scale`` (the decode threshold theta) — a pure CALIBRATION change: the
decode mechanism is untouched, so NF<->SCM parity is preserved. Derived in
``docs/research_artifacts_for_cascaded_ttfs_tuning`` (closed-form g_d from the
quadratic-ramp gain + fire-time drift); recovers ~85% of the cold conversion gap on
the isolated benchmark and, composed with the genuine-forward fine-tune, reaches
near-continuous accuracy.
"""

from __future__ import annotations

import math

from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import SegmentForwardDriver, TtfsSegmentPolicy
from mimarsinan.spiking.segment_partition import perceptron_of

_G_FLOOR = 0.02


def gamma_of(S: int) -> float:
    """Per-hop retention ratio gamma = 1 - sqrt(S)/(S+1) (derived from the fire-time
    drift sqrt(S) cycles/hop and the (T-tau)/(T+1) gain). gamma -> 1 as S grows, so
    the correction self-limits in the already-healthy high-S regime."""
    return 1.0 - math.sqrt(S) / (S + 1.0)


def g_relative(S: int, d: int) -> float:
    """RELATIVE per-depth gain factor theta_d *= gamma^d (no prefactor).

    Layer 0 (the segment entry / encoding layer, fixed by the input-encoding
    contract) is left unchanged (gamma^0 = 1); only the EXTRA per-hop drift relative
    to it is inverted. Self-limiting and the safest default — it never retunes a
    healthy shallow/high-S layer."""
    return max(gamma_of(S) ** int(d), _G_FLOOR)


def g_geometric(S: int, d: int, c: float = 1.9) -> float:
    """Absolute geometric gain factor theta_d *= rho_0 * gamma^d, rho_0 = 1 - c/S.

    Corrects the encode layer too (rho_0 < 1). Stronger cold recovery but can
    over-correct already-healthy layers, so prefer ``relative`` unless the cascade is
    deep/low-S throughout."""
    return max((1.0 - c / S) * (gamma_of(S) ** int(d)), _G_FLOOR)


_RULES = {"relative": g_relative, "geometric": lambda S, d, c: g_geometric(S, d, c)}


def per_perceptron_cascade_depth(model_repr) -> dict:
    """Map ``id(perceptron) -> cascade depth`` (perceptron-hops from its segment
    entry = local ``ChipLatency``) across all neural segments."""
    driver = SegmentForwardDriver(model_repr, 1, TtfsSegmentPolicy())
    policy = driver.policy
    depths: dict = {}
    for seg_nodes in driver.segments.values():
        for node, dep in policy.segment_depths(driver, seg_nodes).items():
            perceptron = perceptron_of(node)
            if perceptron is not None:
                depths[id(perceptron)] = int(dep)
    return depths


def cascaded_gain_factors(model, T: int, *, rule: str = "relative", c: float = 1.9) -> dict:
    """Per-perceptron target gain factor ``g_d`` keyed by ``id(perceptron)`` (no
    mutation) — the rate-1 endpoint of the rate-gated correction.

    Encoding/segment-entry perceptrons (``is_encoding_layer``) are PINNED (g=1.0):
    their scale is fixed by the input spike-encoding contract, and retuning it
    breaks NF↔SCM deployment parity (the geometric rule's ρ₀ would otherwise shrink
    it and crater parity — see scale_aware_boundaries)."""
    if rule not in _RULES:
        raise ValueError(f"gain-correction rule must be one of {sorted(_RULES)}; got {rule!r}")
    gfn = _RULES[rule]
    S = int(T)
    depths = per_perceptron_cascade_depth(model.get_mapper_repr())
    out: dict = {}
    for perceptron in model.get_perceptrons():
        if getattr(perceptron, "is_encoding_layer", False):
            out[id(perceptron)] = 1.0
            continue
        d = depths.get(id(perceptron), 0)
        out[id(perceptron)] = float(gfn(S, d, c)) if rule == "geometric" else float(gfn(S, d))
    return out


def apply_gain_at_rate(
    model, base_scales, factors: dict, rate: float, *, input_data_scale: float = 1.0,
) -> None:
    """Rate-gated gain correction: set ``activation_scale = base * g_d**rate`` per
    perceptron (rate 0 → base, rate 1 → full correction) and re-propagate input
    scales. Co-ramped with the KD blend so the model adapts to the calibration and
    the spiking dynamics together. ``base_scales`` is the parallel list of original
    (uncorrected) per-perceptron scales captured before ramping."""
    r = float(rate)
    for perceptron, base in zip(model.get_perceptrons(), base_scales):
        g = float(factors.get(id(perceptron), 1.0))
        perceptron.set_activation_scale(float(base) * (g ** r))
    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)


def apply_cascaded_gain_correction(
    model, T: int, *, rule: str = "relative", c: float = 1.9,
    input_data_scale: float = 1.0,
) -> dict:
    """Trim each perceptron's ``activation_scale`` by the per-depth gain factor and
    re-propagate input scales (so the boundary un-normalization stays consistent and
    NF<->SCM parity holds). Encoding/entry (``is_encoding_layer``) layers are PINNED
    (g=1.0, parity-critical). Returns a stats dict. Mutates the model in place."""
    S = int(T)
    factors = cascaded_gain_factors(model, T, rule=rule, c=c)
    depths = per_perceptron_cascade_depth(model.get_mapper_repr())

    by_depth: dict = {}
    corrected = 0
    for perceptron in model.get_perceptrons():
        d = depths.get(id(perceptron), 0)
        g = float(factors[id(perceptron)])
        by_depth.setdefault(d, round(g, 4))
        if abs(g - 1.0) > 1e-9:
            perceptron.set_activation_scale(float(perceptron.activation_scale) * g)
            corrected += 1

    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)
    return {
        "rule": rule, "S": S, "gamma": round(gamma_of(S), 4),
        "factor_by_depth": dict(sorted(by_depth.items())),
        "n_corrected": corrected, "n_perceptrons": len(list(model.get_perceptrons())),
    }
