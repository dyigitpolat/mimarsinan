"""Per-cascade-depth gain correction for the deployed single-spike TTFS cascade."""

from __future__ import annotations

import math

from mimarsinan.spiking.scale_aware_boundaries import propagate_boundary_input_scales
from mimarsinan.spiking.segment_forward import SegmentForwardDriver, TtfsSegmentPolicy
from mimarsinan.spiking.segment_partition import perceptron_of

_G_FLOOR = 0.02


def gamma_of(S: int) -> float:
    """Per-hop retention ratio gamma = 1 - sqrt(S)/(S+1); gamma -> 1 as S grows."""
    return 1.0 - math.sqrt(S) / (S + 1.0)


def g_relative(S: int, d: int) -> float:
    """Relative per-depth gain factor theta_d *= gamma^d (layer 0 unchanged)."""
    return max(gamma_of(S) ** int(d), _G_FLOOR)


def g_geometric(S: int, d: int, c: float = 1.9) -> float:
    """Absolute geometric gain factor theta_d *= rho_0 * gamma^d, rho_0 = 1 - c/S
    (corrects the encode layer too; can over-correct healthy layers)."""
    return max((1.0 - c / S) * (gamma_of(S) ** int(d)), _G_FLOOR)


_RULES = {"relative": g_relative, "geometric": lambda S, d, c: g_geometric(S, d, c)}


def per_perceptron_cascade_depth(model_repr) -> dict:
    """Map ``id(perceptron) -> cascade depth`` (perceptron-hops from its segment entry)."""
    driver = SegmentForwardDriver(model_repr, 1, TtfsSegmentPolicy())
    policy = driver.policy
    depths: dict = {}
    for seg_nodes in driver.segments.values():
        for node, dep in policy.segment_depths(driver, seg_nodes).items():
            perceptron = perceptron_of(node)
            if perceptron is not None:
                depths[id(perceptron)] = int(dep)
    return depths


def require_intra_segment_depth(depths: dict) -> None:
    """Fail loud when gain correction is requested on a boundary-dominated graph.

    gamma(S)^d inverts INTRA-segment ramp attenuation; with all intra-segment
    depths <= 1 the graph's composition cost lives at decode/re-encode boundaries
    where the per-hop drift is inflation — the trim is mis-signed there (§5p).
    """
    max_depth = max(depths.values(), default=0)
    if max_depth <= 1:
        raise ValueError(
            "gain correction requested on a boundary-dominated graph "
            f"(max intra-segment depth {max_depth} <= 1): gamma(S)^d is "
            "mis-signed there (per-hop drift is inflation, not attenuation); "
            "disable ttfs_gain_correction / ttfs_gain_correction_ramp for this "
            "vehicle"
        )


def cascaded_gain_factors(model, T: int, *, rule: str = "relative", c: float = 1.9) -> dict:
    """Per-perceptron target gain factor ``g_d`` keyed by ``id(perceptron)`` (no mutation).

    Encoding/segment-entry perceptrons are pinned (g=1.0): retuning their scale
    breaks NF↔SCM deployment parity.
    """
    if rule not in _RULES:
        raise ValueError(f"gain-correction rule must be one of {sorted(_RULES)}; got {rule!r}")
    gfn = _RULES[rule]
    S = int(T)
    depths = per_perceptron_cascade_depth(model.get_mapper_repr())
    require_intra_segment_depth(depths)
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
    perceptron (rate 0 → base, rate 1 → full) and re-propagate input scales.
    ``base_scales`` is the parallel list of pre-ramp per-perceptron scales.
    """
    r = float(rate)
    for perceptron, base in zip(model.get_perceptrons(), base_scales):
        g = float(factors.get(id(perceptron), 1.0))
        perceptron.set_activation_scale(float(base) * (g ** r))
    propagate_boundary_input_scales(model, input_data_scale=input_data_scale)


def apply_cascaded_gain_correction(
    model, T: int, *, rule: str = "relative", c: float = 1.9,
    input_data_scale: float = 1.0,
) -> dict:
    """Trim each perceptron's ``activation_scale`` by its per-depth gain factor and
    re-propagate input scales (encoding/entry layers pinned at g=1.0). Mutates the
    model in place; returns a stats dict.
    """
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
