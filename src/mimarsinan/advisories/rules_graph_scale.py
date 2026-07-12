"""Graph advisory rules for scale/grid pathologies (M4 chain shape, M2 WQ grid)."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from mimarsinan.advisories.advisory import (
    SEVERITY_RISK,
    Advisory,
    lossless_mandate_applies,
)
from mimarsinan.advisories.graph_common import exec_and_deps, name_of
from mimarsinan.pipelining.core.platform_constraints_resolver import (
    resolve_wq_two_scale_projection,
)
from mimarsinan.spiking.segment_forward import perceptron_of
from mimarsinan.transformations.perceptron.perceptron_transformer import (
    PerceptronTransformer,
)

ADV_SCALE_SPREAD = "ADV-SCALE-SPREAD"
ADV_NORMFREE_CHAIN = "ADV-NORMFREE-CHAIN"
ADV_BIAS_GRID_DOMINANCE = "ADV-BIAS-GRID-DOMINANCE"

SCALE_SPREAD_LIMIT = 64.0
"""Healthy vehicles keep live-channel spread within one decade (<=10x); the
failing norm-free chain measured 1870x — 64 sits an order above healthy."""

NORMFREE_CHAIN_MIN_HOPS = 6
"""Chain-shape pathology binds at depth: the 9-hop norm-free chain craters
where the 4-hop normalized vehicle survives the same starved gauge."""

SCALE_SPREAD_MIN_CHAIN = 2
"""Spread starves the scalar-theta grid only where it composes: the perceptron
must lie on a norm-free chain of at least two hops."""

BIAS_DOMINANCE_LEVEL_FLOOR = 2.0
"""Fire when the perceptron's LARGEST weight retains fewer than two shared-grid
levels (max|b|/max|w| > q_max/2): the bias-set grid measured 0.13-0.32 median
weight levels with 58-95% of weights rounding to exactly zero."""


def _is_norm_free(perceptron) -> bool:
    return isinstance(perceptron.normalization, nn.Identity)


def normfree_chain_lengths(model_repr) -> dict:
    """id(perceptron) -> longest normalization-free perceptron chain THROUGH it.

    Structural and host nodes carry the chain (they do not re-center channel
    scales); only an attached normalization resets it.
    """
    order, deps = exec_and_deps(model_repr)
    consumers = model_repr.consumer_map()

    def _extend(node, best: int) -> int:
        perceptron = perceptron_of(node)
        if perceptron is None:
            return best
        return best + 1 if _is_norm_free(perceptron) else 0

    up: dict = {}
    for node in order:
        up[node] = _extend(node, max((up[d] for d in deps[node]), default=0))
    down: dict = {}
    for node in reversed(order):
        down[node] = _extend(
            node, max((down[c] for c in consumers.get(id(node), [])), default=0)
        )

    lengths: dict = {}
    for node in order:
        perceptron = perceptron_of(node)
        if perceptron is not None and _is_norm_free(perceptron):
            lengths[id(perceptron)] = up[node] + down[node] - 1
    return lengths


def _live_spread(magnitudes: list) -> float | None:
    live = [float(m) for m in magnitudes if float(m) > 0.0]
    if len(live) < 2:
        return None
    return max(live) / min(live)


def _weight_channel_norms(perceptron) -> list:
    weight = PerceptronTransformer().get_effective_weight(perceptron)
    return weight.reshape(weight.shape[0], -1).norm(dim=1).tolist()


def rule_scale_spread(model_repr, plan: Any, channel_stats) -> list[Advisory]:
    lengths = normfree_chain_lengths(model_repr)
    worst: tuple | None = None
    for perceptron in model_repr.get_perceptrons():
        if lengths.get(id(perceptron), 0) < SCALE_SPREAD_MIN_CHAIN:
            continue
        if channel_stats is not None and id(perceptron) in channel_stats:
            basis = "runtime activation q99"
            spread = _live_spread(list(channel_stats[id(perceptron)]))
        else:
            basis = (
                "per-output-channel effective-weight L2 proxy (runtime "
                "activation stats unavailable at this seam)"
            )
            spread = _live_spread(_weight_channel_norms(perceptron))
        if spread is None or spread <= SCALE_SPREAD_LIMIT:
            continue
        if worst is None or spread > worst[0]:
            worst = (spread, name_of(perceptron), basis)
    if worst is None:
        return []
    spread, name, basis = worst
    return [Advisory(
        id=ADV_SCALE_SPREAD,
        severity=SEVERITY_RISK,
        title=f"Per-channel scale spread {spread:.0f}x on a norm-free chain ({name})",
        detail=(
            f"Hop {name!r} spreads its live per-output-channel scales "
            f"{spread:.0f}x (basis: {basis}) on a normalization-free chain "
            "segment, while calibration commits ONE scalar theta per "
            "perceptron. Measured on the failing norm-free chain: 1870x "
            "live-channel q99 spread left the median channel 2.1-3.6 of 4 "
            "usable levels and 51-80% of positive mass below one grid step, "
            "zeroed by the deployed floor/ceil kernels and compounding "
            "across the chain. Memo: "
            "docs/research/findings/mixer_column_scale_pathology.md."
        ),
        tentative=True,
        mandate_violation=lossless_mandate_applies(plan),
        suggested_levers=(
            "normalization in the model (re-centers channel scales)",
            "scale_migration (exact cross-layer channel-scale migration)",
            "per-channel theta (ttfs_theta_cotrain container)",
        ),
    )]


def rule_normfree_chain(model_repr, plan: Any, channel_stats) -> list[Advisory]:
    lengths = normfree_chain_lengths(model_repr)
    longest = max(lengths.values(), default=0)
    if longest < NORMFREE_CHAIN_MIN_HOPS:
        return []
    return [Advisory(
        id=ADV_NORMFREE_CHAIN,
        severity=SEVERITY_RISK,
        title=f"Normalization-free perceptron chain of {longest} hops",
        detail=(
            f"The graph carries a {longest}-hop consecutive perceptron chain "
            "with no attached normalization: per-channel activation scales "
            "drift (theta x10 along depth) and spread over decades with "
            "nothing to re-center them, starving the scalar-theta grid. "
            "Measured: the 9-hop norm-free chain lost 68pp one-shot at the "
            "floor/ceil S=8 kernel where a 4-hop normalized vehicle with the "
            "same starved gauge lost 0.4pp — the pathology is the chain "
            "shape, not any one architecture. Memo: "
            "docs/research/findings/mixer_column_scale_pathology.md."
        ),
        tentative=True,
        mandate_violation=lossless_mandate_applies(plan),
        suggested_levers=(
            "normalization in the model (per-hop BatchNorm)",
            "scale_migration",
            "per-channel theta (ttfs_theta_cotrain container)",
        ),
    )]


def rule_bias_grid_dominance(model_repr, plan: Any, channel_stats) -> list[Advisory]:
    if not plan.weight_quantization:
        return []
    if resolve_wq_two_scale_projection(plan.config):
        return []
    bits = int(plan.config.get("weight_bits", 8))
    q_max = float(2 ** (bits - 1) - 1)
    ratio_limit = q_max / BIAS_DOMINANCE_LEVEL_FLOOR
    transformer = PerceptronTransformer()
    worst: tuple | None = None
    for perceptron in model_repr.get_perceptrons():
        weight = transformer.get_effective_weight(perceptron)
        bias = transformer.get_effective_bias(perceptron)
        w_max = float(weight.abs().max()) if weight.numel() else 0.0
        b_max = float(bias.abs().max()) if bias.numel() else 0.0
        if w_max <= 0.0 or b_max / w_max <= ratio_limit:
            continue
        ratio = b_max / w_max
        if worst is None or ratio > worst[0]:
            worst = (ratio, name_of(perceptron))
    if worst is None:
        return []
    ratio, name = worst
    return [Advisory(
        id=ADV_BIAS_GRID_DOMINANCE,
        severity=SEVERITY_RISK,
        title=f"Bias-dominated shared weight-quantization grid ({name})",
        detail=(
            f"Perceptron {name!r} has max|effective bias| / max|effective "
            f"weight| = {ratio:.1f} (> q_max/2 = {ratio_limit:.1f} at "
            f"{bits} bits) and the shared per-perceptron grid is scaled by "
            "max(|w|,|b|): the bias row sets the grid and the largest weight "
            "retains under two levels. Measured with a bias-set grid: the "
            "median weight got 0.13-0.32 grid levels, 58-95% of fc weights "
            "rounded to exactly zero, and the genuine cascaded forward "
            "cratered 0.98 -> 0.67 with the bias rounding error T-amplified. "
            "Read here on the current effective parameters (the same "
            "PerceptronTransformer view the projection quantizes); the ratio "
            "typically GROWS through conversion (theta normalization + DFQ "
            "bias corrections), so a clean read does not certify the WQ "
            "entry. Memo: "
            "docs/research/findings/wq_cascade_crater_repair.md (§3-4)."
        ),
        tentative=True,
        mandate_violation=lossless_mandate_applies(plan),
        suggested_levers=(
            "wq_two_scale_projection (bias on its own grid)",
            "platform has_bias (an on-chip bias register)",
        ),
    )]
