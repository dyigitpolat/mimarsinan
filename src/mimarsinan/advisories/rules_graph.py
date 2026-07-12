"""Graph advisory rules over the mapper ModelRepresentation (post Torch Mapping)."""

from __future__ import annotations

from typing import Any

from mimarsinan.advisories.advisory import (
    SEVERITY_RISK,
    Advisory,
    lossless_mandate_applies,
)
from mimarsinan.advisories.graph_common import exec_and_deps, name_of
from mimarsinan.advisories.rules_config import quantized_spiking_deployment
from mimarsinan.advisories.rules_graph_scale import (
    rule_bias_grid_dominance,
    rule_normfree_chain,
    rule_scale_spread,
)
from mimarsinan.chip_simulation.spiking_semantics import is_lif
from mimarsinan.spiking.gain_correction import per_perceptron_cascade_depth
from mimarsinan.spiking.segment_forward import (
    SegmentForwardDriver,
    TtfsSegmentPolicy,
    perceptron_of,
)

ADV_STAIRCASE_DEPTH = "ADV-STAIRCASE-DEPTH"
ADV_FANIN_DEPTH_IMBALANCE = "ADV-FANIN-DEPTH-IMBALANCE"

STAIRCASE_DEPTH_MIN_HOPS = 6
"""Composed staircase RMSE saturates by hop ~5 (sync memo prefix law), matching
the PROVEN_RECOVERY_DEPTH=6 recovery wall (install_resolution/gauges.py)."""

STAIRCASE_DEPTH_MAX_S = 8
"""The 1/S crater law heals by S=16 (-6.91pp at S=4, -1.91 at S=8, -0.68 at
S=16 measured on the lif tier-0 cells)."""


def rule_staircase_depth(model_repr, plan: Any, channel_stats) -> list[Advisory]:
    if not quantized_spiking_deployment(plan):
        return []
    s = int(plan.config.get("simulation_steps", 0) or 0)
    if s <= 0 or s > STAIRCASE_DEPTH_MAX_S:
        return []
    depths = per_perceptron_cascade_depth(model_repr)
    if not depths:
        return []
    hops = max(depths.values()) + 1
    if hops < STAIRCASE_DEPTH_MIN_HOPS:
        return []
    return [Advisory(
        id=ADV_STAIRCASE_DEPTH,
        severity=SEVERITY_RISK,
        title=f"Deep staircase composition at low S (L={hops}, S={s})",
        detail=(
            f"This graph's longest on-chip hop chain (L={hops}) composes one "
            f"S-level staircase quantization per hop at S={s}. The composed "
            "logit error is entry-dominated and saturates by hop ~5 (measured "
            "prefix RMSE 0.61 -> 1.35 by hop 5, flat to hop 10 — the sync "
            "composition law), while per-hop distortion follows the 1/S law "
            "(crater -6.91pp at S=4, -1.91pp at S=8, -0.68pp at S=16, healed "
            "by S>=16 — the lif exactness law). L>=6 with S<=8 sits in the "
            "entry-dominated composition-distortion regime. Memos: "
            "docs/research/findings/sync_deployment_exactness.md, "
            "docs/research/findings/lif_deployment_exactness.md."
        ),
        tentative=True,
        mandate_violation=lossless_mandate_applies(plan),
        suggested_levers=(
            "simulation_steps (raise S)",
            "lif_membrane_readout (exact readout term)",
            "lif_per_hop_retiming (kills the back-loading deficit)",
            "scale_migration (repairs the entry hops' grids)",
            "starvation_aware_scale_quantile",
        ),
    )]


def _unequal_depth_joins(model_repr) -> list:
    """Names of in-segment nodes whose live fan-in branches meet at unequal
    perceptron depths (the mapper-graph predictor of the IR-level gap>1 edges
    that mapping/latency/depth_balancing.py detects and repairs)."""
    _, deps = exec_and_deps(model_repr)
    driver = SegmentForwardDriver(model_repr, 1, TtfsSegmentPolicy())
    joins: list = []
    for seg_nodes in driver.segments.values():
        seg_set = set(seg_nodes)
        depth = driver.policy.segment_depths(driver, seg_nodes)
        for node in seg_nodes:
            contributions = {
                depth[source] + (1 if perceptron_of(source) is not None else 0)
                for source in deps.get(node, [])
                if source in seg_set
            }
            if len(contributions) > 1:
                perceptron = perceptron_of(node)
                label = (
                    name_of(perceptron)
                    if perceptron is not None
                    else type(node).__name__
                )
                joins.append(label)
    return joins


def rule_fanin_depth_imbalance(model_repr, plan: Any, channel_stats) -> list[Advisory]:
    if not is_lif(plan.spiking_mode):
        return []
    if bool(plan.config.get("lif_depth_balancing_relays", False)):
        return []
    joins = _unequal_depth_joins(model_repr)
    if not joins:
        return []
    return [Advisory(
        id=ADV_FANIN_DEPTH_IMBALANCE,
        severity=SEVERITY_RISK,
        title=f"Unequal-depth fan-in join(s) with depth-balancing relays off "
              f"({', '.join(sorted(set(joins)))})",
        detail=(
            "A fan-in join whose branches meet at unequal intra-segment "
            "depths makes the consumer's window both DROP the shallow "
            "branch's early spikes and RE-READ its stale final buffer (V6): "
            "measured 0.102 mean rate error on ~31% of join neurons at T=4. "
            "The exact remedy is depth-balancing relay insertion "
            "(mapping/latency/depth_balancing.py, knob "
            "lif_depth_balancing_relays), which this deployment has "
            "disabled; the mapping-time analysis there is the IR-level twin "
            "of this mapper-graph check. Memo: "
            "docs/research/findings/lif_deployment_exactness.md (V6/C5)."
        ),
        tentative=True,
        mandate_violation=lossless_mandate_applies(plan),
        suggested_levers=(
            "lif_depth_balancing_relays",
        ),
    )]


GRAPH_RULES = (
    rule_staircase_depth,
    rule_scale_spread,
    rule_normfree_chain,
    rule_bias_grid_dominance,
    rule_fanin_depth_imbalance,
)
