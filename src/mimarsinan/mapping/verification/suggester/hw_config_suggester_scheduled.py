from __future__ import annotations
from typing import Sequence
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec
from mimarsinan.mapping.support.schedule.schedule_partitioner import estimate_passes_for_layout
from mimarsinan.mapping.verification.suggester.hw_suggestion_helpers import _dimension_bounds
from mimarsinan.mapping.verification.suggester.hw_suggestion_types import HardwareSuggestion
from mimarsinan.mapping.verification.suggester.hw_config_suggester import suggest_hardware_config
from mimarsinan.mapping.verification.verifier import verify_soft_core_mapping

def suggest_hardware_config_scheduled(
    softcores: Sequence[LayoutSoftCoreSpec],
    *,
    max_passes: int = 8,
    latency_weight: float = 1.0,
    allow_coalescing: bool = False,
    hardware_bias: bool = True,
    axon_granularity: int = 1,
    neuron_granularity: int = 1,
    safety_margin: float = 0.15,
    allow_neuron_splitting: bool = False,
) -> HardwareSuggestion:
    """Suggest hardware config minimizing core area × pass count (scheduled mapping)."""
    if not softcores:
        return HardwareSuggestion(
            core_types=[], total_cores=0,
            rationale="No softcores — nothing to map.",
        )

    max_passes = max(1, int(max_passes))
    softcores_list = list(softcores)

    common_kwargs = dict(
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
        axon_granularity=axon_granularity,
        neuron_granularity=neuron_granularity,
        safety_margin=safety_margin,
        allow_neuron_splitting=allow_neuron_splitting,
    )

    single = suggest_hardware_config(softcores_list, **common_kwargs)

    def _core_area(suggestion: HardwareSuggestion) -> float:
        return sum(
            ct["max_axons"] * ct["max_neurons"] * ct["count"]
            for ct in suggestion.core_types
        )

    best = single
    best_cost = _core_area(single) * (1.0 ** latency_weight)
    best_passes = 1

    if single.core_types:
        ref_ax = max(ct["max_axons"] for ct in single.core_types)
        ref_neu = max(ct["max_neurons"] for ct in single.core_types)
    else:
        max_ax, max_neu, _, _ = _dimension_bounds(softcores_list)
        ref_ax, ref_neu = max_ax or 256, max_neu or 256

    seen_pass_counts: set[int] = {1}
    budgets_to_try = set()
    for divisor in range(2, max_passes + 1):
        budgets_to_try.add(max(1, single.total_cores // divisor))
    for b in [1, 2, 4, 8, 16, 32]:
        if b < single.total_cores:
            budgets_to_try.add(b)

    for budget in sorted(budgets_to_try, reverse=True):
        est_passes, pass_lists = estimate_passes_for_layout(
            softcores_list, budget,
            max_hw_axons=ref_ax, max_hw_neurons=ref_neu,
            allow_coalescing=allow_coalescing, allow_splitting=allow_neuron_splitting,
        )
        if est_passes <= 1 or est_passes in seen_pass_counts or est_passes > max_passes:
            continue
        seen_pass_counts.add(est_passes)

        largest_pass = max(pass_lists, key=len)
        try:
            suggestion = suggest_hardware_config(largest_pass, **common_kwargs)
        except Exception:
            continue
        if not suggestion.core_types:
            continue

        cost = _core_area(suggestion) * (est_passes ** latency_weight)
        if cost < best_cost:
            best = suggestion
            best_cost = cost
            best_passes = est_passes

    rationale_parts = [best.rationale.rstrip(".")]
    if best_passes > 1:
        rationale_parts.append(f"{best_passes} schedule passes (cores reused)")
        rationale_parts.append(f"latency ~{best_passes}x single-pass")
    best.rationale = ". ".join(rationale_parts) + "."
    best.num_passes = best_passes
    best.estimated_latency_multiplier = float(best_passes)
    return best


def suggest_hardware_config_for_model(
    model_repr,
    *,
    max_axons: int,
    max_neurons: int,
    allow_coalescing: bool = False,
    hardware_bias: bool = True,
    axon_granularity: int = 1,
    neuron_granularity: int = 1,
    safety_margin: float = 0.15,
    allow_neuron_splitting: bool = False,
) -> HardwareSuggestion:
    """Convenience wrapper: run layout mapping then suggest hardware config."""
    result = verify_soft_core_mapping(
        model_repr,
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
    )

    if not result.feasible:
        return HardwareSuggestion(
            core_types=[],
            total_cores=0,
            rationale=f"Layout mapping failed: {result.error}",
        )

    return suggest_hardware_config(
        result.softcores,
        allow_coalescing=allow_coalescing,
        hardware_bias=hardware_bias,
        axon_granularity=axon_granularity,
        neuron_granularity=neuron_granularity,
        safety_margin=safety_margin,
        allow_neuron_splitting=allow_neuron_splitting,
    )
