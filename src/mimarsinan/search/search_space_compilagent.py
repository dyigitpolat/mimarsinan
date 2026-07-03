"""Compilagent lever rendering for SearchSpaceDescription."""

from __future__ import annotations

from typing import Any, List, Tuple

from compilagent import (
    DerivationEvidence,
    EnumChoice,
    IntFreeform,
    Lever,
)

from .search_space_description import CORE_DIM_GRANULARITY, SearchSpaceDescription


def to_compilagent_levers(
    description: SearchSpaceDescription,
    *,
    workload_id: str,
    backend_id: str,
) -> Tuple[Any, ...]:
    """Render the search space as a tuple of compilagent ``Lever``s."""
    levers: List[Any] = []

    if description.searches_model:
        for key, values in description.arch_options:
            lever = Lever(
                id=f"arch.{key}",
                target_kind="arch",
                target_selector=key,
                range=EnumChoice(candidates=tuple(str(v) for v in values)),
                default=values[len(values) // 2] if values else None,
                description=(
                    f"Architecture choice `{key}`: enum over the values "
                    f"declared by the model builder's NAS schema."
                ),
                evidence=DerivationEvidence(
                    rule="mimarsinan.arch_options",
                    signal="JointArchHwProblem.arch_options",
                    citations=("search/search_space_description.py",),
                ),
                backend_id=backend_id,
            )
            levers.append(lever)

    if description.searches_hw:
        hw_bound_map = {
            "max_axons": SearchSpaceDescription.COMPILAGENT_AXON_BOUNDS,
            "max_neurons": SearchSpaceDescription.COMPILAGENT_NEURON_BOUNDS,
            "count": SearchSpaceDescription.COMPILAGENT_COUNT_BOUNDS,
        }
        for core_idx in range(description.num_core_types):
            for dim_name, _bounds_attr in SearchSpaceDescription.HW_DIM_KINDS:
                lo, hi = hw_bound_map[dim_name]
                step = (
                    CORE_DIM_GRANULARITY
                    if dim_name in ("max_axons", "max_neurons")
                    else 1
                )
                units = "" if dim_name == "count" else "wires"
                default = max(lo, min(hi, 128 if dim_name != "count" else 16))
                if step > 1:
                    default = max(step, (default // step) * step)
                lever = Lever(
                    id=f"hw.core.{core_idx}.{dim_name}",
                    target_kind="hw.core",
                    target_selector=f"{core_idx}.{dim_name}",
                    range=IntFreeform(min=int(lo), max=int(hi), step=int(step), units=units),
                    default=int(default),
                    description=(
                        f"Hardware core type {core_idx} `{dim_name}` — "
                        f"clamped to [{lo}, {hi}] and "
                        + (
                            f"snapped to multiples of {CORE_DIM_GRANULARITY}."
                            if dim_name != "count"
                            else "passed straight through."
                        )
                    ),
                    evidence=DerivationEvidence(
                        rule="mimarsinan.compilagent.open_range",
                        signal=f"compilagent_{dim_name}_bounds",
                        citations=("search/search_space_description.py",),
                    ),
                    backend_id=backend_id,
                )
                levers.append(lever)

    return tuple(levers)


def derive_int_candidates(lo: int, hi: int, dim_name: str) -> Tuple[int, ...]:
    """Generate a derived sample of candidate values for one HW dimension."""
    if hi <= lo:
        return (int(lo),)
    if dim_name in ("max_axons", "max_neurons"):
        step = CORE_DIM_GRANULARITY
        target_count = 8
        span = hi - lo
        stride = max(step, (span // target_count // step) * step or step)
        values: List[int] = []
        v = lo
        while v <= hi:
            values.append(int(v))
            v += stride
        if values[-1] != hi:
            values.append(int(hi))
        return tuple(sorted(set(values)))
    target_count = 6
    if hi - lo + 1 <= target_count:
        return tuple(range(int(lo), int(hi) + 1))
    stride = max(1, (hi - lo) // (target_count - 1))
    values = []
    v = lo
    while v <= hi:
        values.append(int(v))
        v += stride
    if values[-1] != hi:
        values.append(int(hi))
    return tuple(sorted(set(values)))
