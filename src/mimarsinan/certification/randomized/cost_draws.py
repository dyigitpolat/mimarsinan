"""The code-level cost-only draws: the registry's cost_only set, partitioned, drawn field by field."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY
from mimarsinan.config_schema.registry.value_schema import judge_declaration, nested_schemas, value_errors

#: Cost-only paths whose value changes the deployed program, drawn by the code-level runner.
PROGRAM_CHANGING: Tuple[str, ...] = (
    "allow_coalescing", "allow_neuron_splitting", "allow_scheduling", "max_schedule_passes",
    "cores.count", "cores.max_axons", "cores.max_neurons", "core_value_granularity",
)
#: Flat geometry keys the resolver derives from ``cores``; drawn through the core-type fields only.
DERIVED_GEOMETRY: Tuple[str, ...] = ("max_axons", "max_neurons")
#: Program-changing, but its consumer (G5's executed residency plan) is not on this tree; drawn from CL1.
DEFERRED: Tuple[str, ...] = ("residency_plan", "residency_plan.index", "residency_plan.step")
#: Cost-only paths no program reads: pricing constants, the priced floorplan and the write law.
PRICING_ONLY: Tuple[str, ...] = (
    "activity_factor", "cores.write_granularity", "cores.write_units_in_parallel", "cores_per_tile",
    "inference_batch_K", "platform_physics_overrides", "platform_physics_profile", "tile_grid_cols",
    "tile_grid_rows", "write_overlap", "write_overlap.mode", "write_overlap.shadow_capacity_fraction",
    "write_ports",
)
PRESERVED_BY_CONSTRUCTION = "pricing-only, preserved by construction"


class CostDrawError(ToolchainRefusal, AssertionError):
    """The runner's partition of the cost-only set disagrees with the registry."""


def registry_paths() -> Iterator[Tuple[str, str]]:
    """``(path, decision class)`` of every registry key and of every classified nested field."""
    for key, entry in REGISTRY.items():
        yield key, str(entry.decision_class)
        if entry.value_schema is None:
            continue
        for path, schema in nested_schemas(entry.value_schema, key):
            if schema.decision_class is not None:
                yield path, str(schema.decision_class)


def cost_only_set() -> FrozenSet[str]:
    """The registry's cost_only set (keys and nested fields): the set the theorem box covers."""
    return frozenset(path for path, cls in registry_paths() if cls == COST_ONLY)


def partition() -> Dict[str, Tuple[str, ...]]:
    """The cost-only set split by what the code-level runner does with each path; refuses any drift."""
    parts = {"program_changing": PROGRAM_CHANGING, "derived_geometry": DERIVED_GEOMETRY,
             "deferred_to_cl1": DEFERRED, "pricing_only": PRICING_ONLY}
    listed = [p for paths in parts.values() for p in paths]
    registry = cost_only_set()
    if len(listed) != len(set(listed)) or set(listed) != registry:
        raise CostDrawError(f"partition drift: unlisted {sorted(registry - set(listed))}, not cost_only "
                            f"{sorted(set(listed) - registry)}, duplicated "
                            f"{sorted(p for p in set(listed) if listed.count(p) > 1)}")
    return parts


@dataclass(frozen=True)
class DrawSpace:
    """The declared X3a declaration space (portfolio X3a, Variables); every range is inclusive."""

    core_types: Tuple[int, int] = (1, 3)
    axons: Tuple[int, int] = (4, 64)
    neurons: Tuple[int, int] = (4, 64)
    count: Tuple[int, int] = (1, 32)
    passes: Tuple[int, int] = (1, 8)
    granularities: Tuple[Mapping[str, str], ...] = ({}, {"threshold": "per_core"}, {"threshold": "per_neuron"})


DEFAULT_DRAWS = DrawSpace()


def draw_declaration(seed: int, space: DrawSpace = DEFAULT_DRAWS) -> Dict[str, Any]:
    """The program-changing cost-only fields, each drawn on its own; every value judged by its schema.

    ``cores.has_bias`` is function-bearing: it belongs to the instance, never to a draw
    (:func:`platform_constraints` adds it).
    """
    rng = random.Random(int(seed))
    cores: List[Dict[str, Any]] = []
    for _ in range(rng.randint(*space.core_types)):
        cores.append({"max_axons": rng.randint(*space.axons), "max_neurons": rng.randint(*space.neurons),
                      "count": rng.randint(*space.count)})
    declaration: Dict[str, Any] = {
        "cores": cores,
        "allow_coalescing": rng.random() < 0.5,
        "allow_neuron_splitting": rng.random() < 0.5,
        "allow_scheduling": rng.random() < 0.5,
        "max_schedule_passes": rng.randint(*space.passes),
        "core_value_granularity": dict(rng.choice(space.granularities)),
    }
    for key, value in declaration.items():
        judge_declaration(REGISTRY[key], value, run_domain=None)
    return declaration


def platform_constraints(declaration: Mapping[str, Any], *, hardware_bias: bool) -> Dict[str, Any]:
    """The platform constraints a deployment runs under: the draw plus the instance's bias mode per core."""
    out = dict(declaration)
    out["cores"] = [dict(core, has_bias=bool(hardware_bias)) for core in declaration["cores"]]
    return out


def drawn_paths(declaration: Mapping[str, Any]) -> FrozenSet[str]:
    """The registry paths a drawn declaration sets: whole keys for scalars, the drawn fields of ``cores``."""
    paths = set()
    for key, value in declaration.items():
        if key == "cores":
            paths.update(f"cores.{field}" for core in value for field in core)
        else:
            paths.add(key)
    return frozenset(paths)


def legal(declaration: Mapping[str, Any]) -> List[str]:
    """Every registry violation of a declaration (empty for a legal one)."""
    errors: List[str] = []
    for key, value in declaration.items():
        schema = REGISTRY[key].value_schema
        if schema is not None:
            errors += value_errors(schema, value, key)
    return errors
