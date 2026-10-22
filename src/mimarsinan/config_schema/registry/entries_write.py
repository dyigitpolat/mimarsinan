"""Registry entries: the write-law policy keys, the core-type field schema the write fields extend, the
declarations the builder injects into entries that live in files S0 does not edit (consumer readings
included), and the platform bias rule both doors apply."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, get_args

from mimarsinan.chip_simulation.nevresim.connectivity import (
    ConnectivityMode,
    resolve_nevresim_connectivity_mode,
)
from mimarsinan.config_schema.registry.decision_classes import COST_ONLY, FUNCTION_BEARING
from mimarsinan.config_schema.registry.types import (
    Category,
    ConfigKeySchema as _E,
    FieldType as T,
)
from mimarsinan.config_schema.registry.value_schema import ValueSchema, enum_legal_set
from mimarsinan.mapping.support.period_census import WRITE_GRANULARITIES, WRITE_OVERLAP_MODES
from mimarsinan.tuning.orchestration.temporal_allocation import (
    S_ALLOCATION_EXPLICIT,
    S_ALLOCATION_MODES,
    S_ALLOCATION_SUPPORTED_MODES,
    S_ALLOCATION_UNIFORM,
    TemporalAllocationResolver,
    resolve_s_allocation_mode,
)

_PC = "platform_constraints"
_OWNER = "deployment_record.cost"

#: The fields of one ``cores`` entry (geometry, bias, write); open: the parser flags the rest.
CORE_TYPE_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "max_axons": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                 doc="axons per core: tiling and packing only",
                                 absent_means="undeclared; mapping needs it"),
        "max_neurons": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                   doc="neurons per core: tiling and packing only",
                                   absent_means="undeclared; mapping needs it"),
        "count": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                             doc="cores of this type: packing and passes only",
                             absent_means="one core"),
        "has_bias": ValueSchema(kind=T.BOOL, decision_class=FUNCTION_BEARING,
                                doc="the bias mode changes the weight-quantization grid",
                                absent_means="the platform bias mode: the flat has_bias, else the nested "
                                             "platform_constraints.has_bias, else true (a bias lane)"),
        "write_granularity": ValueSchema(kind=T.ENUM, options=WRITE_GRANULARITIES,
                                         decision_class=COST_ONLY,
                                         doc="the unit one write commits: write time only",
                                         absent_means="undeclared; a term needing it refuses by name"),
        "write_units_in_parallel": ValueSchema(kind=T.INT, low=1, decision_class=COST_ONLY,
                                               doc="units one write commits at once: write time only",
                                               absent_means="1"),
    },
    closed=False,
)


def s_allocation_as_read(config: Mapping[str, Any]) -> str:
    """The allocation mode as its consumer reads it: ``resolve_s_allocation_mode`` (which lowercases), and
    an explicit per-depth map whose consumer allocation is uniform (``TemporalAllocation.is_uniform``: every
    entry equals simulation_steps) is the uniform declaration it states. No door knows the cascade depth, so
    the map is read at its own length; a map the consumer cannot read leaves the explicit mode. The consumer
    casts each entry with int(); the map's own value schema (``INJECTED_DECLARATIONS``) refuses at both doors
    every map that cast would change, so no admitted document is read through it."""
    mode = resolve_s_allocation_mode(config)
    if mode != S_ALLOCATION_EXPLICIT:
        return mode
    try:
        resolver = TemporalAllocationResolver.from_config(config)
        if resolver.explicit is None:
            return mode
        allocation = resolver.resolve(depth=len(resolver.explicit))
    except (ArithmeticError, TypeError, ValueError):
        return mode
    return S_ALLOCATION_UNIFORM if allocation.is_uniform else mode


#: Declared attributes of keys whose entries live in files S0 does not edit, injected by the builder:
#: the core-type schema of ``cores``, the absent meaning of the platform bias mode
#: (``platform_bias_mode``), the absent meaning of the census batch bound
#: (``simulation_factory.resolve_census_batch_size``: the resolved 8 caps the spiking hard-core read and
#: every OOM retry, the other reads take the census batch), and the legal sets taken from their
#: consumers: a value the consumer refuses is no option, a value it names but no run implements is
#: reserved, and both doors refuse either by name (``enum_legal_set``; the per-depth S map takes the list
#: its consumer parses). A consumer that normalizes what it reads gives the key its reading
#: (``consumer_reading``), by reference.
INJECTED_DECLARATIONS: Mapping[str, Mapping[str, Any]] = {
    "cores": {"value_schema": ValueSchema(kind=T.CORES, item=CORE_TYPE_SCHEMA)},
    "has_bias": {"empty_means": "true — every core type declaring no has_bias gets a bias lane; a flat "
                                "config's nested platform_constraints.has_bias, when present, is taken "
                                "instead"},
    "simulation_batch_size": {"empty_means": "8 on the spiking hard-core read and every OOM retry; the "
                                             "census batch max(test batch, 1024) on the other "
                                             "deployed-metric reads"},
    # chip_simulation/nevresim/connectivity.py: compile_time or runtime, read stripped; else it raises.
    "nevresim_connectivity_mode": {**enum_legal_set(get_args(ConnectivityMode)),
                                   "consumer_reading": resolve_nevresim_connectivity_mode},
    # tuning/orchestration/temporal_allocation.py: three modes are named, read lowercased; only uniform
    # is threaded.
    "s_allocation": {**enum_legal_set(S_ALLOCATION_SUPPORTED_MODES, reserved=tuple(
        mode for mode in S_ALLOCATION_MODES if mode not in S_ALLOCATION_SUPPORTED_MODES)),
        "consumer_reading": s_allocation_as_read},
    # temporal_allocation._parse_explicit: a non-empty per-depth list of positive ints, judged as written
    # (the consumer's int() cast reads 32.9 and "32" as 32).
    "s_allocation_explicit": {"value_schema": ValueSchema(
        kind=T.INT_LIST, item=ValueSchema(kind=T.INT, low=1), min_length=1)},
}


def platform_bias_declarations(config: Mapping[str, Any]) -> Tuple[Any, Any]:
    """The platform bias mode as declared: the flat ``has_bias`` and the nested
    ``platform_constraints.has_bias`` a flat config may carry (None where absent)."""
    nested = config.get("platform_constraints")
    return config.get("has_bias"), (nested.get("has_bias") if isinstance(nested, Mapping) else None)


def platform_bias_mode(config: Mapping[str, Any]) -> Any:
    """The bias mode of every core type that declares no ``has_bias`` of its own: the flat ``has_bias``
    first, then the nested ``platform_constraints.has_bias``, else true (a bias lane)."""
    flat, nested = platform_bias_declarations(config)
    return flat if flat is not None else nested if nested is not None else True


def conflicting_platform_bias_error(config: Mapping[str, Any]) -> Optional[str]:
    """Why the platform bias mode is ambiguous, or None (one rule for both doors): a flat and a nested
    boolean that disagree. A non-boolean is the BOOL schema's to refuse, so it is left to that judge."""
    flat, nested = platform_bias_declarations(config)
    if not (isinstance(flat, bool) and isinstance(nested, bool)) or flat == nested:
        return None
    return (f"has_bias={flat!r} and platform_constraints.has_bias={nested!r} declare the platform bias "
            f"mode twice and disagree: declare it once")


WRITE_OVERLAP_SCHEMA = ValueSchema(
    kind=T.JSON,
    fields={
        "mode": ValueSchema(kind=T.ENUM, options=WRITE_OVERLAP_MODES, decision_class=COST_ONLY,
                            doc="writes exposed or hidden behind compute: write time only"),
        "shadow_capacity_fraction": ValueSchema(
            kind=T.FLOAT, low=0.0, high=1.0, high_exclusive=True, decision_class=COST_ONLY,
            doc="array share lost to shadow buffering: write time only", absent_means="0"),
    },
    required=("mode",),
)

ENTRIES = (
    _E("inference_batch_K", section=_PC, group="mapping_strategy", owner=_OWNER, type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Inference Batch K",
       effect="Inferences served per weight-programming period (write terms at batch K)",
       doc="Inferences each programmed period serves: backing-store writes are amortized over "
           "K, producer writes are charged per inference. Absent: no batch is declared and no "
           "write-at-K term is emitted.",
       bounds=(1, None), passthrough=True,
       empty_means="no batch declared — no write-at-K term is emitted"),
    _E("write_overlap", section=_PC, group="hardware", owner=_OWNER, type=T.JSON,
       category=Category.ADVANCED, exposure="user", label="Write Overlap",
       effect="Whether array writes are exposed after compute or hidden behind it",
       doc="The write-compute overlap discipline: serial exposes every write, overlapped hides "
           "writes behind the previous pass's compute after the declared shadow-capacity loss.",
       value_schema=WRITE_OVERLAP_SCHEMA, passthrough=True,
       empty_means="serial, shadow capacity 0"),
    _E("write_ports", section=_PC, group="hardware", owner=_OWNER, type=T.INT,
       category=Category.ADVANCED, exposure="user", label="Write Ports",
       effect="Concurrent write ports of the chip",
       doc="Write ports that commit to the arrays concurrently; the exposed write time divides "
           "by them.",
       bounds=(1, None), passthrough=True, empty_means="1 — one write port"),
    _E("inference_rate_hz", section=_PC, group="mapping_strategy", owner=_OWNER, type=T.FLOAT,
       category=Category.ADVANCED, exposure="user", label="Inference Rate", unit="Hz",
       effect="Converts the write-endurance lifetime from inferences to hours",
       doc="The declared inference rate: the lifetime the endurance rows price in inferences "
           "is also reported in hours.",
       bounds=(0.0, None), exclusive_bounds=(True, False), passthrough=True,
       empty_means="lifetime reported in inferences only"),
    _E("min_lifetime_hours", section=_PC, group="hardware", owner=_OWNER, type=T.FLOAT,
       category=Category.ADVANCED, exposure="user", label="Minimum Lifetime", unit="h",
       effect="A program whose write-endurance lifetime falls below this is inadmissible",
       doc="The lifetime floor: a program whose priced endurance lifetime is shorter is "
           "inadmissible by name, with every priced term still reported.",
       bounds=(0.0, None), exclusive_bounds=(True, False), passthrough=True,
       empty_means="no lifetime floor"),
)
