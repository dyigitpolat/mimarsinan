"""The catalog: the legacy 8 re-registered byte-equal, plus the record's axes (§8).

Order is contract. The first eight rows are ``search.results.ALL_OBJECTIVES``
verbatim — same keys, same directions, same ORDER — because every optimizer's
objective vector is indexed by that order; ``test_objectives_registry`` pins it
against literal data. The rows after them are the measured/modeled axes the
sealed record finally makes addressable (schema doc §8).

Each row states ONE backing datum, and availability follows from it. The static
layout statistics also carry ``schedule_pass_count``, but the pass-count axis
here is keyed to the SEALED schedule census: lifting a static pass axis into the
search catalog would change every optimizer's objective set, which is the
search surface's decision (W5.1), not this registry's.
"""

from __future__ import annotations

from typing import Tuple

from mimarsinan.deployment_record.objectives.extractors import (
    Backing,
    cost_term,
    priced_term,
    deployed_accuracy_backing,
    energy_field,
    host_op_wall_backing,
    latency_steps_backing,
    layout_field,
    noc_field,
    quantity_field,
    reprogramming_bytes_backing,
    schedule_field,
    sync_barrier_backing,
    view_field,
)
from mimarsinan.deployment_record.objectives.registry import ObjectiveRegistry
from mimarsinan.deployment_record.objectives.spec import (
    Direction,
    ObjectiveProvenance,
    ObjectiveSpecV2,
)

ACCURACY_OBJECTIVE_KEY = "estimated_accuracy"


def objective(
    key: str,
    direction: Direction,
    unit: str,
    provenance: ObjectiveProvenance,
    backing: Backing,
    doc: str,
) -> ObjectiveSpecV2:
    """One axis over one backing datum: availability and value are the same reader."""
    return ObjectiveSpecV2(
        key=key,
        direction=direction,
        unit=unit,
        provenance=provenance,
        requires=backing.requires,
        availability=backing.available,
        extractor=backing.extract,
        doc=doc,
    )


_LEGACY_STATIC_AXES = (
    objective(
        ACCURACY_OBJECTIVE_KEY, "max", "fraction", "training_proxy",
        view_field(
            "estimated_accuracy",
            "a candidate accuracy estimate (the search-side training proxy)",
        ),
        "Search-time accuracy estimate of a candidate architecture.",
    ),
    objective(
        "total_params", "min", "parameters", "static",
        view_field(
            "total_params", "the candidate model's parameter census"
        ),
        "Trainable parameter count of the candidate model.",
    ),
    objective(
        "total_param_capacity", "min", "cells", "static",
        view_field(
            "chip_param_capacity",
            "declared core capacity (max_axons x max_neurons x count)",
        ),
        "Declared crossbar cell capacity of the platform being searched.",
    ),
    objective(
        "total_sync_barriers", "min", "barriers", "static",
        sync_barrier_backing(),
        "Host-side segment slots plus the schedule's inter-pass syncs.",
    ),
    objective(
        "param_utilization_pct", "max", "percent", "static",
        layout_field("mapped_params_pct"),
        "Share of allocated crossbar cells carrying mapped parameters.",
    ),
    objective(
        "neuron_wastage_pct", "min", "percent", "static",
        layout_field("total_wasted_neurons_pct"),
        "Share of allocated neuron rows left unused by the packing.",
    ),
    objective(
        "axon_wastage_pct", "min", "percent", "static",
        layout_field("total_wasted_axons_pct"),
        "Share of allocated axon columns left unused by the packing.",
    ),
    objective(
        "fragmentation_pct", "min", "percent", "static",
        layout_field("fragmentation_pct"),
        "Unused-but-allocated area as a share of the allocated area.",
    ),
)

_RECORD_AXES = (
    objective(
        "deployed_accuracy", "max", "fraction", "measured",
        deployed_accuracy_backing(),
        "Accuracy actually read off the deployed program.",
    ),
    objective(
        "mj_per_sample", "min", "mJ/sample", "measured",
        energy_field("mj_per_sample"),
        "SANA-FE measured energy per inference sample.",
    ),
    objective(
        "latency_steps", "min", "timesteps", "measured",
        latency_steps_backing(),
        "Timesteps executed across the program's neural segments.",
    ),
    objective(
        "host_op_wall_s", "min", "s", "measured",
        host_op_wall_backing(),
        "Measured wall time spent in host ComputeOp stages.",
    ),
    objective(
        "total_spikes", "min", "spikes", "measured",
        energy_field("total_spikes"),
        "Total spikes emitted by the deployed program.",
    ),
    objective(
        "pass_count", "min", "passes", "static",
        schedule_field("pass_count"),
        "Scheduled passes over the program's neural segments.",
    ),
    objective(
        "reprogram_passes", "min", "passes", "static",
        schedule_field("reprogram_passes"),
        "Passes that reprogram core-resident weights instead of reusing them.",
    ),
    objective(
        "reprogramming_bytes", "min", "bytes", "static",
        reprogramming_bytes_backing(),
        "Weight payload bytes moved by the reprogramming passes.",
    ),
    objective(
        "params_reloaded", "min", "parameters", "static",
        schedule_field("params_reloaded"),
        "Parameters reloaded across the schedule's passes.",
    ),
    objective(
        "noc_inter_tile_packets", "min", "packets", "measured",
        noc_field("inter_tile_packets"),
        "SANA-FE packets crossing a tile boundary.",
    ),
    objective(
        "noc_total_packets", "min", "packets", "measured",
        noc_field("total_packets"),
        "SANA-FE packets injected into the NoC in total.",
    ),
    objective(
        "programming_energy_mj", "min", "mJ", "modeled",
        cost_term("energy", "modeled_programming_mj"),
        "Modeled energy of moving the programming payload (banded).",
    ),
    objective(
        "sync_barrier_energy_mj", "min", "mJ", "modeled",
        cost_term("energy", "modeled_sync_mj"),
        "Modeled energy of the schedule's sync barriers (banded).",
    ),
    objective(
        "throughput_samples_per_s", "max", "samples/s", "modeled",
        cost_term("throughput", "samples_per_s"),
        "Steady-state samples per second from the cost model's latency total.",
    ),
)


#: The vendor-priced axes (C2): the chip-designer metrics, available exactly when the
#: run declares a physics profile whose constants can back them. They register LAST —
#: catalog order is contract, so every optimizer's existing vector prefix survives.
_PHYSICS_AXES: Tuple[ObjectiveSpecV2, ...] = (
    objective(
        "chip_area_mm2", "min", "mm^2", "modeled",
        priced_term("area", "chip_area_mm2"),
        "Silicon area of the declared chip, priced from the target's physics.",
    ),
    objective(
        "energy_per_inference_mj", "min", "mJ", "modeled",
        priced_term("energy", "energy_per_inference_mj"),
        "Energy of one inference, host side included, priced from the target's "
        "physics.",
    ),
    objective(
        "e2e_latency_s", "min", "s", "modeled",
        priced_term("latency", "e2e_latency_s"),
        "Steady-state end-to-end latency of one inference in seconds.",
    ),
    objective(
        "throughput_inferences_s", "max", "inferences/s", "modeled",
        priced_term("throughput", "throughput_inferences_s"),
        "Steady-state inferences per second (the inverse of e2e latency).",
    ),
)


#: [N3] The traffic axis at both completenesses: derived from the sealed NoC
#: census on a record, and from the wireload model (wire census x declared
#: activity on the resolved floorplan) on a candidate. No physics needed —
#: hops are a count. Registered after the physics axes: catalog order is
#: contract.
_TRAFFIC_AXES: Tuple[ObjectiveSpecV2, ...] = (
    objective(
        "noc_total_hops", "min", "hops", "modeled",
        quantity_field(
            "noc_total_hops",
            requires="the NoC hop census (a sealed SANA-FE traffic record, or "
                     "candidate NoC fragments + declared activity_factor)",
        ),
        "XY-mesh hops of all NoC messages (the sum of per-link packet loads).",
    ),
)

_CARRY_REQUIRES = (
    "the sealed schedule's pass-carry census (a program that cuts a neural "
    "segment into passes)"
)

#: [B] The pass-buffer axes: the required buffer of a PARTICULAR program is a
#: mapping performance metric of the sealed schedule — record-only by
#: construction (a candidate has no pass structure to size), never searched.
_BUFFER_AXES: Tuple[ObjectiveSpecV2, ...] = (
    objective(
        "carry_peak_live_bytes", "min", "bytes", "measured",
        quantity_field("carry_peak_live_bytes", requires=_CARRY_REQUIRES),
        "Worst pass-boundary live raster bytes the host must buffer at once.",
    ),
    objective(
        "carried_raster_bytes", "min", "bytes", "measured",
        quantity_field("carried_raster_bytes", requires=_CARRY_REQUIRES),
        "Raster bytes carried across the schedule's pass boundaries per "
        "inference.",
    ),
)


def build_catalog() -> ObjectiveRegistry:
    """The program's objective catalog, in contract order."""
    registry = ObjectiveRegistry()
    for spec in (
        _LEGACY_STATIC_AXES + _RECORD_AXES + _PHYSICS_AXES + _TRAFFIC_AXES
        + _BUFFER_AXES
    ):
        registry.register(spec)
    return registry


OBJECTIVES = build_catalog()
