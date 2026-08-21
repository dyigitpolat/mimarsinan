"""The ODIN exporter: a packed ``HardCoreMapping`` becomes images, a program, a manifest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from mimarsinan.chip_simulation.soma_axes import (
    PER_EVENT_FIRING,
    SATURATING_UNSIGNED_MEMBRANE,
)
from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin.expansion import RowPairExpansion
from mimarsinan.mapping.export.odin.feasibility import (
    EMISSION_CEILING,
    check_fan_in,
    check_membrane_init,
    check_sign_granularity,
    check_theta_ceiling,
    entry_event_bound,
    propagate_emission_bounds,
)
from mimarsinan.mapping.export.odin.images import OdinCoreImage, build_core_image
from mimarsinan.mapping.export.odin.manifest import build_manifest
from mimarsinan.mapping.export.odin.program import (
    SequencerProgram,
    barrier_stage,
    clear_stage,
    config_stage,
    drain_bound_cycles,
    gate_stage,
    inject_stage,
    plan_injection,
    readout_stage,
    tref_stage,
)
from mimarsinan.mapping.export.odin.layout import masked_state_byte_writes

#: While the memories are written the network must be gated (doc Sec.2.1).
GATE_ACTIVITY_WHILE_PROGRAMMING = 1


class OdinExportError(ValueError):
    """The mapping or the declared law is not something this exporter can emit."""


@dataclass(frozen=True)
class OdinExport:
    """Per-core images, one sequencer program, and the evidence manifest."""

    cores: Tuple[OdinCoreImage, ...]
    program: SequencerProgram
    manifest: Dict[str, Any]


def export_odin(
    mapping: Any,
    *,
    soma_law: SomaLaw,
    weight_bits: int,
    weight_sign_granularity: str,
    effective_max_axons: int,
    membrane_init: int,
) -> OdinExport:
    """Emit the ODIN deployment of ``mapping`` (after ``ChipLatency.calculate()``).

    Consumes the MAPPING, never the ``ChipModel`` envelope: the envelope pads
    heterogeneous cores to the global max and would program real cells from
    phantom rows.
    """
    _require_odin_soma_law(soma_law)
    _require_computed_latencies(mapping)
    sign_expansion = check_sign_granularity(weight_sign_granularity)
    bounds = propagate_emission_bounds(mapping, ceiling=EMISSION_CEILING)

    images: List[OdinCoreImage] = []
    expansions: List[RowPairExpansion] = []
    geometry_rows: List[Dict[str, Any]] = []
    thetas: Dict[str, int] = {}

    for core_index, core in enumerate(mapping.cores):
        theta = check_theta_ceiling(
            core.threshold, membrane_bits=soma_law.membrane_bits,
            core_index=core_index,
        )
        initial = check_membrane_init(
            membrane_init, theta=theta, core_index=core_index)
        used_axons = int(core.axons_per_core) - int(core.available_axons or 0)
        check_fan_in(
            used_axons, effective_max_axons=effective_max_axons,
            core_index=core_index,
        )
        image, expansion = build_core_image(
            core, core_index=core_index, theta=theta, membrane_init=initial,
            weight_bits=weight_bits,
            gate_activity=GATE_ACTIVITY_WHILE_PROGRAMMING,
        )
        images.append(image)
        expansions.append(expansion)
        thetas[str(core_index)] = theta
        geometry_rows.append(_core_geometry(core, core_index, expansion, used_axons))

    stages: List[Dict[str, Any]] = [
        config_stage(
            core_index=image.core_index,
            register_writes=image.register_writes,
            neuron_words=image.neuron_words,
            synapse_words=image.synapse_words,
        )
        for image in images
    ]
    stages.append(gate_stage(on=False))
    stages.extend(_sample_stages(mapping, images, expansions, bounds))

    return OdinExport(
        cores=tuple(images),
        program=SequencerProgram(stages=tuple(stages)),
        manifest=build_manifest(
            soma_law, bounds, geometry_rows, thetas,
            weight_bits=weight_bits,
            weight_sign_granularity=weight_sign_granularity,
            sign_expansion=sign_expansion,
            effective_max_axons=effective_max_axons,
            cores_exported=len(mapping.cores),
        ),
    )


def _sample_stages(
    mapping: Any,
    images: List[OdinCoreImage],
    expansions: List[RowPairExpansion],
    bounds: Dict[Tuple[int, int], int],
) -> List[Dict[str, Any]]:
    """ONE sample, self-contained: it gates the memory writes and ungates the run."""
    stages: List[Dict[str, Any]] = [gate_stage(on=True)]
    for core_index, core in enumerate(mapping.cores):
        stages.append(_clear_stage_for(core, core_index, images[core_index]))
    stages.append(gate_stage(on=False))
    for core_index, core in enumerate(mapping.cores):
        stages.append(_inject_stage_for(core, core_index, expansions[core_index]))
    stages.append(tref_stage(scope="all"))
    stages.append(barrier_stage(cycles=_drain_bound(mapping, expansions, bounds)))
    for core_index, core in enumerate(mapping.cores):
        stages.append(readout_stage(
            core_index=core_index,
            neurons=range(_used_neurons(core)),
        ))
    return stages


def _require_odin_soma_law(soma_law: SomaLaw) -> None:
    if soma_law.firing_granularity != PER_EVENT_FIRING:
        raise OdinExportError(
            f"the ODIN core evaluates its threshold after EVERY synaptic event, "
            f"so it deploys only firing_granularity={PER_EVENT_FIRING!r}; the "
            f"declared law says {soma_law.firing_granularity!r}. Exporting it "
            f"would report a different physics as the deployed number.")
    if soma_law.membrane_arithmetic != SATURATING_UNSIGNED_MEMBRANE:
        raise OdinExportError(
            f"the ODIN membrane is an 8-bit unsigned register that saturates at "
            f"its ceiling and floors at zero, so it deploys only "
            f"membrane_arithmetic={SATURATING_UNSIGNED_MEMBRANE!r}; the declared "
            f"law says {soma_law.membrane_arithmetic!r}. Declare membrane_bits on "
            f"the platform to derive it.")


def _require_computed_latencies(mapping: Any) -> None:
    for core_index, core in enumerate(mapping.cores):
        if getattr(core, "latency", None) is None:
            raise OdinExportError(
                f"core {core_index} carries no latency; run ChipLatency.calculate() "
                f"on the mapping before exporting, exactly as the nevresim path "
                f"does, rather than re-deriving the schedule here")


def _used_neurons(core: Any) -> int:
    return int(core.neurons_per_core) - int(core.available_neurons or 0)


def _core_geometry(
    core: Any, core_index: int, expansion: RowPairExpansion, used_axons: int
) -> Dict[str, Any]:
    return {
        "core_index": core_index,
        "logical_axons": int(core.axons_per_core),
        "logical_neurons": int(core.neurons_per_core),
        "used_axons": used_axons,
        "used_neurons": _used_neurons(core),
        "bias_slots": list(expansion.bias_slots),
        "physical_rows_used": len(expansion.rows),
        "latency": int(core.latency),
    }


def _clear_stage_for(
    core: Any, core_index: int, image: OdinCoreImage
) -> Dict[str, Any]:
    writes = []
    for neuron in range(_used_neurons(core)):
        for write in masked_state_byte_writes(image.neuron_words[neuron]):
            writes.append({
                "neuron": neuron, "byte_addr": write.byte_addr,
                "value": write.value, "mask": write.mask,
            })
    return clear_stage(core_index=core_index, byte_writes=writes)


def _inject_stage_for(
    core: Any, core_index: int, expansion: RowPairExpansion
) -> Dict[str, Any]:
    """Always-on rows are known at export time; the rest the host fills per sample."""
    bias = set(expansion.bias_slots)
    counts = [1 if slot in bias else 0 for slot in range(int(core.axons_per_core))]
    return inject_stage(
        core_index=core_index,
        events=plan_injection(
            counts=counts, emitting_rows=expansion.emitting_rows_for_slot),
        runtime_rows=[
            [slot, list(expansion.emitting_rows_for_slot(slot))]
            for slot in range(int(core.axons_per_core))
            if slot not in bias and expansion.emitting_rows_for_slot(slot)
        ],
    )


def _slot_event_bound(source: Any, bounds: Dict[Tuple[int, int], int]) -> int:
    entry = entry_event_bound(source)
    if entry is not None:
        return entry
    return bounds.get((int(source.core_), int(source.neuron_)), 0)


def _drain_bound(
    mapping: Any,
    expansions: List[RowPairExpansion],
    bounds: Dict[Tuple[int, int], int],
) -> int:
    injected = 0
    emitted = 0
    for core_index, core in enumerate(mapping.cores):
        expansion = expansions[core_index]
        for slot in range(int(core.axons_per_core)):
            rows = expansion.emitting_rows_for_slot(slot)
            injected += len(rows) * _slot_event_bound(core.axon_sources[slot], bounds)
        for neuron in range(_used_neurons(core)):
            emitted += bounds.get((core_index, neuron), 0)
    return drain_bound_cycles(injected_events=injected, emitted_spike_bound=emitted)
