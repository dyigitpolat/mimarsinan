"""Registry decision classes: what changing a key changes (outputs, admissibility or only cost), in one table.

The rule (owner_review.md, Class Rule): a key takes the class of what its consumers on this tree do with
it, for a fixed program (a search mode picks the program by cost and floors). A measurement-protocol key
takes the class of the strongest decision that reads the measured number: a tuner's target, accept or
rollback (function), a refusing gate (admissibility), a priced cost (cost), a report or log (none). A
key no run reads is not_a_decision; a Wave 1 key carries the class of its key contract (plan section 4).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterable, Mapping, Set, Tuple

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry.value_schema import ValueSchema, nested_schemas

FUNCTION_BEARING = "function_bearing"
ADMISSIBILITY_BEARING = "admissibility_bearing"
COST_ONLY = "cost_only"
NOT_A_DECISION = "not_a_decision"
#: A key that sets other keys; its class for a value is the join over the keys that value sets.
COMPOSITE = "composite"
DECISION_CLASSES = (COST_ONLY, FUNCTION_BEARING, ADMISSIBILITY_BEARING, NOT_A_DECISION, COMPOSITE)
#: Join order of the leaf classes: a set of keys is as decisive as its most decisive member.
LEAF_JOIN_ORDER = (NOT_A_DECISION, COST_ONLY, ADMISSIBILITY_BEARING, FUNCTION_BEARING)

_FB, _AB, _CO, _ND = FUNCTION_BEARING, ADMISSIBILITY_BEARING, COST_ONLY, NOT_A_DECISION
_GEOMETRY = (_CO, "core geometry: changes tiling and packing, never outputs")
_PRICING = (_CO, "a constant or assumption the cost model prices with; nothing else reads it")
_FLOORPLAN = (_CO, "NoC floorplan of the priced chip; traffic cost only")
_PRUNING = (_FB, "removes weights, so the computed function changes")
_RESERVED_S = (_ND, "reserved: no run threads the per-depth S map, so every depth runs simulation_steps")
_GATE = (_AB, "a verification threshold: a deployment outside it is refused")
_SAMPLES_0 = (_AB, "sample count of a refusing certificate gate: fewer samples admit more, 0 disarms it")
_METRIC_SAMPLE = (_AB, "chooses the test samples of the deployed-metric reads the retention gate "
                       "judges; no tuner runs after them")

GROUP_CLASSES: Mapping[str, Tuple[str, str]] = {
    "workload": (_FB, "selects the inputs the network trains and computes on (data, scale, encoding, "
                      "augmentation order)"),
    "model": (_FB, "selects the network architecture"),
    "spiking": (_FB, "selects the core semantics and the neuron dynamics"),
    "hardware": (_CO, "declares the chip the program is laid onto (every key classified below)"),
    "mapping_strategy": (_CO, "chooses how the program is laid onto the chip (every key classified below)"),
    "co_search": (_FB, "the search winner sets architecture and chip keys, target_tq included"),
    "conversion": (_FB, "sets the quantization the deployed weights and activations take"),
    "tuning": (_FB, "steers the adaptation that sets the deployed weights and thresholds"),
    "training": (_FB, "sets the trained weights"),
    "deployment_target": (_ND, "a simulator transport, a trace, a wall-clock bound or a flag-only "
                               "threshold; outputs, cost and admission unchanged"),
    "run": (_ND, "run identity, location, resume point, visualization or a resolved runtime fact; "
                 "outputs, cost and admission unchanged"),
}

KEY_CLASSES: Mapping[str, Tuple[str, str]] = {
    # workload, run, spiking, tuning, deployment_target: rows whose class differs from the group's.
    "datasets_path": (_ND, "where the datasets are read from"),
    "seed": (_FB, "seeds initialization and training, so the weights"),
    "pipeline_mode": (_FB, "phased arms the quantization chain; vanilla keeps float weights"),
    "target_metric_override": (_FB, "seeds the metric the tuners' accept and rollback rules read"),
    "tuner_target_floor_ratio": (_ND, "read by no run: the tuner floor derives from degradation_tolerance"),
    "spike_count_parity_samples": _SAMPLES_0,
    "nf_scm_parity_samples": _SAMPLES_0,
    "scm_torch_sim_parity_samples": _SAMPLES_0,
    "value_parity_samples": _SAMPLES_0,
    "enable_nevresim_simulation": (_AB, "arms the nevresim window-count refusal on an integer chip"),
    "enable_loihi_simulation": (_AB, "arms the Loihi spike-parity refusal"),
    "enable_sanafe_simulation": (_AB, "arms the SANA-FE spike-parity refusal"),
    "sanafe_sample_count": (_AB, "sample count of the SANA-FE parity refusal: fewer samples admit more"),
    "event_census_samples": (_CO, "sample count of the event census whose mean prices the record's spike-dependent "
                                  "energy; outputs and admission unchanged"),
    "loihi_parity_sample_index": (_AB, "selects the sample the Loihi parity refusal reads"),
    "max_simulation_samples": (_AB, "subsamples the nevresim refusal's samples, and the deployed "
                                    "metric's when full eval is off"),
    # PipelineStep._capped_test: every step metric, which Pipeline._record_step_metric makes the target.
    "eval_max_samples": (_FB, "selects the test samples of every step metric: the tuners' target and "
                              "the retention gate read it"),
    "deployment_metric_full_eval": _METRIC_SAMPLE,
    "simulation_batch_count": _METRIC_SAMPLE,
    "simulation_batch_size": (_AB, "with simulation_batch_count, sets how many samples a deployed-metric "
                                   "read takes; the batch of every OOM retry"),
    "scm_degradation_tolerance": _GATE,
    "nf_scm_parity_atol": _GATE,
    "nf_scm_parity_max_mismatch_fraction": _GATE,
    "nf_scm_parity_min_agreement": _GATE,
    "scm_torch_sim_parity_check": (_AB, "arms the torch-versus-simulator agreement gate"),
    "scm_torch_sim_parity_min_agreement": _GATE,
    "onchip_min_fraction": (_AB, "a deployment below this on-chip fraction is refused"),
    "onchip_majority_gate": (_AB, "arms the on-chip floor refusal (false is the intentional-offload escape)"),
    "capacity_gate": (_AB, "refuses a program whose peak footprint exceeds the chip"),
    # The architecture the SANA-FE step simulates, then refuses on a spike-parity divergence of.
    "sanafe_arch_preset": (_AB, "selects the architecture the SANA-FE parity refusal simulates (custom: "
                                "the YAML's own neuron plugins), with its event costs and floorplan"),
    "sanafe_custom_arch_path": (_AB, "the architecture file the SANA-FE parity refusal simulates and "
                                     "measures cost on"),
    # hardware: every key. Wave 1 keys (readout_law, write_*, min_lifetime_hours, target*) carry the class
    # of their key contract; their consumer groups' acceptance tests pin it.
    "hw_config_mode": (_FB, "search mode lets the co-search set chip keys, target_tq included"),
    "activation_bits": (_FB, "the value grid at every host-chip boundary"),
    "core_value_granularity": (_CO, "which per-core quantities constrain co-residency; packing honours it"),
    "cores_per_tile": _FLOORPLAN,
    "tile_grid_rows": _FLOORPLAN,
    "tile_grid_cols": _FLOORPLAN,
    "pass_buffer_capacity_bytes": (_AB, "a schedule whose carried rasters exceed the buffer is refused"),
    "cores": (_FB, "join of its fields: has_bias and operand are function-bearing, geometry, weight-sign and write "
                   "fields cost-only"),
    "max_axons": _GEOMETRY,
    "max_neurons": _GEOMETRY,
    "has_bias": (_FB, "the bias mode changes the weight-quantization grid; a core type without its own "
                      "takes the platform's (the flat key, then the nested one)"),
    "target_tq": (_FB, "the activation quantization levels"),
    "simulation_steps": (_FB, "the spike window S"),
    "weight_bits": (_FB, "the weight grid"),
    "allow_coalescing": (_CO, "lets soft cores share hard cores; outputs unchanged"),
    "allow_neuron_splitting": (_CO, "splits layers across cores; outputs unchanged"),
    "allow_per_layer_s": (_ND, "reserved gate: no mapping decision or validator reads it"),
    "platform_physics_profile": _PRICING,
    "platform_physics_overrides": _PRICING,
    "activity_factor": _PRICING,
    "target": (COMPOSITE, "sets the keys of a target bundle"),
    "target_overrides": (COMPOSITE, "replaces bundle keys one by one"),
    "target_claim_class": (_ND, "a bundle label; nothing decides on it"),
    "target_faithfulness_class": (_ND, "a bundle label; nothing decides on it"),
    "readout_law": (_FB, "join of its fields: the ADC and requantization law sets the integer outputs"),
    "readout_ranges": (_FB, "per consumer group gives each group of a perceptron's columns its own output grid"),
    "quantizer_state": (_FB, "the declared readout ranges and entry grids move every grid they name"),
    "weight_scale_rule": (_FB, "a fractional scale moves the weight codes and the law's multiplier"),
    "bias_resolution": (_FB, "a bias held at the law's resolution moves the law's B and the float bias"),
    "onchip_arithmetic": (_FB, "join of its fields: lanes move rounding (one per lane) and host-op placement"),
    "layer_core_types": (_FB, "each layer's core type sets its weight grid and the cores it packs onto"),
    "write_overlap": (_CO, "join of its fields: exposed write time only"),
    "write_ports": (_CO, "write time only"),
    "min_lifetime_hours": (_AB, "a program below the lifetime floor is inadmissible by name"),
    # mapping_strategy: every key.
    "encoding_layer_placement": (_FB, "moves the encoder between host and chip; transparency not yet proven"),
    "negative_value_shift": (_FB, "a calibrated shift or host subsume-forward at negative boundaries"),
    "lif_per_hop_retiming": (_FB, "re-times every hop boundary"),
    "lif_depth_balancing_relays": (_FB, "changes the spike arrival of unequal-depth fan-in"),
    "scale_migration": (_FB, "moves per-channel scales onto the quantization grids"),
    "scale_migration_clip_ratio": (_FB, "clips the migrated per-channel scales"),
    "s_allocation": _RESERVED_S,
    "s_allocation_explicit": _RESERVED_S,
    "s_allocation_budget": _RESERVED_S,
    "pruning": _PRUNING,
    "pruning_fraction": _PRUNING,
    "prune_sparsity": _PRUNING,
    "prune_criterion": _PRUNING,
    "prune_group_size": _PRUNING,
    "elimination_propagation": (_FB, "how far pruning seeds propagate; exactness not yet in the theorem box"),
    "elimination_constant_folding": (_FB, "folds constant rows into bias carriers; exactness not yet proven"),
    "max_schedule_passes": (_CO, "pass budget of the scheduled build; outputs unchanged by pass transparency"),
    "scheduling_latency_weight": (_ND, "read by no run: it weights the wizard's scheduled hardware "
                                       "suggestion"),
    "allow_scheduling": (_CO, "time-multiplexes core passes; outputs unchanged by pass transparency"),
    # Owner's S0 review: the strongest consumer rules. On spiking cores the moved projection rides the
    # activation grid; on value (MVM) cores the decomposition is exact (a theorem-box finding).
    "decompose_leaf_modules": (_FB, "moves leaf projections onto cores: on spiking cores the projection "
                                    "rides the activation grid (exact on value cores, a theorem-box finding)"),
    "inference_batch_K": (_CO, "amortizes writes over K inferences; priced terms only"),
    "inference_rate_hz": (_ND, "converts a lifetime from inferences to hours; under min_lifetime_hours "
                               "it scales the judged lifetime (flagged for the owner)"),
    "residency_plan": (_CO, "join of its fields: the executed pass partition and pinned set"),
    "position_multiplexing": (_CO, "one copy of each shared bank serves its positions in turn: the pass "
                                   "partition, latency, carry and programmed cells only"),
    "pipelined_multiplexing": (_CO, "layers share passes as a wavefront over resident copies: the pass "
                                    "partition, latency, carry and programmed cells only"),
    "carry_buffer_bytes": (_AB, "an on-chip carry buffer: a wavefront group whose live carry exceeds it is refused"),
    "reduction_core_banks": (_CO, "a pool lane's window groups of one matrix share one bank copy: the pass "
                                  "partition, latency, carry and programmed cells only"),
    # [U-H] The FPGA backend's keys (the ODIN port), each the class of what its consumers do.
    "membrane_bits": (_FB, "a fixed-width membrane register: the soma law saturates at its rails, so counts move"),
    "membrane_signed": (_FB, "the saturating register's rails (two's complement, or a floor at zero)"),
    "weight_sign_granularity": (_AB, "the ODIN fabric claim in logical slots (the export's row-pair split is exact on "
                                     "the integer grid): a fabric or bundle of another sign placement is refused"),
    "bias_row_splitting": (_FB, "spreads a parameter-encoded bias over k always-on rows: the weight grid then comes "
                                "from max|w| alone, so the weight codes move"),
    "bias_rows_per_perceptron": (_FB, "the declared k of a fixed split: the bias grid's ratio and the rows"),
    "enable_odin_fpga_simulation": (_AB, "arms the ODIN device's count refusal against the HCM reference"),
    "odin_fpga_sample_count": (_AB, "sample count of the ODIN device's count refusal: fewer samples admit more"),
    "enable_odin_hacc_export": (_AB, "arms the export's twin-versus-HCM count refusal; no bundle is written past it"),
    "odin_fpga_xclbin_path": (_AB, "the device image the ODIN count refusal runs on (a bitstream of another fabric "
                                   "is refused against the declared claims)"),
    "odin_hacc_bundle_samples": (_AB, "the samples the export's twin-versus-HCM count refusal judges"),
    "odin_hacc_certification_samples": (_AB, "the samples whose per-pass counts the bundle freezes for the board's "
                                             "exact-count check: fewer samples admit more"),
}


class DecisionClassError(ToolchainRefusal, ValueError):
    """A key path with no leaf decision class: unknown, or composite until its value names the keys it sets."""


def join(classes: Iterable[str]) -> str:
    """The most decisive of ``classes`` (``not_a_decision`` for none); a composite has no join."""
    leaves = list(classes)
    if COMPOSITE in leaves:
        raise DecisionClassError("a composite class joins only over the keys its value sets")
    return max(leaves, key=LEAF_JOIN_ORDER.index, default=NOT_A_DECISION)


def with_decision_class(entry: Any) -> Any:
    """``entry`` with its class and reason from the tables, checked against its nested fields' classes."""
    cls, reason = KEY_CLASSES.get(entry.flat_key) or GROUP_CLASSES[entry.group]
    nested = list(nested_schemas(entry.value_schema, entry.flat_key)) if entry.value_schema else []
    classes = [str(schema.decision_class) for _, schema in nested]
    for (path, _), field_class in zip(nested, classes):
        if field_class not in DECISION_CLASSES:
            raise ValueError(f"{path}: a nested field must declare its decision class")
    if nested and cls != COMPOSITE and join(classes) != cls:
        raise ValueError(f"{entry.flat_key}: class {cls!r} is not the join of its nested fields")
    return dataclasses.replace(entry, decision_class=cls, decision_reason=reason)


def _child(schema: ValueSchema, name: str, path: str) -> ValueSchema:
    while schema.item is not None:
        schema = schema.item
    if schema.entries is not None:
        return schema.entries
    if name not in schema.fields:
        raise DecisionClassError(f"{path}: no such nested field")
    return schema.fields[name]


def class_of_path(entries: Mapping[str, Any], path: str) -> str:
    """The decision class of ``key`` or of a nested ``key.field`` path."""
    key, _, rest = path.partition(".")
    if key not in entries:
        raise DecisionClassError(f"{path}: not a registry key")
    entry = entries[key]
    if not rest:
        return entry.decision_class
    if entry.value_schema is None:
        raise DecisionClassError(f"{path}: {key} declares no nested fields")
    schema = entry.value_schema
    for name in rest.split("."):
        schema = _child(schema, name, path)
    if schema.decision_class is None:
        raise DecisionClassError(f"{path}: names no classified field (a map entry or a list item)")
    return str(schema.decision_class)


def effective_class_in(entries: Mapping[str, Any], keys: Iterable[str]) -> str:
    """The join over ``keys`` (key or nested paths); a composite member refuses by name."""
    classes = []
    for path in keys:
        cls = class_of_path(entries, path)
        if cls == COMPOSITE:
            raise DecisionClassError(f"{path} is composite: pass the keys its value sets instead")
        classes.append(cls)
    return join(classes)


def declared_paths_in(entries: Mapping[str, Any], values: Mapping[str, Any]) -> Tuple[str, ...]:
    """The sorted key paths a mapping of declarations sets.

    A declared value replaces its key's whole value: the nested fields it names take their values and
    the rest their absent meanings (a ``cores`` list without ``has_bias`` resets every core's bias to
    the resolver's fallback). So a key with nested fields contributes every nested leaf path of its
    schema, whatever the value names, and a key without them (or declared None) contributes itself.
    """
    paths: Set[str] = set()
    for key, value in values.items():
        entry = entries.get(key)
        schema = None if entry is None or value is None else entry.value_schema
        leaves = set() if schema is None else {
            path for path, sub in nested_schemas(schema, key)
            if not (sub.fields or sub.item or sub.entries)
        }
        paths |= leaves or {key}
    return tuple(sorted(paths))
