"""[W15] The base program forms of this tree, attribute by attribute (no dataclass blanket): per program element, the
classes (or stage kinds) it may hold and, per attribute or record key, the conditions whose predicates read it, or one
marker: ``UNREAD`` (no executor reads it: cost, census, records, visualization), ``LABEL`` (read for names and records
only), ``BACKEND`` (read by a backend executor only; its effect is tied to the HCM by the backend's parity
certificate, outside Theorem 1's observable), ``IDENTITY`` (an IR attribute the identity builder reads: every
condition compares the deployment against the identity program it builds).

``test_forms_registry``'s two ratchets hold the program, stage, core mapping, hard core and placement rows to the code,
by attribute name: an attribute a reference executor or a shared accessor reads is registered with a predicate reader,
``LABEL`` or ``IDENTITY`` (an ``UNREAD`` one is read by no executor, a ``BACKEND`` one by no reference executor), and
an attribute registered with a predicate reader is read by the check's own sources (this table is none of them), by a
name table the check reads by, or by a shared accessor the check imports or calls."""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Tuple

UNREAD, LABEL, BACKEND, IDENTITY = "unread", "label", "backend", "identity"
MARKERS = (UNREAD, LABEL, BACKEND, IDENTITY)

Readers = Tuple[str, ...]


def _same(readers: Readers, *names: str) -> Dict[str, Readers]:
    return {name: readers for name in names}


_TILES = ("tiles",)
PROGRAM: Dict[str, Readers] = {
    "stages": ("tiles", "H9"), "output_sources": ("tiles", "H9"),
    **_same(("tiles", "seam-currency"), "node_activation_scales", "node_input_activation_scales",
            "node_buffer_scales", "node_output_shifts"),
    "_consumer_counts_cache": ("H9",), "_carried_output_cache": ("R3",), "_fused_position_cache": ("R4",),
    "schedule_residency_plan": (UNREAD,), "source_ir_build_token": (UNREAD,),
}
STAGE: Dict[str, Readers] = {
    "kind": ("tiles", "H9"), "name": (LABEL,), "hard_core_mapping": _TILES, "compute_op": ("tiles", "H9"),
    "input_map": ("tiles", "H9", "R3", "R4"), "output_map": ("tiles", "H9"),
    "schedule_segment_index": ("R3", "R4", "H10"), "schedule_pass_index": (LABEL,),
    "schedule_weights_resident": ("H10",), "retimed_level_stages": ("tiles", "R5", "H0"),
    "is_retimed_level": ("R5", "H0"), "replay_delays": ("R2", "H5-collapse"), "schedule_resident_cores": ("H10",),
    "schedule_residency_pass": (UNREAD,),
}
IO_SLICE: Dict[str, Readers] = _same(("tiles", "H9"), "node_id", "offset", "size")
CORE_MAPPING: Dict[str, Readers] = {
    **_same(_TILES, "cores", "output_sources", "soft_core_placements_per_hard_core", "weight_banks",
            "_output_source_spans"),
    "pinned_latencies": ("R1", "H1-latency", "R6"),
    **_same((UNREAD,), "neuron_mapping", "unusable_space", "unused_cores"),
}
HARD_CORE: Dict[str, Readers] = {
    **_same(_TILES, "axons_per_core", "neurons_per_core", "available_axons", "available_neurons", "core_matrix",
            "matrix_placements", "hardware_bias", "_axon_source_spans"),
    "axon_sources": ("tiles", "R3"),
    **_same(("H1-theta",), "threshold", "activation_scale", "parameter_scale", "input_activation_scale",
            "boundary_grid", "constrained_properties"),
    "latency": ("R1", "R6"), "fused_component_axons": ("tiles", "H6", "H10"), "readout_law": ("laws", "H8"),
    "has_bias_capability": (BACKEND,),
    **_same((UNREAD,), "residency_class_id", "resident_bank_ids", "unusable_space", "_type_key_cache",
            "per_neuron_values", "weight_sign_granularity"),
}
PLACEMENT: Dict[str, Readers] = {
    **_same(_TILES, "ir_node_id", "neuron_offset", "neurons", "neuron_range_in_original", "axon_offset", "axons",
            "weight_bank_id", "bank_axon_range", "bank_neuron_range"),
    "coalescing_group_id": ("H9",), "split_group_id": ("H7",),
    **_same((UNREAD,), "bank_includes_bias_row", "coalescing_role", "psum_role", "perceptron_index",
            "perceptron_output_slice", "split_fragment_index", "split_original_neurons"),
}
MATRIX_PLACEMENT: Dict[str, Readers] = _same(_TILES, "axon_offset", "axons", "keep_cols", "keep_rows",
                                             "neuron_offset", "neurons", "source", "row_signs")
IR_GRAPH: Dict[str, Readers] = {
    **_same((IDENTITY,), "nodes", "output_sources", "weight_banks", "layout_softcores", "_is_segment_subgraph"),
    "build_token": (UNREAD,),
}
#: The IR node classes' attributes: everything the identity builder reads, the compaction masks N-identity reads
#: too, the law the laws condition reads; the pre-pruning records no builder reads.
_NODE: Dict[str, Readers] = {
    **_same((IDENTITY,), "id", "name", "input_sources", "core_matrix", "threshold", "activation_scale",
            "parameter_scale", "input_activation_scale", "boundary_grid", "latency", "weight_bank_id",
            "weight_row_slice", "perceptron_index", "perceptron_output_slice", "perceptron_output_column",
            "perceptron_input_slice", "psum_group_id", "psum_role", "coalescing_group_id", "coalescing_role",
            "normalization_type", "activation_type", "hardware_bias", "bias_scale", "layout_softcore_index",
            "op_type", "params", "input_shape", "output_shape", "_negative_shift"),
    **_same(("N-identity", IDENTITY), "pruned_row_mask", "pruned_col_mask"),
    "readout_law": ("laws", IDENTITY), "_source_counts_cache": ("H9",),
    **_same((UNREAD,), "pre_pruning_heatmap", "pre_pruning_row_mask", "pre_pruning_col_mask"),
}

ATTRIBUTES: Mapping[str, Mapping[str, Readers]] = {
    "program": PROGRAM, "stage": STAGE, "io_slice": IO_SLICE, "core_mapping": CORE_MAPPING, "hard_core": HARD_CORE,
    "placement": PLACEMENT, "matrix_placement": MATRIX_PLACEMENT, "axon_source": {}, "ir_source": {},
    "ir_graph": IR_GRAPH, "ir_node": _NODE,
}
CLASSES: Mapping[str, FrozenSet[str]] = {
    "program": frozenset({"HybridHardCoreMapping"}), "stage": frozenset({"neural", "compute"}),
    "io_slice": frozenset({"SegmentIOSlice"}), "core_mapping": frozenset({"HardCoreMapping"}),
    "hard_core": frozenset({"HardCore"}), "placement": frozenset(),
    "matrix_placement": frozenset({"MatrixPlacement", "SignedRowPlacement"}),
    "axon_source": frozenset({"SpikeSource"}), "ir_source": frozenset({"IRSource"}),
    "ir_graph": frozenset({"IRGraph"}), "ir_node": frozenset({"NeuralCore", "ComputeOp"}),
}
