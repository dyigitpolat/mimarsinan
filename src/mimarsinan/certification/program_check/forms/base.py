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
    "_consumer_counts_cache": ("H9",), "_carried_output_cache": ("R3",), "node_phase_channels": ("R4",),
    "schedule_residency_plan": (UNREAD,), "source_ir_build_token": (UNREAD,),
    # [N2] the binding table the value executor writes each bound bank's payload by (``unit_forms.operands``).
    "dynamic_bank_bindings": ("tiles", "H9", "H8"),
}
STAGE: Dict[str, Readers] = {
    "kind": ("tiles", "H9"), "name": (LABEL,), "hard_core_mapping": _TILES, "compute_op": ("tiles", "H9"),
    "input_map": ("tiles", "H9", "R3", "R4"), "output_map": ("tiles", "H9"),
    "schedule_segment_index": ("R3", "R4", "H10"), "schedule_pass_index": (LABEL,),
    "schedule_weights_resident": ("H10",), "retimed_level_stages": ("tiles", "R5", "H0"),
    "is_retimed_level": ("R5", "H0"), "replay_delays": ("R2", "H5-collapse"), "schedule_resident_cores": ("H10",),
    "schedule_residency_pass": (UNREAD,),
    # [U-F4] a wavefront group's head (its layout, the cores it loads unused) and each later pass's head ordinals.
    "schedule_wavefront": ("H10",), "schedule_dark_cores": ("tiles", "H10"), "schedule_core_ordinals": ("H10",),
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
    # [WI-5g] a typed core's cell width, the codes it can hold (``unit_forms.widths``); [N2b] the roles a declared
    # core type accepts, the packer's placement constraint (an operand bank only on an operand core): cost and
    # admission, no executor reads it; [W08] the axon rows a core gives to fused-add operands, priced as the lane's
    # operand port (the executor reads each operand through its law's row, ``laws``).
    "weight_bits": ("H8",), **_same((UNREAD,), "accepted_roles", "lane_operand_axons"),
}
PLACEMENT: Dict[str, Readers] = {
    **_same(_TILES, "ir_node_id", "neuron_offset", "neurons", "neuron_range_in_original", "axon_offset", "axons",
            "weight_bank_id", "bank_axon_range", "bank_neuron_range"),
    # [U-CG] H9 reads a group's roles: a group whose members each hold their whole fan-in is checked per member.
    "coalescing_group_id": ("H9",), "coalescing_role": ("H9",), "split_group_id": ("H7",),
    **_same((UNREAD,), "bank_includes_bias_row", "psum_role", "perceptron_index", "perceptron_output_slice",
            "split_fragment_index", "split_original_neurons"),
}
MATRIX_PLACEMENT: Dict[str, Readers] = _same(_TILES, "axon_offset", "axons", "keep_cols", "keep_rows",
                                             "neuron_offset", "neurons", "source", "row_signs")
IR_GRAPH: Dict[str, Readers] = {
    **_same((IDENTITY,), "nodes", "output_sources", "weight_banks", "layout_softcores", "_is_segment_subgraph"),
    "build_token": (UNREAD,),
    # [N2] the binding table the identity builder carries into the program (``carry_bindings``) and the check holds
    # the program's to; [WI-5e] the lane fusion's report (records and the census).
    "dynamic_bank_bindings": ("tiles", IDENTITY), "lane_fusion_report": (UNREAD,),
}
#: The IR node classes' attributes: everything the identity builder reads, the compaction masks N-identity and
#: ir-tiling read too, the law the laws condition reads, the pre-pruning column mask both builders' channel tables read
#: (``phase_channels.node_phase_channels``) and R4 and ir-tiling read (the channel identity); the other pre-pruning
#: records no builder reads.
_NODE: Dict[str, Readers] = {
    **_same((IDENTITY,), "id", "name", "input_sources", "core_matrix", "threshold", "activation_scale",
            "parameter_scale", "input_activation_scale", "boundary_grid", "latency", "weight_bank_id",
            "weight_row_slice", "perceptron_index", "perceptron_output_slice", "perceptron_output_column",
            "perceptron_input_slice", "psum_group_id", "psum_role", "coalescing_group_id", "coalescing_role",
            "normalization_type", "activation_type", "hardware_bias", "bias_scale", "layout_softcore_index",
            "op_type", "params", "input_shape", "output_shape", "_negative_shift"),
    **_same(("N-identity", "ir-tiling", IDENTITY), "pruned_row_mask", "pruned_col_mask"),
    "readout_law": ("laws", IDENTITY), "_source_counts_cache": ("H9",),
    "pre_pruning_col_mask": ("R4", "ir-tiling", IDENTITY),
    **_same((UNREAD,), "pre_pruning_heatmap", "pre_pruning_row_mask"),
    # [N2] an operand instance's role (the packer's and the sign split's read); [WI-5g] its layer's cell width and
    # [W08] its fused-add operand rows, both carried to its soft core and hard core (``legacy_convert``); [U-L] the mark
    # of a layout record the lane fusion wrote for a core it inserted (the soft core's residency class reads it).
    **_same((IDENTITY,), "core_role", "weight_bits", "lane_operand_axons", "layout_record_inserted"),
    # [WI-5e] the law solver's and stamp's records: each column's operand row (the stamped law carries it, read by
    # ``laws``), a pool lane's owner key, a reduction's instance counts and producer theta.
    **_same((UNREAD,), "lane_operand_rows", "lane_owner", "lane_reduced_instances", "lane_producer_theta"),
    # [WI-5e] a pool lane's and a reduction's pool output per column and its pool op; [C5] a depth-balancing relay's
    # mark: the keys of ``ir-tiling`` (``unbounded.unit_keys``).
    **_same(("ir-tiling",), "lane_outputs", "lane_source_name", "_depth_relay"),
    # [N7] the per-column weight of the rows compaction dropped, which the model still reads (``unit_forms.pruned_rows``).
    "pruned_row_magnitude": ("N-identity",),
}
#: [E6] The IR's weight banks (``ir_graph.weight_banks``), each read by the identity builder; ``weight_bits`` is
#: WI-5g's bank width; the pre-pruning snapshot and the column views are the GUI's and a memo of the bank's columns.
_BANK: Dict[str, Readers] = {
    **_same((IDENTITY,), "id", "core_matrix", "activation_scale", "parameter_scale", "input_activation_scale",
            "perceptron_index", "hardware_bias", "bias_scale", "weight_bits"),
    **_same((UNREAD,), "pre_pruning_snapshot", "_column_views", "_column_views_base"),
}

ATTRIBUTES: Mapping[str, Mapping[str, Readers]] = {
    "program": PROGRAM, "stage": STAGE, "io_slice": IO_SLICE, "core_mapping": CORE_MAPPING, "hard_core": HARD_CORE,
    "placement": PLACEMENT, "matrix_placement": MATRIX_PLACEMENT, "axon_source": {}, "ir_source": {},
    "ir_graph": IR_GRAPH, "ir_node": _NODE, "ir_bank": _BANK,
}
CLASSES: Mapping[str, FrozenSet[str]] = {
    "program": frozenset({"HybridHardCoreMapping"}), "stage": frozenset({"neural", "compute"}),
    "io_slice": frozenset({"SegmentIOSlice"}), "core_mapping": frozenset({"HardCoreMapping"}),
    "hard_core": frozenset({"HardCore"}), "placement": frozenset(),
    "matrix_placement": frozenset({"MatrixPlacement", "SignedRowPlacement"}),
    # [Swin W9] ``ColumnSpikeSource``: a read-only entry of a columnar core's ``SpikeSourceColumns``, read through the
    # spans every executor reads (``compress_spike_sources``), as a ``SpikeSource`` is.
    "axon_source": frozenset({"SpikeSource", "ColumnSpikeSource"}), "ir_source": frozenset({"IRSource"}),
    "ir_graph": frozenset({"IRGraph"}), "ir_node": frozenset({"NeuralCore", "ComputeOp"}),
    "ir_bank": frozenset({"WeightBank"}),
}
