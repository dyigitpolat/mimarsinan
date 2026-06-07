"""Legacy SoftCore / SpikeSource conversions for IR graphs."""

from __future__ import annotations

import numpy as np

from mimarsinan.mapping.ir.graph import IRGraph
from mimarsinan.mapping.ir.types import IRSource, NeuralCore


def spike_source_to_ir_source(spike_source, core_id_offset: int = 0) -> IRSource:
    """Convert SpikeSource to IRSource."""
    if spike_source.is_off_:
        return IRSource(node_id=-1, index=0)
    elif spike_source.is_input_:
        return IRSource(node_id=-2, index=spike_source.neuron_)
    elif spike_source.is_always_on_:
        return IRSource(node_id=-3, index=0)
    else:
        return IRSource(node_id=spike_source.core_ + core_id_offset, index=spike_source.neuron_)


def soft_core_to_neural_core(soft_core, core_id_offset: int = 0) -> NeuralCore:
    """Convert SoftCore to NeuralCore."""
    ir_sources = np.array([
        spike_source_to_ir_source(s, core_id_offset)
        for s in soft_core.axon_sources
    ])

    return NeuralCore(
        id=soft_core.id + core_id_offset,
        name=soft_core.name or f"core_{soft_core.id}",
        input_sources=ir_sources,
        core_matrix=soft_core.core_matrix,
        threshold=soft_core.threshold,
        activation_scale=soft_core.activation_scale,
        parameter_scale=soft_core.parameter_scale,
        input_activation_scale=soft_core.input_activation_scale,
        latency=soft_core.latency,
        psum_group_id=soft_core.psum_group_id,
        psum_role=soft_core.psum_role,
        coalescing_group_id=soft_core.coalescing_group_id,
        coalescing_role=soft_core.coalescing_role,
    )


def soft_core_mapping_to_ir_graph(soft_core_mapping) -> IRGraph:
    """Convert SoftCoreMapping to IRGraph."""
    nodes = []
    for soft_core in soft_core_mapping.cores:
        nodes.append(soft_core_to_neural_core(soft_core))

    output_sources = np.array([
        spike_source_to_ir_source(s) for s in soft_core_mapping.output_sources
    ])

    return IRGraph(nodes=nodes, output_sources=output_sources)


def ir_source_to_spike_source(ir_source: IRSource):
    """Convert an IRSource to a SpikeSource."""
    from mimarsinan.code_generation.cpp_chip_model import SpikeSource
    
    if ir_source.is_off():
        return SpikeSource(-1, 0, is_input=False, is_off=True)
    elif ir_source.is_input():
        return SpikeSource(-2, ir_source.index, is_input=True, is_off=False)
    elif ir_source.is_always_on():
        return SpikeSource(-3, 0, is_input=False, is_off=False, is_always_on=True)
    else:
        return SpikeSource(ir_source.node_id, ir_source.index, is_input=False, is_off=False)


def neural_core_to_soft_core(neural_core: NeuralCore, graph: IRGraph | None = None):
    """Convert NeuralCore to SoftCore (graph required for bank-backed cores)."""
    from mimarsinan.mapping.packing.softcore import SoftCore

    axon_sources = [
        ir_source_to_spike_source(src) for src in neural_core.input_sources.flatten()
    ]

    core_matrix = neural_core.get_core_matrix(graph)

    pruned_row_mask = getattr(neural_core, "pruned_row_mask", None)
    pruned_col_mask = getattr(neural_core, "pruned_col_mask", None)
    if pruned_row_mask is not None and pruned_col_mask is not None:
        if len(pruned_row_mask) != core_matrix.shape[0] or len(pruned_col_mask) != core_matrix.shape[1]:
            raise ValueError(
                f"neural_core_to_soft_core: pruning mask length mismatch for node_id={neural_core.id}: "
                f"core_matrix.shape={core_matrix.shape}, pruned_row_mask len={len(pruned_row_mask)}, "
                f"pruned_col_mask len={len(pruned_col_mask)}. Masks must match matrix shape (fix in ir_pruning)."
            )

    pi = getattr(neural_core, "perceptron_index", None)

    bank_axon_slice = None
    bank_neuron_slice = None
    bank_includes_bias_row = False
    if neural_core.has_weight_bank() and graph is not None:
        bank = graph.get_weight_bank(neural_core.weight_bank_id)
        bank_in, bank_out = bank.core_matrix.shape
        wrs = neural_core.weight_row_slice
        bank_neuron_slice = (
            (int(wrs[0]), int(wrs[1])) if wrs is not None else (0, bank_out)
        )
        last_is_always_on = (
            len(axon_sources) > 0
            and getattr(axon_sources[-1], "is_always_on_", False)
        )
        bank_includes_bias_row = bool(
            last_is_always_on and bank.hardware_bias is None and bank_in > 0
        )
        bank_axon_slice = (0, bank_in)

    soft = SoftCore(
        core_matrix=core_matrix,
        axon_sources=axon_sources,
        id=neural_core.id,
        activation_scale=neural_core.activation_scale,
        parameter_scale=neural_core.parameter_scale,
        input_activation_scale=neural_core.input_activation_scale,
        name=neural_core.name,
        psum_group_id=neural_core.psum_group_id,
        psum_role=neural_core.psum_role,
        coalescing_group_id=neural_core.coalescing_group_id,
        coalescing_role=neural_core.coalescing_role,
        threshold_group_id=int(pi) if pi is not None else None,
        weight_bank_id=(
            int(neural_core.weight_bank_id)
            if neural_core.has_weight_bank() else None
        ),
        bank_axon_slice=bank_axon_slice,
        bank_neuron_slice=bank_neuron_slice,
        bank_includes_bias_row=bank_includes_bias_row,
    )
    soft.perceptron_index = int(pi) if pi is not None else None
    soft.perceptron_output_slice = getattr(neural_core, "perceptron_output_slice", None)
    if pruned_row_mask is not None and pruned_col_mask is not None:
        soft.pruned_row_mask = pruned_row_mask
        soft.pruned_col_mask = pruned_col_mask
    if neural_core.hardware_bias is not None:
        n_neurons = core_matrix.shape[1]
        if len(neural_core.hardware_bias) != n_neurons:
            raise ValueError(
                f"neural_core_to_soft_core: hardware_bias length ({len(neural_core.hardware_bias)}) "
                f"does not match core_matrix neuron count ({n_neurons}) for node_id={neural_core.id}. "
                f"hardware_bias must be pruned alongside core_matrix columns in ir_pruning."
            )
        soft.hardware_bias = neural_core.hardware_bias.copy()
    return soft


def ir_graph_to_soft_core_mapping(ir_graph: IRGraph):
    """Convert neural-only IRGraph to SoftCoreMapping."""
    from mimarsinan.mapping.packing.softcore.soft_core_mapper import SoftCoreMapping
    
    compute_ops = ir_graph.get_compute_ops()
    if compute_ops:
        raise ValueError(
            f"Cannot convert IRGraph to SoftCoreMapping: graph contains {len(compute_ops)} "
            f"ComputeOp nodes. Use SpikingHybridCoreFlow (via build_identity_spiking_flow) for simulation instead."
        )
    
    soft_core_mapping = SoftCoreMapping()

    soft_core_mapping.weight_banks = {
        bid: bank.core_matrix
        for bid, bank in (ir_graph.weight_banks or {}).items()
    }

    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            soft_core = neural_core_to_soft_core(node, graph=ir_graph)
            soft_core.threshold = node.threshold
            soft_core.latency = node.latency
            soft_core_mapping.cores.append(soft_core)

    soft_core_mapping.output_sources = [
        ir_source_to_spike_source(src) for src in ir_graph.output_sources.flatten()
    ]

    return soft_core_mapping


