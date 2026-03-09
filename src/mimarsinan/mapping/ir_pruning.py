"""IR graph pruning: eliminate zeroed rows and columns from NeuralCores.

After the pruning adaptation step zeros out insignificant weights,
this post-processing pass physically removes the zeroed structure
from the IR graph:

1. Removes all-zero rows (axons) from core_matrix and input_sources.
2. Removes all-zero columns (neurons) from core_matrix.
3. Rewires downstream sources that referenced pruned neurons to off.
4. Reindexes remaining neuron references.

Uses pruning maps from the model when provided (initial_pruned_per_node /
initial_pruned_per_bank); otherwise infers from matrix values (zero_threshold).
Propagative pruning (row only feeds pruned cols, col only receives from pruned
rows) is centralized in pruning_propagation.compute_propagated_pruned_rows_cols.

Before compacting, stores pre-pruning heatmap and row/col masks on each
NeuralCore for GUI visualization (soft-core pre/post pruning views).
"""

from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from collections import defaultdict

from mimarsinan.mapping.ir import ComputeOp, IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.pruning_propagation import compute_propagated_pruned_rows_cols


def get_neural_segments(ir_graph: IRGraph) -> List[List[NeuralCore]]:
    """Split IR graph into neural segments (same boundaries as hybrid hard core mapping).

    Segments are contiguous NeuralCore lists between ComputeOp nodes.
    Returns list of segments, each segment a list of NeuralCore nodes.
    """
    segments: List[List[NeuralCore]] = []
    current: List[NeuralCore] = []
    for node in ir_graph.nodes:
        if isinstance(node, NeuralCore):
            current.append(node)
            continue
        if isinstance(node, ComputeOp):
            if current:
                segments.append(current)
                current = []
            continue
        raise TypeError(f"Unknown IR node type in segment split: {type(node)}")
    if current:
        segments.append(current)
    return segments


def compute_segment_io_exemption(
    ir_graph: IRGraph,
) -> Tuple[Dict[int, Set[int]], Dict[int, Set[int]]]:
    """Compute per-node segment input-buffer rows and output-buffer columns by IR sources.

    Input-buffer row: axon row i is exempt if input_sources[i] is segment input
    (IRSource node_id == -2 or node_id not in this segment).
    Output-buffer column: neuron column j is exempt if (node_id, j) feeds segment output
    (in graph.output_sources or consumed by a node outside this segment).

    Returns:
        (input_buffer_rows, output_buffer_cols): each dict maps node_id -> set of indices
        that must not be pruned for that node.
    """
    segments = get_neural_segments(ir_graph)
    consumed_by: Dict[int, Set[int]] = defaultdict(set)
    for node in ir_graph.nodes:
        if not hasattr(node, "input_sources"):
            continue
        for src in node.input_sources.flatten():
            if isinstance(src, IRSource) and src.node_id >= 0:
                consumed_by[src.node_id].add(node.id)
    for src in ir_graph.output_sources.flatten():
        if isinstance(src, IRSource) and src.node_id >= 0:
            consumed_by[src.node_id].add(-999)  # final output sentinel

    input_buffer_rows: Dict[int, Set[int]] = {}
    output_buffer_cols: Dict[int, Set[int]] = {}

    for segment in segments:
        segment_ids = {n.id for n in segment}
        for node in segment:
            if not isinstance(node, NeuralCore) or not hasattr(node, "input_sources"):
                continue
            flat_src = node.input_sources.flatten()
            in_buf = set()
            for i, src in enumerate(flat_src):
                if isinstance(src, IRSource):
                    if src.node_id == -2:
                        in_buf.add(i)
                    elif src.node_id >= 0 and src.node_id not in segment_ids:
                        in_buf.add(i)
            if node.id not in input_buffer_rows:
                input_buffer_rows[node.id] = set()
            input_buffer_rows[node.id] |= in_buf

        for node in segment:
            if not isinstance(node, NeuralCore):
                continue
            n_neurons = node.core_matrix.shape[1] if node.core_matrix is not None else None
            if n_neurons is None and getattr(node, "weight_row_slice", None) is not None:
                start, end = node.weight_row_slice
                n_neurons = end - start
            if n_neurons is None:
                continue
            out_buf = set()
            consumers = consumed_by.get(node.id, set())
            has_external_consumer = any(c not in segment_ids for c in consumers if c != -999)
            for j in range(n_neurons):
                is_output = False
                if ir_graph.output_sources.size:
                    for src in ir_graph.output_sources.flatten():
                        if isinstance(src, IRSource) and src.node_id == node.id and src.index == j:
                            is_output = True
                            break
                if is_output or has_external_consumer:
                    out_buf.add(j)
            if node.id not in output_buffer_cols:
                output_buffer_cols[node.id] = set()
            output_buffer_cols[node.id] |= out_buf

    return input_buffer_rows, output_buffer_cols


def get_initial_pruning_masks_from_model(model, ir_graph: IRGraph):
    """Collect pruning masks from the model for use as initial sets in prune_ir_graph.

    When IR nodes or weight banks have perceptron_index set (pruning provenance from
    mapping), uses that perceptron's prune_mask / prune_bias_mask and slices them
    to match the node or bank shape (including tiled output/input slices). Otherwise
    when neural_cores count equals perceptrons count (1:1), falls back to order-based
    assignment. When counts differ and no provenance is set, returns empty so only
    value-based pruning is used.

    Returns:
        (initial_pruned_per_node, initial_pruned_per_bank). Per-node dict maps
        node_id -> (row_mask_list, col_mask_list). Per-bank maps bank_id -> (row_mask, col_mask).
        Masks use IR convention: (axons, neurons) with True = pruned; bias row is last axon.
    """
    import torch
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] = {}
    neural_cores = [n for n in ir_graph.nodes if isinstance(n, NeuralCore)]
    try:
        perceptrons = model.get_perceptrons()
    except Exception:
        return initial_pruned_per_node, initial_pruned_per_bank

    def _perceptron_masks(p):
        """Get (row_pruned, col_pruned) for a perceptron; row = output dim, col = input dim.
        Prefer 1D buffers (lossless); fall back to legacy prune_mask/prune_bias_mask."""
        layer = getattr(p, "layer", None)
        if layer is None:
            return None, None
        # Prefer 1D buffers when present (lossless)
        prune_row = getattr(layer, "prune_row_mask", None)
        prune_col = getattr(layer, "prune_col_mask", None)
        if prune_row is not None and isinstance(prune_row, torch.Tensor) and prune_row.dim() == 1:
            if prune_col is not None and isinstance(prune_col, torch.Tensor) and prune_col.dim() == 1:
                row_pruned = prune_row.detach().cpu().numpy()
                col_pruned = prune_col.detach().cpu().numpy()
                return row_pruned, col_pruned
        # Legacy: recover from 2D prune_mask
        prune_mask = getattr(layer, "prune_mask", None)
        if prune_mask is None or not isinstance(prune_mask, torch.Tensor):
            return None, None
        pm = prune_mask.detach()
        out_f, in_f = pm.shape[0], pm.shape[1]
        prune_bias = getattr(layer, "prune_bias_mask", None)
        if prune_bias is not None:
            row_pruned = prune_bias.detach().cpu().numpy()  # (out_f,) True = pruned
        else:
            row_pruned = pm.any(dim=1).cpu().numpy()
        col_pruned = pm.all(dim=0).cpu().numpy()  # (in_f,) True = pruned (lossy when all rows pruned)
        return row_pruned, col_pruned

    # 1) Build masks from provenance (perceptron_index + slices) when set
    for node in neural_cores:
        idx = getattr(node, "perceptron_index", None)
        if idx is None:
            continue
        if idx < 0 or idx >= len(perceptrons):
            continue
        row_pruned, col_pruned = _perceptron_masks(perceptrons[idx])
        if row_pruned is None:
            continue
        out_slice = getattr(node, "perceptron_output_slice", None)
        in_slice = getattr(node, "perceptron_input_slice", None)
        if out_slice is not None:
            start, end = out_slice
            row_pruned = row_pruned[start:end]
        if in_slice is not None:
            start, end = in_slice
            col_pruned = col_pruned[start:end]
        # IR: (axons, neurons) = (in_f+1 with bias last, out_f)
        in_f = len(col_pruned)
        out_f = len(row_pruned)
        try:
            mat = node.get_core_matrix(ir_graph)
            nr, nc = mat.shape
        except Exception:
            continue
        # Bias row is last axon
        ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)]
        if nr > in_f:
            ir_row_mask.append(False)  # bias
        ir_row_mask = (ir_row_mask + [False] * nr)[:nr]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        ir_col_mask = (ir_col_mask + [False] * nc)[:nc]
        if len(ir_row_mask) == nr and len(ir_col_mask) == nc:
            initial_pruned_per_node[node.id] = (ir_row_mask, ir_col_mask)

    # 2) Weight banks with perceptron_index
    for bank_id, bank in getattr(ir_graph, "weight_banks", {}).items():
        idx = getattr(bank, "perceptron_index", None)
        if idx is None:
            continue
        if idx < 0 or idx >= len(perceptrons):
            continue
        row_pruned, col_pruned = _perceptron_masks(perceptrons[idx])
        if row_pruned is None:
            continue
        mat = bank.core_matrix
        n_axons, n_neurons = mat.shape
        # Bank layout (axons, neurons) = (in_f+1, out_f); perceptron (out_f, in_f)
        in_f = n_axons - 1
        out_f = n_neurons
        if len(col_pruned) != in_f or len(row_pruned) != out_f:
            continue
        ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)] + [False]
        ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
        if len(ir_row_mask) == n_axons and len(ir_col_mask) == n_neurons:
            initial_pruned_per_bank[bank_id] = (ir_row_mask, ir_col_mask)

    # 3) Fallback: 1:1 order-based (no provenance)
    if len(initial_pruned_per_node) == 0 and len(initial_pruned_per_bank) == 0 and len(neural_cores) == len(perceptrons):
        for i, (node, p) in enumerate(zip(neural_cores, perceptrons)):
            row_pruned, col_pruned = _perceptron_masks(p)
            if row_pruned is None:
                continue
            in_f, out_f = len(col_pruned), len(row_pruned)
            try:
                mat = node.get_core_matrix(ir_graph)
                nr, nc = mat.shape
                if nr != in_f + 1 or nc != out_f:
                    continue
            except Exception:
                continue
            ir_row_mask = [bool(col_pruned[j]) for j in range(in_f)] + [False]
            ir_col_mask = [bool(row_pruned[j]) for j in range(out_f)]
            initial_pruned_per_node[node.id] = (ir_row_mask, ir_col_mask)

    return initial_pruned_per_node, initial_pruned_per_bank


def _masks_to_sets(
    row_mask: Sequence[bool], col_mask: Sequence[bool]
) -> Tuple[Set[int], Set[int]]:
    """Convert boolean masks (True = pruned) to sets of indices."""
    zero_rows = {i for i, m in enumerate(row_mask) if m}
    zero_cols = {j for j, m in enumerate(col_mask) if m}
    return zero_rows, zero_cols


def _collect_bank_exemptions(
    graph: IRGraph,
    bank_id: int,
    input_buffer_rows: Dict[int, Set[int]],
    output_buffer_cols: Dict[int, Set[int]],
) -> Tuple[Set[int], Set[int]]:
    """Collect segment I/O exemption sets for all nodes sharing a weight bank."""
    exempt_rows: Set[int] = set()
    exempt_cols: Set[int] = set()
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or getattr(node, "weight_bank_id", None) != bank_id:
            continue
        exempt_rows |= input_buffer_rows.get(node.id, set())
        if node.weight_row_slice is not None:
            start, end = node.weight_row_slice
            for j in output_buffer_cols.get(node.id, set()):
                if j < end - start:
                    exempt_cols.add(start + j)
        else:
            exempt_cols |= output_buffer_cols.get(node.id, set())
    return exempt_rows, exempt_cols


def prune_ir_graph(
    ir_graph: IRGraph,
    zero_threshold: float = 1e-8,
    *,
    initial_pruned_per_node: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
    initial_pruned_per_bank: Dict[int, Tuple[Sequence[bool], Sequence[bool]]] | None = None,
) -> IRGraph:
    """Remove zeroed rows and columns from all NeuralCores in the IR graph.

    When initial_pruned_per_node or initial_pruned_per_bank are provided,
    uses those masks (True = pruned) as the initial pruned set. Propagation
    always runs: rows that only feed pruned cols and cols that only receive
    from pruned rows are pruned, with segment I/O indices exempt. When no
    initial masks are provided, infers initial set from matrix values below
    zero_threshold.

    Returns a new IRGraph with compacted cores and rewired sources.
    """
    if not ir_graph.nodes:
        return IRGraph(
            nodes=[],
            output_sources=ir_graph.output_sources.copy() if ir_graph.output_sources.size else ir_graph.output_sources,
            weight_banks=copy.deepcopy(ir_graph.weight_banks),
        )

    graph = copy.deepcopy(ir_graph)

    # Pre-Phase: record dead structure without mutating core_matrix.
    # Rows with input source -1 are dead; columns not read by any consumer are dead.
    # Phase 1 will merge these with model-mask or value-based zeros.
    read_sources: Set[Tuple[int, int]] = set()
    for src in graph.output_sources.flatten():
        node_id = getattr(src, "node_id", getattr(src, "core_", None))
        idx = getattr(src, "index", getattr(src, "neuron_", None))
        if node_id >= 0:
            read_sources.add((node_id, idx))
    for node in graph.nodes:
        if not hasattr(node, "input_sources"):
            continue
        for src in node.input_sources.flatten():
            node_id = getattr(src, "node_id", getattr(src, "core_", None))
            idx = getattr(src, "index", getattr(src, "neuron_", None))
            if node_id >= 0:
                read_sources.add((node_id, idx))

    prephase_dead_rows: Dict[int, Set[int]] = {}
    prephase_dead_cols: Dict[int, Set[int]] = {}
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        flat_src = node.input_sources.flatten()
        dead_r = {
            i for i, src in enumerate(flat_src)
            if getattr(src, "node_id", getattr(src, "core_", None)) == -1
        }
        n_neurons = node.core_matrix.shape[1]
        dead_c = {j for j in range(n_neurons) if (node.id, j) not in read_sources}
        if dead_r:
            prephase_dead_rows[node.id] = dead_r
        if dead_c:
            prephase_dead_cols[node.id] = dead_c

    # Phase 1: identify pruned rows/columns with exemption-aware propagation per core.
    input_buffer_rows, output_buffer_cols = compute_segment_io_exemption(graph)

    reindex_maps: Dict[int, Dict[int, int]] = {}
    pruned_neurons: Dict[int, Set[int]] = {}
    pruned_rows: Dict[int, Set[int]] = {}
    init_node = (initial_pruned_per_node or {})

    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix  # (axons, neurons)
        n_axons, n_neurons = mat.shape
        exempt_r = frozenset(input_buffer_rows.get(node.id, set()))
        exempt_c = frozenset(output_buffer_cols.get(node.id, set()))

        init = init_node.get(node.id)
        if init is not None:
            init_row = list(init[0][:n_axons]) + [False] * max(0, n_axons - len(init[0]))
            init_col = list(init[1][:n_neurons]) + [False] * max(0, n_neurons - len(init[1]))
            if len(init_row) == n_axons and len(init_col) == n_neurons:
                init_rows, init_cols = _masks_to_sets(init_row, init_col)
            else:
                init_rows, init_cols = None, None
        else:
            init_rows, init_cols = None, None

        # Merge pre-phase dead structure (never mutate matrix before this)
        pre_r = prephase_dead_rows.get(node.id, set())
        pre_c = prephase_dead_cols.get(node.id, set())
        if init_rows is not None:
            init_rows = init_rows | pre_r
            init_cols = init_cols | pre_c
        else:
            init_rows = pre_r if pre_r else None
            init_cols = pre_c if pre_c else None

        zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
            mat,
            zero_threshold=zero_threshold,
            initial_zero_rows=init_rows,
            initial_zero_cols=init_cols,
            exempt_rows=exempt_r,
            exempt_cols=exempt_c,
        )

        pruned_neurons[node.id] = zero_cols
        pruned_rows[node.id] = zero_rows

        new_idx = 0
        remap = {}
        for old_idx in range(n_neurons):
            if old_idx not in zero_cols:
                remap[old_idx] = new_idx
                new_idx += 1
        reindex_maps[node.id] = remap

    # Phase 2: rewire all sources
    def _rewire_sources(sources: np.ndarray) -> np.ndarray:
        """Rewire source references: pruned neurons -> off, others -> reindexed."""
        flat = sources.flatten()
        for i, src in enumerate(flat):
            if not isinstance(src, IRSource):
                continue
            if src.node_id < 0:
                continue
            if src.node_id in pruned_neurons:
                if src.index in pruned_neurons[src.node_id]:
                    flat[i] = IRSource(node_id=-1, index=0)
                elif src.node_id in reindex_maps and src.index in reindex_maps[src.node_id]:
                    flat[i] = IRSource(node_id=src.node_id, index=reindex_maps[src.node_id][src.index])
                elif src.node_id in reindex_maps:
                    raise ValueError(
                        f"prune_ir_graph: source index {src.index} for node_id {src.node_id} is neither "
                        "pruned nor in reindex map; bookkeeping error upstream."
                    )
        return flat.reshape(sources.shape)

    for node in graph.nodes:
        node.input_sources = _rewire_sources(node.input_sources)

    if graph.output_sources.size:
        graph.output_sources = _rewire_sources(graph.output_sources)

    # Fail fast if all output neurons were pruned (would cause empty output_sources after compaction)
    if graph.output_sources.size:
        flat = graph.output_sources.flatten()
        if all(
            isinstance(s, IRSource) and s.node_id < 0
            for s in flat
        ):
            raise ValueError(
                "prune_ir_graph: all output_sources were rewired to pruned (node_id<0). "
                "At least one output neuron must remain; check initial pruning masks and propagation."
            )

    # Before compacting: store pre-pruning heatmap and masks for GUI (soft-core viz)
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        mat = node.core_matrix
        n_axons, n_neurons = mat.shape
        zero_cols = pruned_neurons.get(node.id, set())
        zero_rows_set = pruned_rows.get(node.id, set())
        node.pre_pruning_heatmap = np.copy(mat).tolist()
        node.pruned_col_mask = [j in zero_cols for j in range(n_neurons)]
        node.pruned_row_mask = [r in zero_rows_set for r in range(n_axons)]

    # Phase 3: physically remove zeroed columns from core matrices
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        zero_cols = pruned_neurons.get(node.id, set())
        if zero_cols:
            keep_cols = [c for c in range(node.core_matrix.shape[1]) if c not in zero_cols]
            if keep_cols:
                node.core_matrix = node.core_matrix[:, keep_cols]
            else:
                # All columns pruned — keep a 1-column zero matrix to avoid empty
                node.core_matrix = node.core_matrix[:, :1] * 0.0

    # Phase 4: remove zeroed rows from core matrices and their input sources
    for node in graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        if node.core_matrix is None:
            continue

        mat = node.core_matrix
        n_axons = mat.shape[0]
        zero_rows_set = pruned_rows.get(node.id, set())
        keep_rows = [int(r) for r in range(n_axons) if r not in zero_rows_set]

        if len(keep_rows) < n_axons:
            flat_src = node.input_sources.flatten()
            if len(keep_rows) > 0:
                node.core_matrix = mat[keep_rows, :]
                node.input_sources = np.array(
                    [flat_src[r] for r in keep_rows], dtype=object
                )
            else:
                # All rows pruned — keep a 1-row zero matrix
                node.core_matrix = mat[:1, :] * 0.0
                node.input_sources = np.array([flat_src[0]], dtype=object)

    # After Phase 4: align mask lengths with post-compaction matrix for non–bank-backed cores.
    # (Bank-backed nodes get masks in Phase 5.) Post-compaction there are no pruned rows/cols left.
    # Save pre-compaction masks for GUI (pre_pruning_heatmap_image red markings) before overwriting.
    for node in graph.nodes:
        if not isinstance(node, NeuralCore) or node.core_matrix is None:
            continue
        if getattr(node, "weight_bank_id", None) is not None:
            continue
        mat = node.core_matrix
        n_axons_new, n_neurons_new = mat.shape[0], mat.shape[1]
        if node.pruned_row_mask is not None and node.pruned_col_mask is not None:
            node.pre_pruning_row_mask = list(node.pruned_row_mask)
            node.pre_pruning_col_mask = list(node.pruned_col_mask)
        node.pruned_row_mask = [False] * n_axons_new
        node.pruned_col_mask = [False] * n_neurons_new

    # Phase 5: set pruning maps on bank-backed cores from each weight bank
    # (banks are not compacted; masks let soft-core compaction drop rows/cols when materializing)
    # Uses same propagative helper as Phase 1. Apply segment I/O exemption at bank level.
    weight_banks = getattr(graph, "weight_banks", None) or {}
    init_bank = (initial_pruned_per_bank or {})
    for bank_id, bank in weight_banks.items():
        mat = bank.core_matrix
        n_axons, n_neurons = mat.shape
        init = init_bank.get(bank_id)
        bank_exempt_rows, bank_exempt_cols = _collect_bank_exemptions(
            graph, bank_id, input_buffer_rows, output_buffer_cols
        )
        exempt_r = frozenset(bank_exempt_rows)
        exempt_c = frozenset(bank_exempt_cols)
        if init is not None and len(init[0]) == n_axons and len(init[1]) == n_neurons:
            zero_rows, zero_cols = _masks_to_sets(init[0], init[1])
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=zero_threshold,
                initial_zero_rows=zero_rows,
                initial_zero_cols=zero_cols,
                exempt_rows=exempt_r,
                exempt_cols=exempt_c,
            )
        else:
            zero_rows, zero_cols = compute_propagated_pruned_rows_cols(
                mat,
                zero_threshold=zero_threshold,
                exempt_rows=exempt_r,
                exempt_cols=exempt_c,
            )
        pruned_row_mask = [i in zero_rows for i in range(n_axons)]
        pruned_col_mask_full = [j in zero_cols for j in range(n_neurons)]
        for node in graph.nodes:
            if not isinstance(node, NeuralCore) or getattr(node, "weight_bank_id", None) != bank_id:
                continue
            # Per-node effective matrix shape (same as get_core_matrix will return)
            if node.weight_row_slice is not None:
                start, end = node.weight_row_slice
                node.pre_pruning_heatmap = np.copy(bank.core_matrix[:, start:end]).tolist()
                node.pruned_col_mask = pruned_col_mask_full[start:end]
            else:
                node.pre_pruning_heatmap = np.copy(bank.core_matrix).tolist()
                node.pruned_col_mask = pruned_col_mask_full
            node.pruned_row_mask = pruned_row_mask

    return graph
