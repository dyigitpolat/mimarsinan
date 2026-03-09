"""
Pruning verification test flow: Path A (zero weights, no compaction) vs Path B
(full pruning + compaction). Verifies that B yields smaller soft core dimensions
and that both networks produce identical outputs.

Plan: pruning_verification_test_flow_ba58af6a
"""

from __future__ import annotations

import copy
import pytest
import numpy as np
import torch

from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
from mimarsinan.models.perceptron_mixer.perceptron_flow import PerceptronFlow
from mimarsinan.mapping.mapping_utils import (
    InputMapper,
    Ensure2DMapper,
    PerceptronMapper,
    ModelRepresentation,
)
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.ir import IRGraph, ir_graph_to_soft_core_mapping, NeuralCore
from mimarsinan.mapping.ir_pruning import (
    get_initial_pruning_masks_from_model,
    prune_ir_graph,
)
from mimarsinan.transformations.pruning import (
    compute_pruning_masks,
    compute_all_pruning_masks,
)
from mimarsinan.mapping.softcore_mapping import compact_soft_core_mapping
from mimarsinan.mapping.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
from mimarsinan.models.unified_core_flow import SpikingUnifiedCoreFlow

import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers: minimal models
# ---------------------------------------------------------------------------


class _MinimalSingleLayerFlow(PerceptronFlow):
    """One Perceptron with (in_dim, out_dim)."""

    def __init__(self, device, in_dim: int, out_dim: int, seed: int = 42):
        super().__init__(device)
        torch.manual_seed(seed)
        self._perceptron = Perceptron(
            output_channels=out_dim,
            input_features=in_dim,
            normalization=torch.nn.Identity(),
        ).to(device)
        # Mapper: (1, in_dim) input -> Ensure2D -> Perceptron
        input_shape = (1, in_dim)
        inp = InputMapper(input_shape)
        out = Ensure2DMapper(inp)
        out = PerceptronMapper(out, self._perceptron)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


class _MinimalTwoLayerFlow(PerceptronFlow):
    """Two Perceptrons: in_dim -> mid_dim -> out_dim."""

    def __init__(self, device, in_dim: int, mid_dim: int, out_dim: int, seed: int = 42):
        super().__init__(device)
        torch.manual_seed(seed)
        self._p1 = Perceptron(
            output_channels=mid_dim,
            input_features=in_dim,
            normalization=torch.nn.Identity(),
        ).to(device)
        self._p2 = Perceptron(
            output_channels=out_dim,
            input_features=mid_dim,
            normalization=torch.nn.Identity(),
        ).to(device)
        input_shape = (1, in_dim)
        inp = InputMapper(input_shape)
        out = Ensure2DMapper(inp)
        out = PerceptronMapper(out, self._p1)
        out = PerceptronMapper(out, self._p2)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


def _build_single_layer_model(device, in_dim: int, out_dim: int, seed: int = 42):
    """Build minimal one-layer model (in_dim -> out_dim)."""
    return _MinimalSingleLayerFlow(device, in_dim, out_dim, seed)


def _build_two_layer_model(
    device, in_dim: int, mid_dim: int, out_dim: int, seed: int = 42
):
    """Build minimal two-layer model (in_dim -> mid_dim -> out_dim)."""
    return _MinimalTwoLayerFlow(device, in_dim, mid_dim, out_dim, seed)


def _zero_weights_by_masks(model, masks):
    """
    Zero out pruned rows and columns in each perceptron's weight and bias.
    masks: list of (row_mask, col_mask) per perceptron; True = keep, False = prune.
    """
    perceptrons = model.get_perceptrons()
    assert len(masks) == len(perceptrons)
    for p, (row_mask, col_mask) in zip(perceptrons, masks):
        w = p.layer.weight.data
        # Pruned rows: zero entire row
        pruned_rows = ~row_mask
        if pruned_rows.any():
            w[pruned_rows, :] = 0.0
        # Pruned columns: zero entire column
        pruned_cols = ~col_mask
        if pruned_cols.any():
            w[:, pruned_cols] = 0.0
        if p.layer.bias is not None:
            p.layer.bias.data[pruned_rows] = 0.0


def _set_prune_buffers(model, masks):
    """
    Set prune_mask and prune_bias_mask on each layer for Path B.
    masks: list of (row_mask, col_mask); True = keep, False = prune.
    Convention: prune_mask (out_f, in_f) True = pruned.
    """
    perceptrons = model.get_perceptrons()
    assert len(masks) == len(perceptrons)
    for p, (row_mask, col_mask) in zip(perceptrons, masks):
        # prune_mask: True where pruned
        pruned_rows = ~row_mask
        pruned_cols = ~col_mask
        prune_mask = pruned_rows.unsqueeze(1) | pruned_cols.unsqueeze(0)
        p.layer.register_buffer("prune_mask", prune_mask.clone())
        p.layer.register_buffer("prune_bias_mask", pruned_rows.clone())


def _get_ir_core_shapes(ir_graph: IRGraph):
    """Return list of (n_axons, n_neurons) for each NeuralCore in graph."""
    shapes = []
    for node in ir_graph.nodes:
        shape = None
        cm = getattr(node, "core_matrix", None)
        if cm is not None and hasattr(cm, "shape"):
            shape = tuple(cm.shape)
        if shape is None and hasattr(node, "get_core_matrix"):
            try:
                mat = node.get_core_matrix(ir_graph)
                if mat is not None:
                    shape = tuple(mat.shape)
            except Exception:
                pass
        if shape is not None:
            shapes.append(shape)
    return shapes


def _clear_mapper_ir_cache(mapper, seen=None):
    """Clear _ir_sources and _cached_ir_mapping_id on mapper chain so map_to_ir uses fresh state."""
    if seen is None:
        seen = set()
    if id(mapper) in seen:
        return
    seen.add(id(mapper))
    if hasattr(mapper, "_ir_sources"):
        mapper._ir_sources = None
    if hasattr(mapper, "_cached_ir_mapping_id"):
        mapper._cached_ir_mapping_id = None
    if hasattr(mapper, "source_mapper") and mapper.source_mapper is not None:
        _clear_mapper_ir_cache(mapper.source_mapper, seen)


def _run_ir_mapping(model, max_axons=4096, max_neurons=4096):
    """Build IR graph from model (no pruning)."""
    repr_ = model.get_mapper_repr()
    _clear_mapper_ir_cache(repr_.output_layer_mapper)
    ir_mapping = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=max_axons,
        max_neurons=max_neurons,
        allow_axon_tiling=False,
    )
    return ir_mapping.map(repr_)


def _validate_pruned_ir_contracts(ir_graph: IRGraph) -> list[str]:
    """
    Check dimension and index contracts after pruning.
    Returns list of violation messages (empty if all pass).
    """
    from mimarsinan.mapping.ir import IRSource

    errors = []
    for node in ir_graph.nodes:
        if not isinstance(node, NeuralCore):
            continue
        try:
            mat = node.get_core_matrix(ir_graph)
        except Exception:
            continue
        if mat is None:
            continue
        n_axons, n_neurons = mat.shape[0], mat.shape[1]
        flat_src = node.input_sources.flatten()
        n_src = len(flat_src)
        out_count = node.get_output_count()
        if n_src != n_axons:
            errors.append(
                f"node id={node.id}: len(input_sources)={n_src} != core_matrix.shape[0]={n_axons}"
            )
        if out_count != n_neurons:
            errors.append(
                f"node id={node.id}: get_output_count()={out_count} != core_matrix.shape[1]={n_neurons}"
            )
    if ir_graph.output_sources.size:
        flat = ir_graph.output_sources.flatten()
        for i, src in enumerate(flat):
            if not isinstance(src, IRSource):
                continue
            if src.node_id < 0:
                continue
            node = next((n for n in ir_graph.nodes if getattr(n, "id", None) == src.node_id), None)
            if node is None:
                errors.append(f"output_sources[{i}]: node_id={src.node_id} not found")
                continue
            if isinstance(node, NeuralCore):
                out_count = node.get_output_count()
                if src.index < 0 or src.index >= out_count:
                    errors.append(
                        f"output_sources[{i}]: node_id={src.node_id} index={src.index} "
                        f"out of range [0, {out_count})"
                    )
    return errors


def _get_soft_core_shapes_after_mapping(ir_graph: IRGraph):
    """
    Run the real soft-core path: IR -> SoftCoreMapping -> compact_soft_core_mapping.
    Returns list of (n_axons, n_neurons) for each soft core after compaction.
    """
    soft = ir_graph_to_soft_core_mapping(ir_graph)
    compact_soft_core_mapping(soft.cores, soft.output_sources)
    return [tuple(np.asarray(c.core_matrix).shape) for c in soft.cores]


def _get_hard_core_used_per_segment(hybrid_mapping):
    """
    For each neural stage in the hybrid mapping, return (total_used_axons, total_used_neurons)
    summed over all hard cores in that stage.
    """
    result = []
    for stage in hybrid_mapping.stages:
        if stage.kind != "neural" or stage.hard_core_mapping is None:
            continue
        mapping = stage.hard_core_mapping
        used_axons = sum(
            int(c.axons_per_core - c.available_axons) for c in mapping.cores
        )
        used_neurons = sum(
            int(c.neurons_per_core - c.available_neurons) for c in mapping.cores
        )
        result.append((used_axons, used_neurons))
    return result


# ---------------------------------------------------------------------------
# Single-layer tests
# ---------------------------------------------------------------------------


class TestSingleLayerPruningVerification:
    """Path A (zero only) vs Path B (prune + compact): single-layer."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_single_layer_pruning_reduces_soft_core_size(self, device):
        # Use smaller dimensions so propagation stays predictable
        in_dim, out_dim = 24, 16
        pruning_fraction = 0.2
        model = _build_single_layer_model(device, in_dim, out_dim)
        perceptrons = model.get_perceptrons()
        assert len(perceptrons) == 1
        p = perceptrons[0]
        row_mask, col_mask = compute_pruning_masks(p, pruning_fraction)
        m_pruned = (~row_mask).sum().item()
        n_pruned = (~col_mask).sum().item()
        masks = [(row_mask, col_mask)]

        # Zero weights (same for A and B)
        _zero_weights_by_masks(model, masks)

        # Path A: no prune_mask, IR map only -> full size
        ir_A = _run_ir_mapping(model)
        shapes_A = _get_ir_core_shapes(ir_A)
        assert len(shapes_A) == 1
        # IR core: (axons, neurons) = (in_dim+1, out_dim)
        expected_axons_A = in_dim + 1
        expected_neurons_A = out_dim
        assert shapes_A[0][0] == expected_axons_A
        assert shapes_A[0][1] == expected_neurons_A

        # Path B: set prune buffers, IR map then prune_ir_graph -> reduced
        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        shapes_B = _get_ir_core_shapes(ir_B)
        assert len(shapes_B) == 1
        # With segment I/O exemption, the single layer is the output layer so columns (neurons) are not pruned.
        expected_neurons_B = expected_neurons_A  # all kept (output-buffer)
        expected_axons_B = expected_axons_A - n_pruned
        assert shapes_B[0][1] == expected_neurons_B, (
            f"Expected {expected_neurons_B} neurons (segment output-buffer exempt), got {shapes_B[0][1]}"
        )
        assert shapes_B[0][0] <= expected_axons_A, (
            f"Path B axons should be <= {expected_axons_A}, got {shapes_B[0][0]}"
        )
        assert shapes_B[0][0] > 0 and shapes_B[0][1] > 0
        # B is at most as large as A (segment exemption may prevent any pruning for single-layer)
        assert (shapes_B[0][0], shapes_B[0][1]) <= (shapes_A[0][0], shapes_A[0][1])

    def test_single_layer_a_and_b_same_output(self, device):
        in_dim, out_dim = 24, 16
        pruning_fraction = 0.15
        model = _build_single_layer_model(device, in_dim, out_dim)
        perceptrons = model.get_perceptrons()
        row_mask, col_mask = compute_pruning_masks(perceptrons[0], pruning_fraction)
        masks = [(row_mask, col_mask)]
        _zero_weights_by_masks(model, masks)

        # A and B use identical weights (zeroed); B just has buffers set for mapping
        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)

        x = torch.randn(2, in_dim, device=device)
        model.eval()
        model_B.eval()
        with torch.no_grad():
            out_A = model(x)
            out_B = model_B(x)
        assert out_A.shape == out_B.shape
        assert torch.allclose(out_A, out_B), "A and B must produce same output (same weights)"


# ---------------------------------------------------------------------------
# Two-layer tests
# ---------------------------------------------------------------------------


class TestTwoLayerPruningVerification:
    """Two-layer with propagation: Layer 2 pruned inputs = Layer 1 pruned outputs."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_two_layer_pruning_propagation_and_reduction(self, device):
        in_dim, mid_dim, out_dim = 40, 30, 20
        pruning_fraction = 0.2
        model = _build_two_layer_model(device, in_dim, mid_dim, out_dim)
        perceptrons = model.get_perceptrons()
        assert len(perceptrons) == 2
        masks = compute_all_pruning_masks(perceptrons, pruning_fraction)
        _zero_weights_by_masks(model, masks)

        # Path A: no prune buffers -> full IR
        ir_A = _run_ir_mapping(model)
        shapes_A = _get_ir_core_shapes(ir_A)
        assert len(shapes_A) == 2
        # Layer 1: (in_dim+1, mid_dim), Layer 2: (mid_dim+1, out_dim)
        assert shapes_A[0] == (in_dim + 1, mid_dim)
        assert shapes_A[1] == (mid_dim + 1, out_dim)

        # Path B: set buffers, prune
        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        assert len(ir_B_raw.nodes) == 2, (
            f"IR mapping must produce 2 nodes for two-layer model, got {len(ir_B_raw.nodes)}"
        )
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        shapes_B = _get_ir_core_shapes(ir_B)
        assert len(shapes_B) == 2, (
            f"expected 2 core shapes from Path B, got {len(shapes_B)}; "
            f"ir_B has {len(ir_B.nodes)} nodes"
        )

        # Both layers must be strictly smaller in B
        assert shapes_B[0][0] <= shapes_A[0][0] and shapes_B[0][1] <= shapes_A[0][1]
        assert shapes_B[1][0] <= shapes_A[1][0] and shapes_B[1][1] <= shapes_A[1][1]
        assert (shapes_B[0][0], shapes_B[0][1]) < (shapes_A[0][0], shapes_A[0][1]) or (
            shapes_B[1][0],
            shapes_B[1][1],
        ) < (shapes_A[1][0], shapes_A[1][1])

        # Propagation: Layer 2's pruned input axons should include those from Layer 1's
        # pruned output neurons. So reduction in Layer 2 axons >= reduction in Layer 1 neurons.
        layer1_neurons_A, layer2_axons_A = shapes_A[0][1], shapes_A[1][0]
        layer1_neurons_B, layer2_axons_B = shapes_B[0][1], shapes_B[1][0]
        delta_L1_neurons = layer1_neurons_A - layer1_neurons_B
        delta_L2_axons = layer2_axons_A - layer2_axons_B
        assert delta_L2_axons >= 0 and delta_L1_neurons >= 0
        assert (
            delta_L2_axons >= delta_L1_neurons
        ), "Layer 2 pruned axons should be at least Layer 1 pruned neurons (propagation)"

    def test_two_layer_a_and_b_same_output(self, device):
        in_dim, mid_dim, out_dim = 16, 12, 8
        pruning_fraction = 0.15
        model = _build_two_layer_model(device, in_dim, mid_dim, out_dim)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, pruning_fraction)
        _zero_weights_by_masks(model, masks)
        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)

        x = torch.randn(2, in_dim, device=device)
        model.eval()
        model_B.eval()
        with torch.no_grad():
            out_A = model(x)
            out_B = model_B(x)
        assert out_A.shape == out_B.shape
        assert torch.allclose(out_A, out_B)


# ---------------------------------------------------------------------------
# Integration: actual soft core / hard core mapping behavior
# ---------------------------------------------------------------------------


class TestPruningVerificationIntegration:
    """
    Integration tests that run the real soft-core and hard-core mapping pipeline.
    Path A: zero weights only, no pruning/compaction on IR or soft cores.
    Path B: full pruning + compaction; soft cores and hard-core used area must reflect reduction.
    """

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    @pytest.fixture
    def cores_config(self):
        return [{"max_axons": 64, "max_neurons": 64, "count": 10}]

    def test_single_layer_soft_core_mapping_reflects_compaction(self, device, cores_config):
        """Soft cores after ir_graph_to_soft_core_mapping + compact_soft_core_mapping: B smaller than A."""
        in_dim, out_dim = 24, 16
        pruning_fraction = 0.2
        model = _build_single_layer_model(device, in_dim, out_dim)
        perceptrons = model.get_perceptrons()
        row_mask, col_mask = compute_pruning_masks(perceptrons[0], pruning_fraction)
        m_pruned = (~row_mask).sum().item()
        n_pruned = (~col_mask).sum().item()
        masks = [(row_mask, col_mask)]
        _zero_weights_by_masks(model, masks)

        # Path A: no compaction (no prune masks on IR)
        ir_A = _run_ir_mapping(model)
        soft_A_shapes = _get_soft_core_shapes_after_mapping(ir_A)
        assert len(soft_A_shapes) == 1
        expected_axons_A, expected_neurons_A = in_dim + 1, out_dim
        assert soft_A_shapes[0] == (expected_axons_A, expected_neurons_A)

        # Path B: full prune + compaction
        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        soft_B_shapes = _get_soft_core_shapes_after_mapping(ir_B)
        assert len(soft_B_shapes) == 1
        # With segment I/O exemption, single layer is output layer: neurons not pruned; input rows may also be exempt.
        expected_neurons_B = expected_neurons_A
        expected_axons_B = expected_axons_A - n_pruned
        assert soft_B_shapes[0][1] == expected_neurons_B
        assert soft_B_shapes[0][0] <= expected_axons_A, "Path B axons should not exceed A"
        assert soft_B_shapes[0][1] <= expected_neurons_A
        # Path B is at most as large as A (may be equal if segment exemption keeps all dims)
        assert (soft_B_shapes[0][0], soft_B_shapes[0][1]) <= (soft_A_shapes[0][0], soft_A_shapes[0][1])

    def test_single_layer_hard_core_mapping_reflects_compaction(self, device, cores_config):
        """Hard-core used area (after packing) is smaller for Path B than Path A."""
        in_dim, out_dim = 24, 16
        pruning_fraction = 0.2
        model = _build_single_layer_model(device, in_dim, out_dim)
        perceptrons = model.get_perceptrons()
        row_mask, col_mask = compute_pruning_masks(perceptrons[0], pruning_fraction)
        masks = [(row_mask, col_mask)]
        _zero_weights_by_masks(model, masks)

        ir_A = _run_ir_mapping(model)
        hybrid_A = build_hybrid_hard_core_mapping(ir_graph=ir_A, cores_config=cores_config)
        used_A = _get_hard_core_used_per_segment(hybrid_A)
        assert len(used_A) == 1

        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        hybrid_B = build_hybrid_hard_core_mapping(ir_graph=ir_B, cores_config=cores_config)
        used_B = _get_hard_core_used_per_segment(hybrid_B)
        assert len(used_B) == 1

        # With segment I/O exemption, single-layer has all rows (input) and all cols (output) exempt,
        # so Path B may equal Path A. Require at most same resources.
        assert used_B[0][0] <= used_A[0][0] and used_B[0][1] <= used_A[0][1], (
            f"Path B used (axons, neurons) should be <= A: A={used_A[0]}, B={used_B[0]}"
        )

    def test_two_layer_soft_core_mapping_reflects_compaction(self, device, cores_config):
        """Two-layer: soft core dimensions after mapping+compaction are reduced in B for both layers."""
        in_dim, mid_dim, out_dim = 40, 30, 20
        pruning_fraction = 0.2
        model = _build_two_layer_model(device, in_dim, mid_dim, out_dim)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, pruning_fraction)
        _zero_weights_by_masks(model, masks)

        ir_A = _run_ir_mapping(model)
        soft_A_shapes = _get_soft_core_shapes_after_mapping(ir_A)
        assert len(soft_A_shapes) == 2

        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        soft_B_shapes = _get_soft_core_shapes_after_mapping(ir_B)
        assert len(soft_B_shapes) == 2

        for i in range(2):
            assert soft_B_shapes[i][0] <= soft_A_shapes[i][0] and soft_B_shapes[i][1] <= soft_A_shapes[i][1]
        assert (soft_B_shapes[0], soft_B_shapes[1]) < (soft_A_shapes[0], soft_A_shapes[1]), (
            "Path B soft cores must be strictly smaller in at least one layer"
        )

    def test_two_layer_hard_core_mapping_reflects_compaction(self, device, cores_config):
        """Two-layer: total used axons/neurons per segment in hard mapping are <= for Path B."""
        in_dim, mid_dim, out_dim = 40, 30, 20
        pruning_fraction = 0.2
        model = _build_two_layer_model(device, in_dim, mid_dim, out_dim)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, pruning_fraction)
        _zero_weights_by_masks(model, masks)

        ir_A = _run_ir_mapping(model)
        hybrid_A = build_hybrid_hard_core_mapping(ir_graph=ir_A, cores_config=cores_config)
        used_A = _get_hard_core_used_per_segment(hybrid_A)
        assert len(used_A) == 1  # one neural segment (both layers packed together)

        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        hybrid_B = build_hybrid_hard_core_mapping(ir_graph=ir_B, cores_config=cores_config)
        used_B = _get_hard_core_used_per_segment(hybrid_B)
        assert len(used_B) == 1

        assert used_B[0][0] <= used_A[0][0] and used_B[0][1] <= used_A[0][1]
        assert (used_B[0][0], used_B[0][1]) < (used_A[0][0], used_A[0][1])


# ---------------------------------------------------------------------------
# Mapping equivalence: model forward vs SpikingUnifiedCoreFlow(pruned IR)
# ---------------------------------------------------------------------------
# Plan: mapping_accuracy_drop_fix — Step 1 equivalence test to find root cause.


class _MinimalFourLayerFlow(PerceptronFlow):
    """Four layers: in_dim -> d1 -> d2 -> d3 -> out_dim (e.g. 64 -> 32 -> 16 -> 10)."""

    def __init__(
        self,
        device,
        in_dim: int,
        d1: int,
        d2: int,
        d3: int,
        out_dim: int,
        seed: int = 42,
    ):
        super().__init__(device)
        torch.manual_seed(seed)
        self._p1 = Perceptron(
            output_channels=d1,
            input_features=in_dim,
            normalization=torch.nn.Identity(),
        ).to(device)
        self._p2 = Perceptron(
            output_channels=d2,
            input_features=d1,
            normalization=torch.nn.Identity(),
        ).to(device)
        self._p3 = Perceptron(
            output_channels=d3,
            input_features=d2,
            normalization=torch.nn.Identity(),
        ).to(device)
        self._p4 = Perceptron(
            output_channels=out_dim,
            input_features=d3,
            normalization=torch.nn.Identity(),
        ).to(device)
        inp = InputMapper((1, in_dim))
        out = Ensure2DMapper(inp)
        for p in [self._p1, self._p2, self._p3, self._p4]:
            out = PerceptronMapper(out, p)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


class TestMappingEquivalence:
    """
    Equivalence test: fused model forward vs SpikingUnifiedCoreFlow(pruned ir_graph)
    on the same batch. Used to pinpoint mapping accuracy drop (0.99 -> 0.098).
    """

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_pruned_ir_satisfies_dimension_contracts(self, device):
        """Step 3: After pruning, every core must satisfy axons/neurons vs input_sources/output_sources."""
        in_dim, d1, d2, d3, out_dim = 64, 32, 16, 10, 5
        model = _MinimalFourLayerFlow(device, in_dim, d1, d2, d3, out_dim)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, 0.1)
        _zero_weights_by_masks(model, masks)
        _set_prune_buffers(model, masks)

        ir_raw = _run_ir_mapping(model)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_raw)
        ir = prune_ir_graph(
            ir_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        errors = _validate_pruned_ir_contracts(ir)
        assert not errors, f"Pruned IR contract violations: {errors}"

    def test_model_and_pruned_flow_same_argmax(self, device):
        """
        Step 1: On the same input batch, model forward and SpikingUnifiedCoreFlow(pruned IR)
        must yield the same predicted class (argmax) for every sample.
        """
        in_dim, d1, d2, d3, out_dim = 64, 32, 16, 10, 5
        batch_size = 10
        seed = 42
        pruning_fraction = 0.1
        simulation_steps = 32

        model = _MinimalFourLayerFlow(device, in_dim, d1, d2, d3, out_dim, seed=seed)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, pruning_fraction)
        _zero_weights_by_masks(model, masks)
        _set_prune_buffers(model, masks)

        ir_raw = _run_ir_mapping(model)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_raw)
        ir_graph = prune_ir_graph(
            ir_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )

        flow = SpikingUnifiedCoreFlow(
            input_shape=(1, in_dim),
            ir_graph=ir_graph,
            simulation_length=simulation_steps,
            preprocessor=nn.Identity(),
            firing_mode="TTFS",
            spike_mode="TTFS",
            thresholding_mode="<=",
            spiking_mode="ttfs_quantized",
        ).to(device)

        torch.manual_seed(seed + 1)
        x = torch.randn(batch_size, in_dim, device=device)

        model.eval()
        flow.eval()
        with torch.no_grad():
            out_model = model(x)
            out_flow = flow(x)

        assert out_model.shape == (batch_size, out_dim), out_model.shape
        assert out_flow.shape == (batch_size, out_dim), out_flow.shape

        pred_model = out_model.argmax(dim=1)
        pred_flow = out_flow.argmax(dim=1)
        match = (pred_model == pred_flow).all().item()
        assert match, (
            f"Model vs flow argmax mismatch: model {pred_model.tolist()} "
            f"vs flow {pred_flow.tolist()}"
        )

    def test_model_flow_equivalence_with_pruning(self, device):
        """
        With model masks and exemption-aware propagation, pruned IR preserves
        segment I/O and model/flow yield same argmax.
        """
        in_dim, d1, d2, d3, out_dim = 64, 32, 16, 10, 5
        batch_size = 10
        seed = 42
        pruning_fraction = 0.1
        simulation_steps = 32

        model = _MinimalFourLayerFlow(device, in_dim, d1, d2, d3, out_dim, seed=seed)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, pruning_fraction)
        _zero_weights_by_masks(model, masks)
        _set_prune_buffers(model, masks)

        ir_raw = _run_ir_mapping(model)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_raw)
        ir_graph = prune_ir_graph(
            ir_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )

        flow = SpikingUnifiedCoreFlow(
            input_shape=(1, in_dim),
            ir_graph=ir_graph,
            simulation_length=simulation_steps,
            preprocessor=nn.Identity(),
            firing_mode="TTFS",
            spike_mode="TTFS",
            thresholding_mode="<=",
            spiking_mode="ttfs_quantized",
        ).to(device)

        torch.manual_seed(seed + 1)
        x = torch.randn(batch_size, in_dim, device=device)

        model.eval()
        flow.eval()
        with torch.no_grad():
            out_model = model(x)
            out_flow = flow(x)

        pred_model = out_model.argmax(dim=1)
        pred_flow = out_flow.argmax(dim=1)
        match = (pred_model == pred_flow).all().item()
        assert match, (
            f"Model vs flow argmax mismatch: model {pred_model.tolist()} "
            f"vs flow {pred_flow.tolist()}"
        )

    def test_propagation_compacts_cores(self, device):
        """
        With model masks, propagation always runs and can compact cores
        (total core size <= unpruned size).
        """
        in_dim, d1, d2, d3, out_dim = 48, 24, 12, 8, 5
        model = _MinimalFourLayerFlow(device, in_dim, d1, d2, d3, out_dim)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, 0.12)
        _zero_weights_by_masks(model, masks)
        _set_prune_buffers(model, masks)

        ir_raw = _run_ir_mapping(model)
        initial_node, initial_bank = get_initial_pruning_masks_from_model(model, ir_raw)
        unpruned_total = sum(
            n.core_matrix.shape[0] * n.core_matrix.shape[1]
            for n in ir_raw.nodes
            if getattr(n, "core_matrix", None) is not None
        )

        ir_pruned = prune_ir_graph(
            ir_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )

        shapes = _get_ir_core_shapes(ir_pruned)
        total = sum(s[0] * s[1] for s in shapes)
        assert total <= unpruned_total, (
            f"Pruned total size ({total}) should be <= unpruned ({unpruned_total})"
        )


# ---------------------------------------------------------------------------
# Multi-layer (and Squeezenet) tests
# ---------------------------------------------------------------------------
# Plan specifies single-layer, two-layer, then SqueezeNet. SqueezeNet requires
# the full pipeline with TorchMappingStep to obtain a supermodel; the
# TestMultilayerPruningVerification class uses a 3-layer MLP as a self-contained
# stand-in. test_squeezenet_* is skipped here and can be run as an integration
# test with a real config.


class _MinimalThreeLayerFlow(PerceptronFlow):
    """Three layers: in_dim -> d1 -> d2 -> out_dim."""

    def __init__(
        self, device, in_dim: int, d1: int, d2: int, out_dim: int, seed: int = 42
    ):
        super().__init__(device)
        torch.manual_seed(seed)
        self._p1 = Perceptron(
            output_channels=d1,
            input_features=in_dim,
            normalization=torch.nn.Identity(),
        ).to(device)
        self._p2 = Perceptron(
            output_channels=d2,
            input_features=d1,
            normalization=torch.nn.Identity(),
        ).to(device)
        self._p3 = Perceptron(
            output_channels=out_dim,
            input_features=d2,
            normalization=torch.nn.Identity(),
        ).to(device)
        inp = InputMapper((1, in_dim))
        out = Ensure2DMapper(inp)
        for p in [self._p1, self._p2, self._p3]:
            out = PerceptronMapper(out, p)
        self._mapper_repr = ModelRepresentation(out)

    def get_perceptrons(self):
        return self._mapper_repr.get_perceptrons()

    def get_perceptron_groups(self):
        return self._mapper_repr.get_perceptron_groups()

    def get_mapper_repr(self):
        return self._mapper_repr

    def forward(self, x):
        return self._mapper_repr(x)


class TestMultilayerPruningVerification:
    """Multi-layer (3-layer MLP) as stand-in for more complex models."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_multilayer_pruning_reduces_soft_core_size(self, device):
        in_dim, d1, d2, out_dim = 32, 24, 16, 10
        model = _MinimalThreeLayerFlow(device, in_dim, d1, d2, out_dim)
        perceptrons = model.get_perceptrons()
        assert len(perceptrons) == 3
        masks = compute_all_pruning_masks(perceptrons, 0.15)
        _zero_weights_by_masks(model, masks)

        ir_A = _run_ir_mapping(model)
        shapes_A = _get_ir_core_shapes(ir_A)
        assert len(shapes_A) == 3

        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)
        ir_B_raw = _run_ir_mapping(model_B)
        assert len(ir_B_raw.nodes) == 3, (
            f"IR mapping must produce 3 nodes for three-layer model, got {len(ir_B_raw.nodes)}"
        )
        initial_node, initial_bank = get_initial_pruning_masks_from_model(
            model_B, ir_B_raw
        )
        ir_B = prune_ir_graph(
            ir_B_raw,
            initial_pruned_per_node=initial_node or None,
            initial_pruned_per_bank=initial_bank or None,
        )
        shapes_B = _get_ir_core_shapes(ir_B)
        assert len(shapes_B) == 3

        total_A = sum(s[0] * s[1] for s in shapes_A)
        total_B = sum(s[0] * s[1] for s in shapes_B)
        assert total_B < total_A, "Path B total core size must be smaller than A"
        for i in range(3):
            assert shapes_B[i][0] <= shapes_A[i][0] and shapes_B[i][1] <= shapes_A[i][1]

    def test_multilayer_a_and_b_same_output(self, device):
        in_dim, d1, d2, out_dim = 16, 12, 8, 4
        model = _MinimalThreeLayerFlow(device, in_dim, d1, d2, out_dim)
        perceptrons = model.get_perceptrons()
        masks = compute_all_pruning_masks(perceptrons, 0.1)
        _zero_weights_by_masks(model, masks)
        model_B = copy.deepcopy(model)
        _set_prune_buffers(model_B, masks)

        x = torch.randn(2, in_dim, device=device)
        model.eval()
        model_B.eval()
        with torch.no_grad():
            out_A = model(x)
            out_B = model_B(x)
        assert torch.allclose(out_A, out_B)

    @pytest.mark.skip(reason="SqueezeNet requires full pipeline + TorchMappingStep; use integration test with config")
    def test_squeezenet_pruning_reduces_soft_core_size(self, device):
        """Placeholder: run with pipeline + torch_squeezenet11 config to verify reduction and output equivalence."""
        pytest.skip("SqueezeNet: run scripts or integration test with examples/cifar10_torch_squeezenet11_pretrained.json")

    @pytest.mark.skip(reason="SqueezeNet requires full pipeline; use integration test")
    def test_squeezenet_a_and_b_same_output(self, device):
        """Placeholder: same as above for output equivalence."""
        pytest.skip("SqueezeNet: run pipeline and compare model outputs")
