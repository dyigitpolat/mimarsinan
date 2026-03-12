"""End-to-end tests for the hardware_bias chain.

Validates that bias flows correctly through:
  IRMapping (hardware_bias=True) → NeuralCore → SoftCore → HardCore → ChipModel

and is never encoded as an always-on axon row when hardware_bias mode is active.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.ir_mapping import IRMapping
from mimarsinan.mapping.softcore_mapping import (
    HardCore,
    HardCoreMapping,
    SoftCore,
    compact_soft_core_mapping,
)
from mimarsinan.mapping.ir import neural_core_to_soft_core
from mimarsinan.mapping.mapping_utils import hard_cores_to_chip
from mimarsinan.code_generation.cpp_chip_model import SpikeSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inp(idx):
    return IRSource(node_id=-2, index=idx)


def _on():
    return IRSource(node_id=-3, index=0)


def _make_neural_core_with_hw_bias(node_id, in_features, out_features, *, seed=0):
    """Create a NeuralCore with hardware_bias (no always-on row)."""
    rng = np.random.RandomState(seed)
    weights = rng.randn(in_features, out_features).astype(np.float64)
    bias = rng.randn(out_features).astype(np.float64)
    input_sources = np.array([_inp(i) for i in range(in_features)])
    return NeuralCore(
        id=node_id,
        name=f"hw_bias_core_{node_id}",
        input_sources=input_sources,
        core_matrix=weights,
        hardware_bias=bias,
        threshold=1.0,
        latency=0,
    )


def _make_neural_core_legacy(node_id, in_features, out_features, *, seed=0):
    """Create a NeuralCore with legacy always-on bias row."""
    rng = np.random.RandomState(seed)
    weights = rng.randn(in_features, out_features).astype(np.float64)
    bias = rng.randn(1, out_features).astype(np.float64)
    core_matrix = np.vstack([weights, bias])
    input_sources = np.array([_inp(i) for i in range(in_features)] + [_on()])
    return NeuralCore(
        id=node_id,
        name=f"legacy_core_{node_id}",
        input_sources=input_sources,
        core_matrix=core_matrix,
        hardware_bias=None,
        threshold=1.0,
        latency=0,
    )


# ---------------------------------------------------------------------------
# IRMapping creates hardware_bias correctly
# ---------------------------------------------------------------------------

class TestIRMappingHardwareBias:
    """IRMapping with hardware_bias=True produces NeuralCores without always-on rows."""

    def _map_fc(self, mapper, in_features, out_features, biases):
        """Helper to call map_fc with the correct API and return the created NeuralCore."""
        weights = torch.randn(out_features, in_features)
        input_sources = np.array([_inp(i) for i in range(in_features)])
        output_shape = np.array([out_features])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=output_shape,
            fc_weights=weights,
            fc_biases=biases,
            name="test_fc",
        )
        # The created NeuralCore is the last node in mapper.nodes
        return mapper.nodes[-1]

    def test_no_always_on_source(self):
        mapper = IRMapping(hardware_bias=True)
        biases = torch.randn(4)
        core = self._map_fc(mapper, in_features=3, out_features=4, biases=biases)

        for src in core.input_sources.flatten():
            assert not src.is_always_on(), "hardware_bias mode should not create always-on sources"

    def test_hardware_bias_array_set(self):
        mapper = IRMapping(hardware_bias=True)
        biases = torch.randn(4)
        core = self._map_fc(mapper, in_features=3, out_features=4, biases=biases)

        assert core.hardware_bias is not None
        np.testing.assert_allclose(core.hardware_bias, biases.numpy(), atol=1e-6)

    def test_core_matrix_has_no_extra_row(self):
        mapper = IRMapping(hardware_bias=True)
        in_features, out_features = 5, 3
        core = self._map_fc(mapper, in_features, out_features, biases=torch.randn(out_features))

        assert core.core_matrix.shape == (in_features, out_features)

    def test_legacy_mode_has_always_on_row(self):
        mapper = IRMapping(hardware_bias=False)
        in_features, out_features = 5, 3
        core = self._map_fc(mapper, in_features, out_features, biases=torch.randn(out_features))

        assert core.hardware_bias is None
        assert core.core_matrix.shape == (in_features + 1, out_features)
        assert core.input_sources[-1].is_always_on()


# ---------------------------------------------------------------------------
# NeuralCore → SoftCore conversion
# ---------------------------------------------------------------------------

class TestNeuralCoreToSoftCore:
    """neural_core_to_soft_core preserves hardware_bias and doesn't create always-on rows."""

    def test_hw_bias_passed_through(self):
        nc = _make_neural_core_with_hw_bias(0, in_features=4, out_features=3)
        sc = neural_core_to_soft_core(nc)

        assert sc.hardware_bias is not None
        np.testing.assert_array_equal(sc.hardware_bias, nc.hardware_bias)

    def test_no_always_on_in_axon_sources(self):
        nc = _make_neural_core_with_hw_bias(0, in_features=4, out_features=3)
        sc = neural_core_to_soft_core(nc)

        for src in sc.axon_sources:
            assert not getattr(src, "is_always_on_", False), \
                "SoftCore should not have always-on axon when hardware_bias is set"

    def test_core_matrix_shape_preserved(self):
        nc = _make_neural_core_with_hw_bias(0, in_features=4, out_features=3)
        sc = neural_core_to_soft_core(nc)

        assert sc.core_matrix.shape == (4, 3)

    def test_legacy_always_on_preserved(self):
        nc = _make_neural_core_legacy(0, in_features=4, out_features=3)
        sc = neural_core_to_soft_core(nc)

        assert sc.hardware_bias is None
        assert sc.core_matrix.shape == (5, 3)  # 4 inputs + 1 bias row
        assert sc.axon_sources[-1].is_always_on_


# ---------------------------------------------------------------------------
# SoftCore → HardCore packing
# ---------------------------------------------------------------------------

class TestHardCoreBiasPacking:
    """HardCore.add_softcore correctly accumulates hardware_bias."""

    def test_single_softcore_bias_copied(self):
        bias = np.array([0.1, 0.2, 0.3])
        sc = SoftCore(
            core_matrix=np.eye(3),
            axon_sources=[SpikeSource(-2, i, is_input=True, is_off=False) for i in range(3)],
            id=0,
        )
        sc.hardware_bias = bias

        hc = HardCore(axons_per_core=8, neurons_per_core=8)
        hc.add_softcore(sc)

        assert hc.hardware_bias is not None
        np.testing.assert_array_equal(hc.hardware_bias[:3], bias)
        # Remaining neurons should be zero-padded
        np.testing.assert_array_equal(hc.hardware_bias[3:], 0.0)

    def test_two_softcores_bias_sliced_correctly(self):
        bias1 = np.array([1.0, 2.0])
        bias2 = np.array([3.0, 4.0])

        sc1 = SoftCore(
            core_matrix=np.ones((2, 2)),
            axon_sources=[SpikeSource(-2, 0, is_input=True, is_off=False)] * 2,
            id=0,
        )
        sc1.hardware_bias = bias1

        sc2 = SoftCore(
            core_matrix=np.ones((2, 2)),
            axon_sources=[SpikeSource(-2, 0, is_input=True, is_off=False)] * 2,
            id=1,
        )
        sc2.hardware_bias = bias2

        hc = HardCore(axons_per_core=4, neurons_per_core=4)
        hc.threshold = 1.0
        hc.add_softcore(sc1)
        hc.add_softcore(sc2)

        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(hc.hardware_bias, expected)

    def test_no_bias_softcore_leaves_hardcore_none(self):
        sc = SoftCore(
            core_matrix=np.eye(3),
            axon_sources=[SpikeSource(-2, i, is_input=True, is_off=False) for i in range(3)],
            id=0,
        )
        # sc.hardware_bias is None by default

        hc = HardCore(axons_per_core=4, neurons_per_core=4)
        hc.add_softcore(sc)

        assert hc.hardware_bias is None


# ---------------------------------------------------------------------------
# compact_soft_core_mapping compacts hardware_bias
# ---------------------------------------------------------------------------

class TestCompactHardwareBias:
    """compact_soft_core_mapping compacts hardware_bias alongside columns."""

    def test_bias_compacted_with_pruned_cols(self):
        mat = np.array([
            [1.0, 0.0, 3.0],
            [4.0, 0.0, 6.0],
        ])
        bias = np.array([0.1, 0.2, 0.3])

        sc = SoftCore(
            core_matrix=mat,
            axon_sources=[
                SpikeSource(-2, 0, is_input=True, is_off=False),
                SpikeSource(-2, 1, is_input=True, is_off=False),
            ],
            id=0,
        )
        sc.hardware_bias = bias
        sc.pruned_row_mask = [False, False]
        sc.pruned_col_mask = [False, True, False]  # col 1 pruned

        output_sources = [SpikeSource(0, 0, is_input=False, is_off=False),
                          SpikeSource(0, 2, is_input=False, is_off=False)]

        compact_soft_core_mapping([sc], output_sources)

        # Col 1 removed → bias should be [0.1, 0.3]
        np.testing.assert_array_equal(sc.hardware_bias, [0.1, 0.3])
        assert sc.core_matrix.shape == (2, 2)

    def test_bias_unchanged_when_no_pruning(self):
        mat = np.eye(3)
        bias = np.array([1.0, 2.0, 3.0])

        sc = SoftCore(
            core_matrix=mat,
            axon_sources=[SpikeSource(-2, i, is_input=True, is_off=False) for i in range(3)],
            id=0,
        )
        sc.hardware_bias = bias.copy()

        output_sources = [SpikeSource(0, i, is_input=False, is_off=False) for i in range(3)]
        compact_soft_core_mapping([sc], output_sources)

        np.testing.assert_array_equal(sc.hardware_bias, bias)


# ---------------------------------------------------------------------------
# HardCore → ChipModel (hard_cores_to_chip Mode 1)
# ---------------------------------------------------------------------------

class TestHardCoresToChipHardwareBias:
    """hard_cores_to_chip Mode 1: hardware_bias → per-neuron bias, no always-on row."""

    def test_bias_emitted_as_neuron_bias(self):
        """hardware_bias values appear in Neuron.bias fields."""
        hw_bias = np.array([0.5, -0.3])
        weight_mat = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
        ], dtype=np.float64)

        hc = HardCore(axons_per_core=2, neurons_per_core=2, has_bias_capability=True)
        hc.core_matrix = weight_mat
        hc.hardware_bias = hw_bias
        hc.axon_sources = [
            SpikeSource(-2, 0, is_input=True, is_off=False),
            SpikeSource(-2, 1, is_input=True, is_off=False),
        ]
        hc.threshold = 1.0
        hc.latency = 0

        mapping = HardCoreMapping(chip_cores=[])
        mapping.cores = [hc]
        mapping.output_sources = np.array([
            SpikeSource(0, 0, is_input=False, is_off=False),
            SpikeSource(0, 1, is_input=False, is_off=False),
        ])

        chip = hard_cores_to_chip(
            input_size=2, hardcore_mapping=mapping,
            axons_per_core=2, neurons_per_core=2,
            leak=0, weight_type=float,
        )

        core = chip.cores[0]
        assert core.neurons[0].bias == pytest.approx(0.5)
        assert core.neurons[1].bias == pytest.approx(-0.3)

    def test_no_always_on_connection_in_hw_bias_mode(self):
        """No axon source should be always-on when hardware_bias is used."""
        hc = HardCore(axons_per_core=3, neurons_per_core=2, has_bias_capability=True)
        hc.core_matrix = np.zeros((3, 2))
        hc.hardware_bias = np.array([0.1, 0.2])
        hc.axon_sources = [
            SpikeSource(-2, 0, is_input=True, is_off=False),
            SpikeSource(-2, 1, is_input=True, is_off=False),
            SpikeSource(-1, 0, is_input=False, is_off=True),
        ]
        hc.threshold = 1.0
        hc.latency = 0

        mapping = HardCoreMapping(chip_cores=[])
        mapping.cores = [hc]
        mapping.output_sources = np.array([
            SpikeSource(0, 0, is_input=False, is_off=False),
            SpikeSource(0, 1, is_input=False, is_off=False),
        ])

        chip = hard_cores_to_chip(
            input_size=2, hardcore_mapping=mapping,
            axons_per_core=3, neurons_per_core=2,
            leak=0, weight_type=float,
        )

        conn = chip.connections[0]
        for src in conn.axon_sources:
            assert not src.is_always_on_, "hardware_bias mode must not have always-on axons"

    def test_weight_matrix_uses_full_axons(self):
        """In hardware_bias mode, all core_matrix rows are weights (no bias row peeled)."""
        in_ax = 3
        hw_bias = np.array([1.0, 2.0])
        weight_mat = np.array([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ], dtype=np.float64)

        hc = HardCore(axons_per_core=3, neurons_per_core=2, has_bias_capability=True)
        hc.core_matrix = weight_mat
        hc.hardware_bias = hw_bias
        hc.axon_sources = [
            SpikeSource(-2, i, is_input=True, is_off=False) for i in range(in_ax)
        ]
        hc.threshold = 1.0
        hc.latency = 0

        mapping = HardCoreMapping(chip_cores=[])
        mapping.cores = [hc]
        mapping.output_sources = np.array([
            SpikeSource(0, 0, is_input=False, is_off=False),
            SpikeSource(0, 1, is_input=False, is_off=False),
        ])

        chip = hard_cores_to_chip(
            input_size=3, hardcore_mapping=mapping,
            axons_per_core=3, neurons_per_core=2,
            leak=0, weight_type=float,
        )

        core = chip.cores[0]
        # Neuron 0 weights should be column 0 of weight_mat.T = row 0 of weight_mat.T
        # weight_tensor = weight_mat.transpose() → (2, 3)
        # neuron 0 weights: [0.1, 0.3, 0.5]
        assert core.neurons[0].weights[:3] == pytest.approx([0.1, 0.3, 0.5])
        assert core.neurons[1].weights[:3] == pytest.approx([0.2, 0.4, 0.6])


# ---------------------------------------------------------------------------
# Full chain: IRMapping → NeuralCore → SoftCore → HardCore → ChipModel
# ---------------------------------------------------------------------------

class TestFullHardwareBiasChain:
    """Validate bias values are preserved through the entire chain."""

    def test_bias_preserved_end_to_end(self):
        in_features, out_features = 4, 3
        biases = np.array([0.5, -0.3, 0.7])

        # Step 1: NeuralCore with hardware_bias
        nc = _make_neural_core_with_hw_bias(0, in_features, out_features, seed=42)
        nc.hardware_bias = biases.copy()

        # Step 2: NeuralCore → SoftCore
        sc = neural_core_to_soft_core(nc)
        np.testing.assert_allclose(sc.hardware_bias, biases, atol=1e-6)
        assert sc.core_matrix.shape == (in_features, out_features)

        # Step 3: SoftCore → HardCore
        hc = HardCore(axons_per_core=8, neurons_per_core=8, has_bias_capability=True)
        hc.add_softcore(sc)
        np.testing.assert_allclose(hc.hardware_bias[:out_features], biases, atol=1e-6)

        # Step 4: HardCore → ChipModel
        mapping = HardCoreMapping(chip_cores=[])
        mapping.cores = [hc]
        hc.threshold = 1.0
        hc.latency = 0
        while len(hc.axon_sources) < 8:
            hc.axon_sources.append(SpikeSource(-1, 0, is_input=False, is_off=True))
        mapping.output_sources = np.array([
            SpikeSource(0, i, is_input=False, is_off=False) for i in range(out_features)
        ])

        chip = hard_cores_to_chip(
            input_size=in_features, hardcore_mapping=mapping,
            axons_per_core=8, neurons_per_core=8,
            leak=0, weight_type=float,
        )

        for i in range(out_features):
            assert chip.cores[0].neurons[i].bias == pytest.approx(
                biases[i], abs=1e-6
            ), f"Neuron {i} bias mismatch"

    def test_no_always_on_anywhere_in_chain(self):
        """When hardware_bias is active, no always-on source exists at any stage."""
        nc = _make_neural_core_with_hw_bias(0, in_features=4, out_features=3)

        # NeuralCore level
        for src in nc.input_sources.flatten():
            assert not src.is_always_on()

        # SoftCore level
        sc = neural_core_to_soft_core(nc)
        for src in sc.axon_sources:
            assert not getattr(src, "is_always_on_", False)

        # HardCore level
        hc = HardCore(axons_per_core=8, neurons_per_core=8, has_bias_capability=True)
        hc.add_softcore(sc)
        for src in hc.axon_sources:
            assert not getattr(src, "is_always_on_", False)

    def test_ir_mapping_to_chip_end_to_end(self):
        """Full chain through IRMapping.map_fc → NeuralCore → SoftCore → HardCore → ChipModel."""
        in_features, out_features = 4, 3
        biases = torch.randn(out_features)

        mapper = IRMapping(hardware_bias=True)
        input_sources = np.array([_inp(i) for i in range(in_features)])
        output_shape = np.array([out_features])
        mapper.map_fc(
            input_tensor_sources=input_sources,
            output_shape=output_shape,
            fc_weights=torch.randn(out_features, in_features),
            fc_biases=biases,
            name="test",
        )
        nc = mapper.nodes[-1]
        np.testing.assert_allclose(nc.hardware_bias, biases.numpy(), atol=1e-6)

        sc = neural_core_to_soft_core(nc)
        np.testing.assert_allclose(sc.hardware_bias, biases.numpy(), atol=1e-6)

        hc = HardCore(axons_per_core=8, neurons_per_core=8, has_bias_capability=True)
        hc.add_softcore(sc)
        hc.threshold = 1.0
        hc.latency = 0
        while len(hc.axon_sources) < 8:
            hc.axon_sources.append(SpikeSource(-1, 0, is_input=False, is_off=True))

        mapping = HardCoreMapping(chip_cores=[])
        mapping.cores = [hc]
        mapping.output_sources = np.array([
            SpikeSource(0, i, is_input=False, is_off=False) for i in range(out_features)
        ])

        chip = hard_cores_to_chip(
            input_size=in_features, hardcore_mapping=mapping,
            axons_per_core=8, neurons_per_core=8,
            leak=0, weight_type=float,
        )

        for i in range(out_features):
            assert chip.cores[0].neurons[i].bias == pytest.approx(
                biases[i].item(), abs=1e-6
            )
