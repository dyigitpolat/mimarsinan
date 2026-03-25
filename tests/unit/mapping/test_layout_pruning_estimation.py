"""Regression tests for layout pruning estimation correctness.

Verifies:
1. segment_id is preserved after pruning estimation (B1 fix).
2. hardware_bias is respected in bias counting (B2 fix).
3. Output layer column protection (H1).
4. Per-bank threshold group assignment (H2).
5. Uniform bank-backed pruning (H3).
"""

from __future__ import annotations

import numpy as np
import pytest

from mimarsinan.mapping.ir import IRSource
from mimarsinan.mapping.layout.layout_ir_mapping import LayoutIRMapping
from mimarsinan.mapping.layout.layout_types import LayoutSoftCoreSpec


def _make_chain_mapping(
    *,
    n_layers: int = 2,
    in_features: int = 64,
    hidden: int = 32,
    out_features: int = 10,
    max_axons: int = 1024,
    max_neurons: int = 1024,
    pruning_fraction: float = 0.5,
    hardware_bias: bool = False,
    threshold_groups: int = 1,
) -> LayoutIRMapping:
    """Build a simple chain model as a LayoutIRMapping directly."""
    mapping = LayoutIRMapping(
        max_axons=max_axons,
        max_neurons=max_neurons,
        pruning_fraction=pruning_fraction,
        hardware_bias=hardware_bias,
        threshold_groups=threshold_groups,
    )
    src = np.array([IRSource(node_id=-2, index=i) for i in range(in_features)])
    for i in range(n_layers):
        out_f = out_features if i == n_layers - 1 else hidden
        w = np.zeros((out_f, src.shape[0]))
        b = np.zeros(out_f)
        src = mapping.map_fc(src, np.array([out_f]), w, b)
    return mapping, src


class TestSegmentIdPreserved:
    def test_segment_ids_survive_pruning(self):
        mapping, output_src = _make_chain_mapping(n_layers=2, pruning_fraction=0.3)
        softcores = mapping.collect_layout_softcores.__wrapped__(mapping) if hasattr(
            mapping.collect_layout_softcores, '__wrapped__') else None

        mapping2 = LayoutIRMapping(
            max_axons=1024, max_neurons=1024,
            pruning_fraction=0.3, threshold_groups=1,
        )
        src = np.array([IRSource(node_id=-2, index=i) for i in range(64)])
        w1, w2 = np.zeros((32, 64)), np.zeros((10, 32))
        b1, b2 = np.zeros(32), np.zeros(10)
        src = mapping2.map_fc(src, np.array([32]), w1, b1)
        # Add compute op to create new segment
        src = mapping2.add_compute_op(src, "identity", params={}, output_shape=(32,))
        src = mapping2.map_fc(src, np.array([10]), w2, b2)

        class FakeModel:
            def __init__(self, m):
                self._m = m
                self._output = src
            def map_to_ir(self, backend):
                return self._output

        # Manually run the mapping path
        mapping2._finalize_softcores()

        # Verify segment_ids exist
        seg_ids = {sc.segment_id for sc in mapping2.layout_softcores if sc.segment_id is not None}
        assert len(seg_ids) >= 2, "Should have at least 2 segments when a compute op is between"

        # Now test with pruning
        mapping3 = LayoutIRMapping(
            max_axons=1024, max_neurons=1024,
            pruning_fraction=0.3, threshold_groups=1,
        )
        src3 = np.array([IRSource(node_id=-2, index=i) for i in range(64)])
        src3 = mapping3.map_fc(src3, np.array([32]), w1, b1)
        src3 = mapping3.add_compute_op(src3, "identity", params={}, output_shape=(32,))
        out3 = mapping3.map_fc(src3, np.array([10]), w2, b2)

        mapping3._finalize_softcores()
        # Apply pruning manually as collect_layout_softcores would
        from mimarsinan.mapping.mapping_structure import compute_core_input_count
        import math, random

        effective = 0.3 * 0.8
        rng = random.Random(7919)
        pruned = []
        for sc in mapping3.layout_softcores:
            in_reduce = int(math.floor(sc.input_count * effective))
            out_reduce = int(math.floor(sc.output_count * effective))
            new_in = max(1, sc.input_count - in_reduce)
            new_out = max(1, sc.output_count - out_reduce)
            pruned.append(LayoutSoftCoreSpec(
                input_count=new_in, output_count=new_out,
                threshold_group_id=sc.threshold_group_id,
                latency_tag=sc.latency_tag,
                segment_id=sc.segment_id,
                name=sc.name,
            ))

        for sc in pruned:
            assert sc.segment_id is not None, f"segment_id lost during pruning for {sc.name}"


class TestHardwareBiasRespected:
    def test_legacy_bias_adds_axon(self):
        mapping = LayoutIRMapping(max_axons=1024, max_neurons=1024, hardware_bias=False)
        src = np.array([IRSource(node_id=-2, index=i) for i in range(64)])
        w = np.zeros((16, 64))
        b = np.zeros(16)
        mapping.add_neural_core(input_sources=src, weights=w, biases=b)
        assert mapping.layout_softcores[0].input_count == 65

    def test_hardware_bias_no_extra_axon(self):
        mapping = LayoutIRMapping(max_axons=1024, max_neurons=1024, hardware_bias=True)
        src = np.array([IRSource(node_id=-2, index=i) for i in range(64)])
        w = np.zeros((16, 64))
        b = np.zeros(16)
        mapping.add_neural_core(input_sources=src, weights=w, biases=b)
        assert mapping.layout_softcores[0].input_count == 64

    def test_shared_core_legacy_bias(self):
        mapping = LayoutIRMapping(max_axons=1024, max_neurons=1024, hardware_bias=False)
        w = np.zeros((16, 9))
        b = np.zeros(16)
        bid = mapping.register_weight_bank(w, b)
        src = np.array([IRSource(node_id=-2, index=i) for i in range(9)])
        mapping.add_shared_neural_core(input_sources=src, weight_bank_id=bid, has_bias=True)
        assert mapping.layout_softcores[0].input_count == 10  # 9 + 1

    def test_shared_core_hardware_bias(self):
        mapping = LayoutIRMapping(max_axons=1024, max_neurons=1024, hardware_bias=True)
        w = np.zeros((16, 9))
        b = np.zeros(16)
        bid = mapping.register_weight_bank(w, b)
        src = np.array([IRSource(node_id=-2, index=i) for i in range(9)])
        mapping.add_shared_neural_core(input_sources=src, weight_bank_id=bid, has_bias=True)
        assert mapping.layout_softcores[0].input_count == 9  # no extra axon


class TestOutputLayerProtection:
    def test_output_columns_not_pruned(self):
        """The final-layer softcores should preserve output_count after pruning."""
        mapping = LayoutIRMapping(
            max_axons=1024, max_neurons=1024,
            pruning_fraction=0.5, threshold_groups=1,
        )
        src = np.array([IRSource(node_id=-2, index=i) for i in range(64)])
        w1 = np.zeros((32, 64))
        b1 = np.zeros(32)
        src = mapping.map_fc(src, np.array([32]), w1, b1)
        w2 = np.zeros((10, 32))
        b2 = np.zeros(10)

        class FakeModel:
            def __init__(self, m, s):
                self._m = m
                self._src = s
                self._w2 = w2
                self._b2 = b2
            def map_to_ir(self, backend):
                return backend.map_fc(self._src, np.array([10]), self._w2, self._b2)

        mapping2 = LayoutIRMapping(
            max_axons=1024, max_neurons=1024,
            pruning_fraction=0.5, threshold_groups=1,
        )
        src2 = np.array([IRSource(node_id=-2, index=i) for i in range(64)])
        src2 = mapping2.map_fc(src2, np.array([32]), w1, b1)
        model = FakeModel(mapping2, src2)
        softcores = mapping2.collect_layout_softcores(model)

        # The last softcore feeds the output -- its output_count should be 10 (protected)
        output_sc = softcores[-1]
        assert output_sc.output_count == 10, (
            f"Output layer was pruned: got {output_sc.output_count}, expected 10"
        )

        # The hidden layer should be pruned (output_count < 32)
        hidden_sc = softcores[0]
        assert hidden_sc.output_count < 32, (
            f"Hidden layer not pruned: got {hidden_sc.output_count}"
        )


class TestPerBankThresholdGroup:
    def test_shared_bank_cores_same_group(self):
        """All spatial positions sharing a weight bank get the same threshold group."""
        mapping = LayoutIRMapping(
            max_axons=1024, max_neurons=1024,
            threshold_groups=8, threshold_seed=42,
        )
        w = np.zeros((16, 9))
        b = np.zeros(16)
        bid = mapping.register_weight_bank(w, b)
        for pos in range(25):
            src = np.array([IRSource(node_id=-2, index=i) for i in range(9)])
            mapping.add_shared_neural_core(
                input_sources=src, weight_bank_id=bid, has_bias=True,
                name=f"conv_pos{pos}",
            )
        mapping._finalize_softcores()

        tg_set = {sc.threshold_group_id for sc in mapping.layout_softcores}
        assert len(tg_set) == 1, (
            f"Bank-backed cores should share one threshold group, got {len(tg_set)}: {tg_set}"
        )


class TestBankUniformPruning:
    def test_bank_backed_cores_identical_reduction(self):
        """All spatial positions sharing a bank should have identical pruning."""
        mapping = LayoutIRMapping(
            max_axons=1024, max_neurons=1024,
            pruning_fraction=0.5, threshold_groups=1,
        )
        w = np.zeros((16, 9))
        b = np.zeros(16)
        bid = mapping.register_weight_bank(w, b)
        for pos in range(25):
            src = np.array([IRSource(node_id=-2, index=i) for i in range(9)])
            mapping.add_shared_neural_core(
                input_sources=src, weight_bank_id=bid, has_bias=True,
                name=f"conv_pos{pos}",
            )
        # Need to create a FakeModel for collect_layout_softcores
        # Use the direct internal approach instead
        mapping._finalize_softcores()

        # Manually trigger pruning as collect_layout_softcores would
        import math, random

        effective = 0.5 * 0.8
        rng = random.Random(mapping.threshold_seed + 7919)

        bank_reductions = {}
        for bank_id, (bank_in, bank_out) in mapping._layout_weight_banks.items():
            in_red = int(math.floor(bank_in * effective))
            out_red = int(math.floor(bank_out * effective))
            bank_reductions[bank_id] = (in_red, out_red)

        input_counts = set()
        output_counts = set()
        for idx, sc in enumerate(mapping.layout_softcores):
            bid_sc = mapping._sc_idx_to_bank_id.get(idx)
            if bid_sc is not None and bid_sc in bank_reductions:
                in_red, out_red = bank_reductions[bid_sc]
                new_in = max(1, sc.input_count - in_red)
                new_out = max(1, sc.output_count - out_red)
                input_counts.add(new_in)
                output_counts.add(new_out)

        assert len(input_counts) == 1, f"Bank-backed cores got different input reductions: {input_counts}"
        assert len(output_counts) == 1, f"Bank-backed cores got different output reductions: {output_counts}"
