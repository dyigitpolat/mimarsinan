"""Error-contract tests: mapping/verification/pruning failures must fail loud (or degrade explicitly)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore
from mimarsinan.mapping.layout.layout_types import (
    LayoutHardCoreType,
    LayoutPackingResult,
    LayoutSoftCoreSpec,
)


def _sources(specs):
    return np.array([IRSource(node_id=n, index=i) for n, i in specs], dtype=object)


def _perceptron_with_1d_masks(in_features: int, out_features: int):
    layer = SimpleNamespace(
        prune_row_mask=torch.zeros(out_features, dtype=torch.bool),
        prune_col_mask=torch.zeros(in_features, dtype=torch.bool),
    )
    return SimpleNamespace(layer=layer, is_encoding_layer=False)


class TestPruningMaskErrorsPropagate:
    def test_get_perceptrons_failure_propagates(self):
        from mimarsinan.mapping.pruning.ir_pruning_masks import (
            get_initial_pruning_masks_from_model,
        )

        def _boom():
            raise RuntimeError("get_perceptrons broke")

        model = SimpleNamespace(get_perceptrons=_boom)
        graph = IRGraph(nodes=[], output_sources=np.array([], dtype=object))
        with pytest.raises(RuntimeError, match="get_perceptrons broke"):
            get_initial_pruning_masks_from_model(model, graph)

    def test_core_matrix_resolution_failure_propagates(self):
        from mimarsinan.mapping.pruning.ir_pruning_masks import (
            get_initial_pruning_masks_from_model,
        )

        core = NeuralCore(
            id=0, name="c0",
            input_sources=_sources([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=None, weight_bank_id=99,
            perceptron_index=0,
        )
        graph = IRGraph(nodes=[core], output_sources=_sources([(0, 0), (0, 1)]))
        model = SimpleNamespace(
            get_perceptrons=lambda: [_perceptron_with_1d_masks(2, 2)]
        )
        with pytest.raises(KeyError):
            get_initial_pruning_masks_from_model(model, graph)

    def test_force_dead_nodes_core_matrix_failure_propagates(self):
        from mimarsinan.mapping.pruning.ir_pruning_helpers import (
            _force_dead_nodes_fully_pruned,
        )
        from mimarsinan.mapping.pruning.graph.pruning_graph_types import (
            GlobalPruningResult,
        )

        core = NeuralCore(
            id=0, name="dead",
            input_sources=_sources([(-2, 0), (-3, 0)]),
            core_matrix=None, weight_bank_id=42,
        )
        graph = IRGraph(nodes=[core], output_sources=np.array([], dtype=object))
        with pytest.raises(KeyError):
            _force_dead_nodes_fully_pruned(graph, [0], GlobalPruningResult())


class TestPackLayoutContract:
    def test_infeasible_packing_is_a_result_not_an_exception(self):
        from mimarsinan.mapping.layout.layout_packer import pack_layout

        result = pack_layout(
            softcores=[LayoutSoftCoreSpec(input_count=100, output_count=100)],
            core_types=[LayoutHardCoreType(max_axons=4, max_neurons=4, count=1)],
            allow_neuron_splitting=False,
            allow_coalescing=False,
        )
        assert result.feasible is False
        assert result.error

    def test_non_packing_errors_propagate(self, monkeypatch):
        import mimarsinan.mapping.layout.layout_packer as lp

        def _bug(**kwargs):
            raise ValueError("placement invariant broken")

        monkeypatch.setattr(lp, "run_placement", _bug)
        with pytest.raises(ValueError, match="placement invariant broken"):
            lp.pack_layout(
                softcores=[LayoutSoftCoreSpec(input_count=2, output_count=2)],
                core_types=[LayoutHardCoreType(max_axons=4, max_neurons=4, count=1)],
                allow_neuron_splitting=False,
                allow_coalescing=False,
            )


class TestScheduleErrorsPropagate:
    def test_split_softcores_by_capacity_propagates_pack_bugs(self, monkeypatch):
        import mimarsinan.mapping.layout.layout_packer as lp
        from mimarsinan.mapping.support.schedule.schedule_split import (
            split_softcores_by_capacity,
        )

        def _bug(**kwargs):
            raise ValueError("pack bug")

        monkeypatch.setattr(lp, "pack_layout", _bug)
        scs = [
            LayoutSoftCoreSpec(input_count=2, output_count=2, latency_tag=0),
            LayoutSoftCoreSpec(input_count=2, output_count=2, latency_tag=1),
        ]
        with pytest.raises(ValueError, match="pack bug"):
            split_softcores_by_capacity(
                scs, [LayoutHardCoreType(max_axons=4, max_neurons=4, count=2)]
            )

    def test_estimate_passes_validated_propagates_pack_bugs(self, monkeypatch):
        import mimarsinan.mapping.layout.layout_packer as lp
        from mimarsinan.mapping.support.schedule.schedule_partitioner import (
            estimate_passes_for_layout_validated,
        )

        def _bug(**kwargs):
            raise ValueError("pack bug")

        monkeypatch.setattr(lp, "pack_layout", _bug)
        with pytest.raises(ValueError, match="pack bug"):
            estimate_passes_for_layout_validated(
                [LayoutSoftCoreSpec(input_count=2, output_count=2)],
                1,
                core_types=[LayoutHardCoreType(max_axons=4, max_neurons=4, count=1)],
            )

    def test_estimate_passes_validated_reports_genuine_infeasibility(self):
        from mimarsinan.mapping.support.schedule.schedule_partitioner import (
            estimate_passes_for_layout_validated,
        )

        n_passes, pass_lists, all_ok = estimate_passes_for_layout_validated(
            [LayoutSoftCoreSpec(input_count=100, output_count=100)],
            1,
            core_types=[LayoutHardCoreType(max_axons=4, max_neurons=4, count=1)],
        )
        assert all_ok is False

    def test_compute_mapping_stats_propagates_pack_bugs(self, monkeypatch):
        import mimarsinan.mapping.verification.layout_verification_scheduling as lvs

        sc = LayoutSoftCoreSpec(input_count=2, output_count=2)
        calls = {"n": 0}

        def _pack_stub(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return LayoutPackingResult(
                    feasible=False, cores_used=0, total_capacity=0,
                    used_area=0, unused_area_total=0,
                    avg_unused_area_per_core=0.0, error="infeasible",
                )
            raise ValueError("pack bug")

        monkeypatch.setattr(lvs, "pack_layout", _pack_stub)
        monkeypatch.setattr(
            lvs, "estimate_passes_for_layout_validated",
            lambda scs, budget, **kw: (1, [list(scs)], True),
        )
        with pytest.raises(ValueError, match="pack bug"):
            lvs.compute_mapping_stats(
                [sc],
                [LayoutHardCoreType(max_axons=4, max_neurons=4, count=1)],
                allow_scheduling=True,
            )

    def test_compute_mapping_stats_scheduled_feasibility_is_not_silently_lost(self):
        """Regression: dict core_types leaked into the packer and the resulting
        AttributeError was swallowed, reporting every scheduled layout infeasible."""
        import mimarsinan.mapping.verification.layout_verification_scheduling as lvs

        scs = [
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=0),
            LayoutSoftCoreSpec(input_count=4, output_count=4, latency_tag=1),
        ]
        stats, error = lvs.compute_mapping_stats(
            scs,
            [LayoutHardCoreType(max_axons=4, max_neurons=4, count=1)],
            allow_scheduling=True,
        )
        assert error is None
        assert stats.feasible is True
        assert stats.schedule_pass_count == 2

    def test_verify_hardware_config_propagates_pack_bugs(self, monkeypatch):
        import mimarsinan.mapping.support.schedule.schedule_partitioner as sp
        import mimarsinan.mapping.verification.verifier.mapping_verifier_hw as mvh

        sc = LayoutSoftCoreSpec(input_count=2, output_count=2)
        calls = {"n": 0}

        def _pack_stub(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return LayoutPackingResult(
                    feasible=False, cores_used=0, total_capacity=0,
                    used_area=0, unused_area_total=0,
                    avg_unused_area_per_core=0.0, error="infeasible",
                )
            raise ValueError("pack bug")

        monkeypatch.setattr(mvh, "pack_layout", _pack_stub)
        monkeypatch.setattr(
            sp, "estimate_passes_for_layout_validated",
            lambda scs, budget, **kw: (1, [list(scs)], True),
        )
        with pytest.raises(ValueError, match="pack bug"):
            mvh.verify_hardware_config(
                [sc],
                [{"max_axons": 4, "max_neurons": 4, "count": 1}],
                allow_scheduling=True,
            )


class TestVerifierExplicitFallbacks:
    def test_soft_verifier_reports_error_result_and_warns(self, caplog):
        from mimarsinan.mapping.verification.verifier.mapping_verifier_soft import (
            verify_soft_core_mapping,
        )

        class NotAModelRepr:
            pass

        with caplog.at_level(logging.WARNING, logger="mimarsinan.mapping"):
            result = verify_soft_core_mapping(
                NotAModelRepr(), max_axons=64, max_neurons=64
            )
        assert result.feasible is False
        assert result.error
        assert any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), "soft-mapping verification failure must be logged at warning level"


class TestSuggesterExplicitFallback:
    def test_failed_candidate_is_skipped_with_warning(self, monkeypatch, caplog):
        import mimarsinan.mapping.verification.suggester.hw_config_suggester_scheduled as mod
        from mimarsinan.mapping.verification.suggester.hw_suggestion_types import (
            HardwareSuggestion,
        )

        single = HardwareSuggestion(
            core_types=[{"max_axons": 8, "max_neurons": 8, "count": 4}],
            total_cores=4,
            rationale="stub single-pass",
        )
        calls = {"n": 0}

        def _suggest_stub(scs, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return single
            raise ValueError("candidate suggestion bug")

        monkeypatch.setattr(mod, "suggest_hardware_config", _suggest_stub)
        monkeypatch.setattr(
            mod, "estimate_passes_for_layout",
            lambda scs, budget, **kw: (2, [list(scs), list(scs)]),
        )

        scs = [LayoutSoftCoreSpec(input_count=2, output_count=2)]
        with caplog.at_level(logging.WARNING, logger="mimarsinan.mapping"):
            result = mod.suggest_hardware_config_scheduled(scs)
        assert result.core_types == single.core_types
        assert any(r.levelno >= logging.WARNING for r in caplog.records), (
            "skipped scheduling candidate must be logged at warning level"
        )


class TestWizardSnapshotDegradation:
    def test_model_repr_from_model_returns_none_on_failure(self):
        from mimarsinan.mapping.verification.wizard_layout_verify import (
            model_repr_from_model,
        )

        def _boom():
            raise ValueError("repr extraction broke")

        model = SimpleNamespace(get_mapper_repr=_boom)
        assert model_repr_from_model(model) is None

    def test_model_repr_from_model_reraises_keyboard_interrupt(self):
        from mimarsinan.mapping.verification.wizard_layout_verify import (
            model_repr_from_model,
        )

        def _interrupt():
            raise KeyboardInterrupt()

        model = SimpleNamespace(get_mapper_repr=_interrupt)
        with pytest.raises(KeyboardInterrupt):
            model_repr_from_model(model)


class TestChipQuantizeScalarScale:
    def _graph(self, parameter_scale):
        core = NeuralCore(
            id=0, name="c0",
            input_sources=_sources([(-2, 0), (-2, 1), (-3, 0)]),
            core_matrix=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            parameter_scale=parameter_scale,
        )
        return IRGraph(nodes=[core], output_sources=_sources([(0, 0), (0, 1)]))

    def test_verify_accepts_tensor_and_float_scales(self):
        from mimarsinan.mapping.export.chip_quantize import verify_ir_graph_quantized

        verify_ir_graph_quantized(self._graph(torch.tensor(1.0)), bits=8)
        verify_ir_graph_quantized(self._graph(1.0), bits=8)
