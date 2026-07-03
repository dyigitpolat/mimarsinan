"""Locks in the best_effort degrade contract for gui/snapshot broad-catch sites.

Every function here must (a) never raise on malformed/broken input, and
(b) fall back to the exact value the pre-migration hand-rolled
``try/except`` produced (None / empty list / skip-the-key / continue-the-loop).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.gui.snapshot.heatmap import (
    LIVENESS_LIVE,
    _detect_neural_core_liveness,
    _make_bias_strip_producer,
    _make_heatmap_producer,
)
from mimarsinan.gui.snapshot.ir_graph.ir_graph_resources import (
    _extract_core_connectivity,
    _make_segment_spans_extractor,
)
from mimarsinan.gui.snapshot.ir_graph.ir_graph_topology import snapshot_ir_graph
from mimarsinan.gui.snapshot.mapping_snapshot import (
    RESOURCE_KIND_HARD_CORE_HEATMAP,
    snapshot_hard_core_mapping,
    snapshot_mapping_performance_planned,
)
from mimarsinan.gui.snapshot.model_snapshot import _get_model_perceptrons, snapshot_model
from mimarsinan.gui.snapshot.rebuild import rebuild_step_snapshot_from_disk
from mimarsinan.gui.snapshot.sanafe_snapshot import _find_ir_graph_promiser
from mimarsinan.gui.snapshot.search_snapshot import snapshot_search_result
from mimarsinan.gui.snapshot.util.helpers import _safe_scalar
from mimarsinan.mapping.ir import IRGraph, IRSource, NeuralCore, WeightBank


def _src(specs):
    return np.array([IRSource(node_id=nid, index=idx) for nid, idx in specs], dtype=object)


class TestSearchSnapshotDictFormDegrade:
    def test_best_degrades_to_none_on_bad_shape(self):
        result = snapshot_search_result({"best": "not-a-dict"})
        assert result["best"] is None

    def test_pareto_front_keeps_partial_results_before_failure(self):
        d = {
            "pareto_front": [
                {"configuration": {"x": 1}, "objectives": {"y": 2}},
                "invalid-entry",
            ]
        }
        result = snapshot_search_result(d)
        assert len(result["pareto_front"]) == 1
        assert result["pareto_front"][0]["config"] == {"x": 1}

    def test_history_degrades_to_empty_list_when_not_iterable(self):
        result = snapshot_search_result({"history": 42})
        assert result["history"] == []

    def test_objectives_degrades_to_empty_list_when_not_iterable(self):
        result = snapshot_search_result({"objectives": 42})
        assert result["objectives"] == []


class TestSearchSnapshotObjFormDegrade:
    def test_degrades_fully_when_attrs_missing(self):
        class Empty:
            pass

        result = snapshot_search_result(Empty())
        assert result["best"] is None
        assert result["pareto_front"] == []
        assert result["history"] == []
        assert result["objectives"] == []
        assert result["num_candidates"] == 0


class TestSnapshotModelDegrade:
    def test_weight_and_bias_degrade_to_none_but_layer_still_recorded(self):
        class BadLayer:
            pass

        p = SimpleNamespace(name="p0", layer=BadLayer())
        model = SimpleNamespace(get_perceptrons=lambda: [p], parameters=lambda: iter(()))
        result = snapshot_model(model)
        assert len(result["layers"]) == 1
        assert result["layers"][0]["weight"] is None
        assert result["layers"][0]["bias"] is None

    def test_total_params_falls_back_to_manual_sum_when_parameters_raises(self):
        p = SimpleNamespace(
            name="p0",
            layer=SimpleNamespace(
                weight=SimpleNamespace(data=torch.ones(2, 3)),
                bias=SimpleNamespace(data=torch.ones(3)),
            ),
        )

        def bad_parameters():
            raise RuntimeError("boom")

        model = SimpleNamespace(get_perceptrons=lambda: [p], parameters=bad_parameters)
        result = snapshot_model(model)
        assert result["total_params"] == 2 * 3 + 3

    def test_get_model_perceptrons_falls_through_to_next_strategy_on_exception(self):
        class BadModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(2, 3)

            def get_perceptrons(self):
                raise RuntimeError("boom")

        model = BadModel()
        perceptrons = _get_model_perceptrons(model)
        assert len(perceptrons) == 1
        assert perceptrons[0].layer is model.fc


class TestMappingSnapshotDegrade:
    def test_planned_mapping_performance_returns_none_when_verify_raises(self, monkeypatch):
        import mimarsinan.mapping.verification.wizard_layout_verify as wlv

        monkeypatch.setattr(wlv, "model_repr_from_model", lambda *a, **k: {"dummy": True})

        def _raise(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(wlv, "verify_planned_mapping_performance", _raise)

        result = snapshot_mapping_performance_planned(
            model=object(), platform_constraints={"cores": [{"max_axons": 4}]},
        )
        assert result is None

    def test_hard_core_heatmap_registration_skipped_on_broken_core_matrix(self):
        from mimarsinan.mapping.ir import ir_graph_to_soft_core_mapping
        from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping

        w1 = np.ones((5, 4), dtype=np.float32) * 0.1
        s1 = _src([(-2, 0), (-2, 1), (-2, 2), (-2, 3), (-3, 0)])
        c1 = NeuralCore(id=0, name="h", input_sources=s1, core_matrix=w1, latency=0)
        out = _src([(0, 0), (0, 1)])
        ir = IRGraph(nodes=[c1], output_sources=out)
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}]
        )
        neural_stage = next(
            s for s in hm.stages if s.kind == "neural" and s.hard_core_mapping is not None
        )
        neural_stage.hard_core_mapping.cores[0].core_matrix = None

        snap, descs = snapshot_hard_core_mapping(hm)
        neural_stages = [s for s in snap["stages"] if s.get("kind") == "neural" and s.get("cores")]
        core0 = neural_stages[0]["cores"][0]
        # The failure happens after `has_heatmap` is set but before the shape
        # fields / descriptor are populated (matches pre-migration partial-mutation
        # behavior: only the fields written *after* the failing line are missing).
        assert "heatmap_axons" not in core0
        assert "heatmap_neurons" not in core0
        assert not any(d.kind == RESOURCE_KIND_HARD_CORE_HEATMAP and d.rid.endswith("core/0") for d in descs)
        # Connectivity/placement bookkeeping (set outside the broken try) is unaffected.
        assert core0["has_connectivity"] is True
        assert "mapped_placements" in core0

    def test_io_map_extraction_skipped_when_input_map_entries_are_malformed(self):
        from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping

        w1 = np.ones((5, 4), dtype=np.float32) * 0.1
        s1 = _src([(-2, 0), (-2, 1), (-2, 2), (-2, 3), (-3, 0)])
        c1 = NeuralCore(id=0, name="h", input_sources=s1, core_matrix=w1, latency=0)
        out = _src([(0, 0), (0, 1)])
        ir = IRGraph(nodes=[c1], output_sources=out)
        hm = build_hybrid_hard_core_mapping(
            ir_graph=ir, cores_config=[{"max_axons": 32, "max_neurons": 32, "count": 2}]
        )
        neural_stage = next(
            s for s in hm.stages if s.kind == "neural" and s.hard_core_mapping is not None
        )
        neural_stage.input_map = [object()]  # lacks .node_id/.offset/.size

        snap, _descs = snapshot_hard_core_mapping(hm)
        neural_stages = [s for s in snap["stages"] if s.get("kind") == "neural" and s.get("cores")]
        assert "input_map" not in neural_stages[0]
        assert neural_stages[0]["has_connectivity"] is True


class TestHeatmapDegrade:
    def test_detect_neural_core_liveness_falls_back_to_live_when_mat_shape_raises(self):
        class BadMat:
            @property
            def shape(self):
                raise RuntimeError("boom")

        node = SimpleNamespace(
            pre_pruning_row_mask=None,
            pre_pruning_col_mask=None,
            input_sources=SimpleNamespace(flatten=lambda: []),
            pruned_row_mask=None,
            pruned_col_mask=None,
        )
        assert _detect_neural_core_liveness(node, BadMat()) == LIVENESS_LIVE

    def test_make_heatmap_producer_does_not_raise_when_conversion_fails(self):
        class Unconvertible:
            def __array__(self, *a, **k):
                raise RuntimeError("boom")

        producer = _make_heatmap_producer(Unconvertible(), copy=True)
        assert callable(producer)

    def test_make_bias_strip_producer_does_not_raise_when_conversion_fails(self):
        class Unconvertible:
            def __array__(self, *a, **k):
                raise RuntimeError("boom")

        producer = _make_bias_strip_producer(Unconvertible())
        assert callable(producer)


class TestSanafeSnapshotFindPromiserDegrade:
    def test_returns_none_when_steps_property_raises(self):
        class BadPipeline:
            @property
            def steps(self):
                raise RuntimeError("boom")

        assert _find_ir_graph_promiser(BadPipeline()) is None

    def test_skips_step_whose_promises_raises_but_finds_next(self):
        class BadStep:
            @property
            def promises(self):
                raise RuntimeError("boom")

        class GoodStep:
            promises = ("ir_graph",)

        pipeline = SimpleNamespace(
            steps=(("Bad Step", BadStep()), ("Good Step", GoodStep()))
        )
        assert _find_ir_graph_promiser(pipeline) == "Good Step"


class TestRebuildDegrade:
    def test_returns_none_on_corrupted_pickle(self, tmp_path):
        pickle_path = tmp_path / "SANA-FE Simulation.sanafe_simulation_results.pickle"
        pickle_path.write_bytes(b"not a valid pickle stream")
        assert rebuild_step_snapshot_from_disk(str(tmp_path), "SANA-FE Simulation") is None


class TestHelpersSafeScalarDegrade:
    def test_returns_none_when_attribute_access_raises_non_attribute_error(self):
        class Explodes:
            @property
            def bad(self):
                raise RuntimeError("boom")

        assert _safe_scalar(Explodes(), "bad") is None


class TestIRGraphNodesDegrade:
    def test_hardware_bias_stats_skipped_when_conversion_fails(self):
        core = NeuralCore(
            id=1, name="c",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            hardware_bias=["not", "numeric"],
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[core], output_sources=_src([(1, 0)]))
        snap, _descs = snapshot_ir_graph(graph)
        info = snap["nodes"][0]
        assert "hardware_bias_stats" not in info


class TestIRGraphResourcesDegrade:
    def test_extract_core_connectivity_skips_core_whose_span_extraction_raises(self):
        class BadCore:
            def get_axon_source_spans(self):
                raise RuntimeError("boom")

        class GoodCore:
            def get_axon_source_spans(self):
                return [
                    SimpleNamespace(
                        kind="core", src_core=0, src_start=0, src_end=1,
                        dst_start=0, dst_end=1, length=1,
                    )
                ]

        hcm = SimpleNamespace(cores=[BadCore(), GoodCore()], output_sources=None)
        spans = _extract_core_connectivity(hcm, segment_index=0)
        assert len(spans) == 1
        assert spans[0]["dst_core"] == 1

    def test_segment_spans_extractor_falls_back_to_empty_list_on_failure(self):
        class BrokenHCM:
            @property
            def cores(self):
                raise RuntimeError("boom")

        get_all = _make_segment_spans_extractor(BrokenHCM(), segment_index=0)
        assert get_all() == []


class TestIRGraphTopologyWeightBanksDegrade:
    def test_bad_bank_skipped_but_good_bank_still_registered(self):
        good_bank = WeightBank(id=0, core_matrix=np.ones((4, 4), dtype=np.float64))
        bad_bank = WeightBank(id=1, core_matrix=np.ones((4, 4), dtype=np.float64))
        graph = IRGraph(
            nodes=[],
            output_sources=np.array([], dtype=object),
            weight_banks={0: good_bank, "not-an-int": bad_bank},
        )
        snap, _descs = snapshot_ir_graph(graph)
        assert 0 in snap["weight_banks"]
        assert len(snap["weight_banks"]) == 1

    def test_output_sources_extraction_degrades_without_raising(self):
        core = NeuralCore(
            id=0, name="c",
            input_sources=_src([(-2, 0)]),
            core_matrix=np.array([[1.0]], dtype=np.float64),
            threshold=1.0, latency=0,
        )
        graph = IRGraph(nodes=[core], output_sources=object())
        snap, _descs = snapshot_ir_graph(graph)
        # ir_graph.output_sources.flatten() raises (object() has no .flatten());
        # no "-> output" edge is produced, but per-node edges (from input_sources)
        # are unaffected since they are extracted in a separate, earlier step.
        assert all(e["to"] != "output" for e in snap["edges"])
