"""W3 threading: `elimination_propagation` flows config -> prune_ir_graph.

The key is read once by ``apply_ir_pruning_if_enabled`` and passed verbatim
into both the ledger and ``prune_ir_graph``; an absent key resolves to the
byte-identical default arm ("cascade"); an invalid value fails loud before
any pruning runs.
"""

from __future__ import annotations

import pytest

from conftest import make_tiny_ir_graph
from mimarsinan.pipelining.pipeline_steps.mapping import (
    soft_core_mapping_ir_pruning as mod,
)


class _StepStub:
    def __init__(self, pipeline):
        self.pipeline = pipeline


class _ModelStub:
    def get_perceptrons(self):
        return []


class _LedgerStub:
    def summary(self):
        return "[Ledger] stub"


def _run_pruning(mock_pipeline, monkeypatch, extra_config):
    mock_pipeline.config["pruning"] = True
    mock_pipeline.config.update(extra_config)
    captured = {"prune": None, "ledger": None}

    def fake_prune(ir_graph, **kwargs):
        captured["prune"] = kwargs
        return ir_graph

    def fake_ledger(ir_graph, **kwargs):
        captured["ledger"] = kwargs
        return _LedgerStub()

    monkeypatch.setattr(mod, "prune_ir_graph", fake_prune)
    monkeypatch.setattr(mod, "compute_elimination_ledger", fake_ledger)
    monkeypatch.setattr(
        mod, "get_initial_pruning_masks_from_model", lambda m, g: ({}, {})
    )
    step = _StepStub(mock_pipeline)
    mod.apply_ir_pruning_if_enabled(
        step, _ModelStub(), make_tiny_ir_graph(), "test_phase"
    )
    return captured


class TestEliminationPropagationThreading:
    def test_absent_key_defaults_to_cascade(self, mock_pipeline, monkeypatch):
        captured = _run_pruning(mock_pipeline, monkeypatch, {})
        assert captured["prune"]["elimination_propagation"] == "cascade"
        assert captured["ledger"]["elimination_propagation"] == "cascade"

    def test_configured_mode_is_threaded_through(self, mock_pipeline, monkeypatch):
        captured = _run_pruning(
            mock_pipeline, monkeypatch, {"elimination_propagation": "closure"}
        )
        assert captured["prune"]["elimination_propagation"] == "closure"
        assert captured["ledger"]["elimination_propagation"] == "closure"

    def test_invalid_mode_fails_loud_before_pruning(
        self, mock_pipeline, monkeypatch
    ):
        with pytest.raises(ValueError, match="elimination_propagation"):
            _run_pruning(
                mock_pipeline, monkeypatch, {"elimination_propagation": "bogus"}
            )

    def test_pruning_disabled_skips_everything(self, mock_pipeline, monkeypatch):
        mock_pipeline.config["pruning"] = False
        called = []
        monkeypatch.setattr(
            mod, "prune_ir_graph",
            lambda *a, **k: called.append("prune"),
        )
        graph = make_tiny_ir_graph()
        out = mod.apply_ir_pruning_if_enabled(
            _StepStub(mock_pipeline), _ModelStub(), graph, "test_phase"
        )
        assert out is graph
        assert called == []


class TestConfigSchemaRegistration:
    def test_key_registered_with_mode_options(self):
        from mimarsinan.config_schema.registry import REGISTRY
        from mimarsinan.mapping.pruning.graph.propagation_mode import (
            ELIMINATION_PROPAGATION_MODES,
        )

        entry = REGISTRY["elimination_propagation"]
        assert tuple(entry.options) == ELIMINATION_PROPAGATION_MODES
