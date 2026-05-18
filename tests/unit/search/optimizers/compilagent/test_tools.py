"""Tests for the four ``ToolDecl``s exposed by ``MimarsinanLayoutBackend``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compilagent import Plan, ToleranceConfig, WorkloadKind, WorkloadSpec

from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.tools import build_introspection_tools
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)

# Re-use the fakes from the backend tests
from .test_backend import _make_problem, _make_workload  # noqa: E402


@pytest.fixture
def compiled_backend(tmp_path: Path):
    workload_id = "fake_layout_tools_test"
    problem = _make_problem()
    register_problem(workload_id, problem)
    backend = MimarsinanLayoutBackend()
    workload = _make_workload(workload_id)
    cdir = tmp_path / "cand-abc"
    cdir.mkdir()
    result = backend.compile(workload, Plan(), artifact_dir=cdir)
    assert result.ok, result.diagnostics
    candidate_id = cdir.name
    try:
        yield backend, candidate_id
    finally:
        unregister_problem(workload_id)


def _by_name(decls, name):
    return next(d for d in decls if d.name == name)


class TestSurfaceShape:
    def test_four_tools_returned(self, compiled_backend):
        backend, _ = compiled_backend
        decls = build_introspection_tools(backend)
        names = sorted(d.name for d in decls)
        assert names == [
            "inspect_layer_breakdown",
            "inspect_layout_stats",
            "inspect_softcores",
            "list_objectives",
        ]

    def test_all_tools_are_read_only(self, compiled_backend):
        backend, _ = compiled_backend
        decls = build_introspection_tools(backend)
        assert all(d.read_only for d in decls)


class TestInspectSoftcores:
    def test_returns_known_count(self, compiled_backend):
        backend, candidate_id = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        result = json.loads(decl.invoke({"candidate_id": candidate_id}))
        assert result["candidate_id"] == candidate_id
        assert result["count"] == 3
        assert {sc["name"] for sc in result["softcores"]} >= {
            "conv1_pos0_0", "conv1_pos1_0", "fc1_tile_0_64",
        }

    def test_unknown_candidate_raises(self, compiled_backend):
        backend, _ = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError, match="unknown candidate"):
            decl.invoke({"candidate_id": "nope"})


class TestInspectLayerBreakdown:
    def test_collapses_to_unique_layers(self, compiled_backend):
        backend, candidate_id = compiled_backend
        decl = _by_name(
            build_introspection_tools(backend), "inspect_layer_breakdown",
        )
        result = json.loads(decl.invoke({"candidate_id": candidate_id}))
        layers = {row["layer"] for row in result["per_layer"]}
        assert layers == {"conv1", "fc1"}
        assert result["layer_count"] == 2


class TestInspectLayoutStats:
    def test_returns_layout_stats_and_objectives(self, compiled_backend):
        backend, candidate_id = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_layout_stats")
        result = json.loads(decl.invoke({"candidate_id": candidate_id}))
        assert "layout_stats" in result and result["layout_stats"]
        assert "hw_objectives" in result and "total_param_capacity" in result["hw_objectives"]


class TestListObjectives:
    def test_returns_objective_catalog(self, compiled_backend):
        backend, _ = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "list_objectives")
        result = json.loads(decl.invoke({}))
        names = {entry["name"] for entry in result["objectives"]}
        assert names >= {
            "total_param_capacity", "param_utilization_pct", "fragmentation_pct",
        }
        for entry in result["objectives"]:
            assert entry["goal"] in ("min", "max")

    def test_returns_empty_when_no_candidate_compiled(self):
        backend = MimarsinanLayoutBackend()
        decl = _by_name(build_introspection_tools(backend), "list_objectives")
        result = json.loads(decl.invoke({}))
        assert result == {"objectives": []}


class TestArgsValidation:
    def test_missing_candidate_id_is_validation_error(self, compiled_backend):
        backend, _ = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError):
            decl.invoke({})  # candidate_id is required
