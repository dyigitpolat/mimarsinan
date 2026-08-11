"""Tests for the four ``ToolDecl``s exposed by ``MimarsinanLayoutBackend``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from compilagent import Plan

from mimarsinan.search.optimizers.compilagent.backend import MimarsinanLayoutBackend
from mimarsinan.search.optimizers.compilagent.tools import build_introspection_tools
from mimarsinan.search.optimizers.compilagent.workload import (
    register_problem,
    unregister_problem,
)

# The REAL problem the backend adapts — see real_problem.py.
from .real_problem import HW_OBJECTIVES, make_problem, make_workload


@pytest.fixture
def compiled_backend(tmp_path: Path):
    workload_id = "real_layout_tools_test"
    problem = make_problem("hardware")
    register_problem(workload_id, problem)
    backend = MimarsinanLayoutBackend()
    workload = make_workload(workload_id)
    cdir = tmp_path / "cand-abc"
    cdir.mkdir()
    result = backend.compile(workload, Plan(), artifact_dir=cdir)
    assert result.ok, result.diagnostics
    candidate_id = cdir.name
    try:
        yield backend, candidate_id, problem
    finally:
        unregister_problem(workload_id)


def _by_name(decls, name):
    return next(d for d in decls if d.name == name)


class TestSurfaceShape:
    def test_four_tools_returned(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decls = build_introspection_tools(backend)
        names = sorted(d.name for d in decls)
        assert names == [
            "inspect_layer_breakdown",
            "inspect_layout_stats",
            "inspect_softcores",
            "list_objectives",
        ]

    def test_all_tools_are_read_only(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decls = build_introspection_tools(backend)
        assert all(d.read_only for d in decls)


class TestInspectSoftcores:
    def test_returns_the_candidates_own_softcores(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        result = json.loads(decl.invoke({"candidate_id": candidate_id}))
        expected = problem.candidate_layout(
            backend.get_candidate_payload(candidate_id)["config"]
        ).softcores
        assert result["candidate_id"] == candidate_id
        assert result["count"] == len(expected) > 0
        assert [sc["input_count"] for sc in result["softcores"]] == [
            sc.input_count for sc in expected
        ]
        assert [sc["output_count"] for sc in result["softcores"]] == [
            sc.output_count for sc in expected
        ]

    def test_unknown_candidate_raises(self, compiled_backend):
        backend, _, _problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError, match="unknown candidate"):
            decl.invoke({"candidate_id": "nope"})


class TestInspectLayerBreakdown:
    def test_collapses_softcores_into_layer_rows(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        decl = _by_name(
            build_introspection_tools(backend), "inspect_layer_breakdown",
        )
        result = json.loads(decl.invoke({"candidate_id": candidate_id}))
        softcores = problem.candidate_layout(
            backend.get_candidate_payload(candidate_id)["config"]
        ).softcores
        assert result["layer_count"] == len(result["per_layer"]) > 0
        assert sum(row["softcore_count"] for row in result["per_layer"]) == len(
            softcores
        )
        assert sum(row["total_area"] for row in result["per_layer"]) == sum(
            sc.area for sc in softcores
        )


class TestInspectLayoutStats:
    def test_returns_layout_stats_and_objectives(self, compiled_backend):
        backend, candidate_id, problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_layout_stats")
        result = json.loads(decl.invoke({"candidate_id": candidate_id}))
        assert result["layout_stats"]["feasible"] is True
        assert set(result["hw_objectives"]) >= set(HW_OBJECTIVES)
        scored = problem.evaluate(
            backend.get_candidate_payload(candidate_id)["config"]
        )
        for name, value in scored.items():
            assert result["hw_objectives"][name] == value


class TestListObjectives:
    def test_returns_objective_catalog(self, compiled_backend):
        backend, _, _problem = compiled_backend
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
        backend, _, _problem = compiled_backend
        decl = _by_name(build_introspection_tools(backend), "inspect_softcores")
        with pytest.raises(ValueError):
            decl.invoke({})  # candidate_id is required
