"""Stale hybrid-mapping detection across pipeline resumes.

The packed ``hybrid_mapping`` is cached under a flat key and derived from the
SCM step's ``ir_graph``. A resumed run regenerates the ir_graph but used to
reuse the cached mapping built from the *previous* run's graph — the HCM (and
every downstream simulation) then silently measured stale weights (2026-06-08
incident: SCM 0.954 vs HCM 0.916 on the cascaded+offload rerun). The contract:
every fresh IRGraph carries a unique ``build_token``; the derived mapping
records it; the consumer rebuilds on mismatch.
"""

from __future__ import annotations

from conftest import MockPipeline, make_tiny_ir_graph

from mimarsinan.pipelining.core.hybrid_mapping_consumer import (
    load_hybrid_mapping_for_step,
)
from mimarsinan.pipelining.core.simulation_factory import (
    build_hybrid_mapping_for_pipeline,
)

_PLATFORM = {"cores": [{"max_axons": 256, "max_neurons": 256, "count": 20}]}


class _Cache(dict):
    def add(self, key, obj, strategy="basic"):
        self[key] = obj


class _Step:
    def __init__(self, ir_graph):
        self._entries = {
            "ir_graph": ir_graph,
            "platform_constraints_resolved": _PLATFORM,
        }

    def get_entry(self, key):
        return self._entries[key]


def _pipeline():
    p = MockPipeline()
    p.cache = _Cache()
    return p


class TestBuildToken:
    def test_fresh_ir_graphs_get_distinct_tokens(self):
        a, b = make_tiny_ir_graph(), make_tiny_ir_graph()
        assert a.build_token and b.build_token
        assert a.build_token != b.build_token

    def test_hybrid_mapping_records_source_token(self):
        ir = make_tiny_ir_graph()
        mapping = build_hybrid_mapping_for_pipeline(ir, _PLATFORM)
        assert mapping.source_ir_build_token == ir.build_token


class TestLoadHybridMapping:
    def test_reuses_cached_mapping_for_same_ir_graph(self):
        p = _pipeline()
        ir = make_tiny_ir_graph()
        first = load_hybrid_mapping_for_step(p, _Step(ir))
        second = load_hybrid_mapping_for_step(p, _Step(ir))
        assert second is first

    def test_rebuilds_when_cached_mapping_is_stale(self):
        p = _pipeline()
        old_ir, new_ir = make_tiny_ir_graph(), make_tiny_ir_graph()
        stale = load_hybrid_mapping_for_step(p, _Step(old_ir))
        fresh = load_hybrid_mapping_for_step(p, _Step(new_ir))
        assert fresh is not stale
        assert fresh.source_ir_build_token == new_ir.build_token
        assert p.cache.get("hybrid_mapping") is fresh

    def test_rebuilds_when_cached_mapping_predates_tokens(self):
        """A legacy pickled mapping (no token) must not be trusted against a
        token-stamped ir_graph."""
        p = _pipeline()
        ir = make_tiny_ir_graph()
        legacy = build_hybrid_mapping_for_pipeline(ir, _PLATFORM)
        del legacy.source_ir_build_token
        p.cache.add("hybrid_mapping", legacy, "pickle")
        loaded = load_hybrid_mapping_for_step(p, _Step(ir))
        assert loaded is not legacy
        assert loaded.source_ir_build_token == ir.build_token

    def test_legacy_mapping_and_legacy_ir_graph_still_pair(self):
        """Both artifacts predating tokens (old cached runs) keep working."""
        p = _pipeline()
        ir = make_tiny_ir_graph()
        legacy = build_hybrid_mapping_for_pipeline(ir, _PLATFORM)
        del legacy.source_ir_build_token
        ir.build_token = None
        p.cache.add("hybrid_mapping", legacy, "pickle")
        loaded = load_hybrid_mapping_for_step(p, _Step(ir))
        assert loaded is legacy
