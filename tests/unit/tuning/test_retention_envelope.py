"""The pipeline retention-envelope SSOT.

One home for the incoming model's own clean accuracy (the first seeded pipeline
metric = the pretrained/float envelope), seeded ONCE and never overwritten, and
read by the endpoint-recovery target so an absolute floor can never demand more
than the model's own envelope. Resume-safe (a run-scoped cache key, like the
D-hat high-water mark).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.tuning.orchestration import retention_envelope
from mimarsinan.tuning.orchestration.run_ledger import RUN_SCOPED_KEYS


def _dict_pipeline():
    return SimpleNamespace(cache={})


def _cache_pipeline():
    return SimpleNamespace(cache=PipelineCache())


@pytest.mark.parametrize("make", [_dict_pipeline, _cache_pipeline],
                         ids=["dict-cache", "pipeline-cache"])
class TestSeedPeek:
    def test_absent_peek_is_none(self, make):
        assert retention_envelope.peek(make()) is None

    def test_seed_writes_and_peek_reads(self, make):
        pipeline = make()
        assert retention_envelope.seed(pipeline, 0.86) == pytest.approx(0.86)
        assert retention_envelope.peek(pipeline) == pytest.approx(0.86)

    def test_seed_is_write_once_not_a_ratchet(self, make):
        # Unlike the high-water mark, the envelope is the FIXED incoming accuracy:
        # a second seed (a later, possibly higher or lower metric) never overwrites.
        pipeline = make()
        retention_envelope.seed(pipeline, 0.86)
        assert retention_envelope.seed(pipeline, 0.99) == pytest.approx(0.86)
        assert retention_envelope.seed(pipeline, 0.10) == pytest.approx(0.86)
        assert retention_envelope.peek(pipeline) == pytest.approx(0.86)

    def test_nonpositive_seed_is_ignored(self, make):
        pipeline = make()
        assert retention_envelope.seed(pipeline, 0.0) is None
        assert retention_envelope.peek(pipeline) is None


def test_key_is_run_scoped_for_resume_safety():
    # Reset by a fresh run, kept by an explicit resume, snapshot around draws.
    assert retention_envelope.RETENTION_ENVELOPE_CACHE_KEY in RUN_SCOPED_KEYS
