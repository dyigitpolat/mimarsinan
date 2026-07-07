"""The run-scoped ledger SSOT: one key registry, one cache seam, one lifecycle.

``run_ledger`` owns the set of run-scoped pipeline-cache keys (the D-hat
high-water anchor and the endpoint-step budget) plus the duck-typed cache
write/remove/snapshot/restore seam every consumer shares — the fresh-run
reset (pipeline.run), the per-draw isolation (conversion_draws), and the
domain ledgers themselves.
"""

from __future__ import annotations

import pytest

from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.tuning.orchestration import (
    dhat_highwater,
    endpoint_steps,
    run_ledger,
)


def _dict_cache():
    return {}


def _pipeline_cache():
    return PipelineCache()


@pytest.mark.parametrize("make", [_dict_cache, _pipeline_cache],
                         ids=["dict-cache", "pipeline-cache"])
class TestCacheSeam:
    def test_write_then_read(self, make):
        cache = make()
        run_ledger.cache_write(cache, "k", 3)
        assert cache.get("k") == 3

    def test_remove_is_idempotent(self, make):
        cache = make()
        run_ledger.cache_write(cache, "k", 3)
        run_ledger.cache_remove(cache, "k")
        run_ledger.cache_remove(cache, "k")
        assert cache.get("k") is None


class TestKeyRegistry:
    def test_registry_is_the_two_run_scoped_ledgers(self):
        assert set(run_ledger.RUN_SCOPED_KEYS) == {
            dhat_highwater.HIGHWATER_CACHE_KEY,
            endpoint_steps.STEPS_CACHE_KEY,
        }

    def test_domain_modules_share_the_registry_constants(self):
        assert dhat_highwater.HIGHWATER_CACHE_KEY == run_ledger.HIGHWATER_CACHE_KEY
        assert endpoint_steps.STEPS_CACHE_KEY == run_ledger.ENDPOINT_STEPS_CACHE_KEY


@pytest.mark.parametrize("make", [_dict_cache, _pipeline_cache],
                         ids=["dict-cache", "pipeline-cache"])
class TestLifecycle:
    def test_reset_clears_every_run_scoped_key(self, make):
        cache = make()
        for key in run_ledger.RUN_SCOPED_KEYS:
            run_ledger.cache_write(cache, key, 1)
        run_ledger.reset(cache)
        assert all(cache.get(key) is None for key in run_ledger.RUN_SCOPED_KEYS)

    def test_snapshot_restore_round_trips_values(self, make):
        cache = make()
        run_ledger.cache_write(cache, run_ledger.HIGHWATER_CACHE_KEY, 0.9)
        snapshot = run_ledger.snapshot(cache)
        run_ledger.cache_write(cache, run_ledger.HIGHWATER_CACHE_KEY, 0.99)
        run_ledger.cache_write(cache, run_ledger.ENDPOINT_STEPS_CACHE_KEY, 500)
        run_ledger.restore(cache, snapshot)
        assert cache.get(run_ledger.HIGHWATER_CACHE_KEY) == 0.9
        assert cache.get(run_ledger.ENDPOINT_STEPS_CACHE_KEY) is None

    def test_restore_removes_keys_absent_from_the_snapshot(self, make):
        cache = make()
        snapshot = run_ledger.snapshot(cache)
        for key in run_ledger.RUN_SCOPED_KEYS:
            run_ledger.cache_write(cache, key, 7)
        run_ledger.restore(cache, snapshot)
        assert all(cache.get(key) is None for key in run_ledger.RUN_SCOPED_KEYS)
