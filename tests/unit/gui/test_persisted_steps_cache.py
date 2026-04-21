"""Tests for the mtime-keyed cache over :func:`load_persisted_steps`.

Historical runs and the ``ProcessManager`` poll ``steps.json`` on every
REST call. For a long run with many snapshots this file can be tens of
megabytes and reparsing it several times per second dominates the GUI
server's CPU.

The cache keys on ``(abspath, mtime, size)`` so:

* Repeated reads when the file hasn't changed hit the cache.
* A new write (different mtime or size) invalidates the entry so the
  caller never sees a stale payload.
* Eviction is LRU-bounded so the cache cannot grow unbounded when a
  user browses many historical runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from mimarsinan.gui.persistence import (
    _GUI_STATE_DIR,
    _STEPS_FILENAME,
    load_persisted_steps,
    load_persisted_steps_cache_clear,
    load_persisted_steps_cache_info,
)


def _write_steps(working_dir: Path, payload: dict) -> None:
    state_dir = working_dir / _GUI_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / _STEPS_FILENAME).write_text(json.dumps(payload))


class TestPersistedStepsCache:
    def setup_method(self) -> None:
        load_persisted_steps_cache_clear()

    def teardown_method(self) -> None:
        load_persisted_steps_cache_clear()

    def test_repeated_reads_hit_cache(self, tmp_path: Path) -> None:
        _write_steps(tmp_path, {"steps": {"s1": {"start_time": 1.0}}})
        load_persisted_steps(str(tmp_path))
        load_persisted_steps(str(tmp_path))
        load_persisted_steps(str(tmp_path))

        info = load_persisted_steps_cache_info()
        assert info.hits >= 2, f"expected cache hits, got {info}"
        assert info.misses >= 1

    def test_modifying_file_invalidates_cache(self, tmp_path: Path) -> None:
        _write_steps(tmp_path, {"steps": {"s1": {"value": 1}}})
        first = load_persisted_steps(str(tmp_path))
        assert first["s1"]["value"] == 1

        time.sleep(0.01)
        _write_steps(tmp_path, {"steps": {"s1": {"value": 2}}})
        # Force mtime change so the cache key differs even on fast FSes.
        steps_file = tmp_path / _GUI_STATE_DIR / _STEPS_FILENAME
        later = time.time() + 1.0
        import os
        os.utime(steps_file, (later, later))

        second = load_persisted_steps(str(tmp_path))
        assert second["s1"]["value"] == 2

    def test_missing_file_returns_empty_without_caching_stale(self, tmp_path: Path) -> None:
        # No steps.json yet.
        assert load_persisted_steps(str(tmp_path)) == {}
        # Create the file, and the next call should pick it up.
        _write_steps(tmp_path, {"steps": {"s1": {}}})
        steps_file = tmp_path / _GUI_STATE_DIR / _STEPS_FILENAME
        later = time.time() + 1.0
        import os
        os.utime(steps_file, (later, later))
        result = load_persisted_steps(str(tmp_path))
        assert "s1" in result

    def test_cache_info_exposes_size_bound(self) -> None:
        info = load_persisted_steps_cache_info()
        assert info.maxsize is not None, "cache must have an LRU bound"
        assert info.maxsize > 0

    def test_different_runs_get_separate_entries(self, tmp_path: Path) -> None:
        run_a = tmp_path / "runA"
        run_b = tmp_path / "runB"
        _write_steps(run_a, {"steps": {"a": {}}})
        _write_steps(run_b, {"steps": {"b": {}}})

        assert "a" in load_persisted_steps(str(run_a))
        assert "b" in load_persisted_steps(str(run_b))
        assert "a" in load_persisted_steps(str(run_a))
        info = load_persisted_steps_cache_info()
        assert info.hits >= 1
