"""Resource sources must store each distinct payload ONCE on disk.

Measured defect (2026-08-07): every run wrote **179 GB** of resource sources
(95 GB from SCM, 84 GB from HCM) because each of 4,925 cores persisted its own
dense copy of what is really 27 distinct payloads — the same per-instance
duplication already removed from the pickles. Sources now reference a
content-addressed payload store; equal content is written once.
"""

import numpy as np
import pytest

from mimarsinan.gui.resources import HeatmapSource
from mimarsinan.gui.runtime.persistence.payload_store import reset_payload_session
from mimarsinan.gui.runtime.persistence.resource_paths import resource_source_root
from mimarsinan.gui.runtime.persistence.resource_sources import (
    load_resource_source,
    save_resource_source,
)


@pytest.fixture(autouse=True)
def _clean_session():
    reset_payload_session()
    yield
    reset_payload_session()


def _tree_bytes(path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _payload(seed=1, rows=256, cols=256):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(rows, cols)).astype(np.float64)


class TestPayloadsAreStoredOnce:
    def test_identical_payloads_do_not_multiply_disk(self, tmp_path):
        shared = _payload()
        for i in range(12):
            save_resource_source(
                str(tmp_path), "Step", "heatmap", f"core{i}",
                HeatmapSource(shared, copy=False),
            )
        total = _tree_bytes(resource_source_root(str(tmp_path)))
        one = shared.nbytes
        assert total < one * 2, (
            f"12 sources of one payload wrote {total/1e6:.1f} MB; a shared "
            f"payload store should cost ~{one/1e6:.1f} MB"
        )

    def test_views_of_one_base_share_by_window(self, tmp_path):
        base = _payload()
        for i in range(8):
            save_resource_source(
                str(tmp_path), "Step", "heatmap", f"slice{i}",
                HeatmapSource(base[:, 0:64], copy=False),
            )
        total = _tree_bytes(resource_source_root(str(tmp_path)))
        assert total < base[:, 0:64].nbytes * 2, "equal views must share one payload"

    def test_distinct_payloads_are_all_kept(self, tmp_path):
        a, b = _payload(seed=1), _payload(seed=2)
        save_resource_source(str(tmp_path), "Step", "heatmap", "a",
                             HeatmapSource(a, copy=False))
        save_resource_source(str(tmp_path), "Step", "heatmap", "b",
                             HeatmapSource(b, copy=False))
        total = _tree_bytes(resource_source_root(str(tmp_path)))
        assert total > a.nbytes, "distinct payloads must not be conflated"
        got_a = load_resource_source(str(tmp_path), "Step", "heatmap", "a")
        got_b = load_resource_source(str(tmp_path), "Step", "heatmap", "b")
        assert np.array_equal(got_a.to_state()["matrix"], a)
        assert np.array_equal(got_b.to_state()["matrix"], b)


class TestRoundTripIsBitExact:
    @pytest.mark.parametrize("maker", [
        lambda: _payload(),
        lambda: np.zeros((300, 300), dtype=np.float64),
        lambda: np.full((200, 200), -0.0, dtype=np.float64),
        lambda: np.arange(40000, dtype=np.int8).astype(np.int8).reshape(200, 200),
    ])
    def test_loaded_payload_matches_saved(self, tmp_path, maker):
        arr = maker()
        save_resource_source(str(tmp_path), "Step", "heatmap", "x",
                             HeatmapSource(arr, copy=False))
        got = load_resource_source(str(tmp_path), "Step", "heatmap", "x")
        values = got.to_state()["matrix"]
        assert values.dtype == arr.dtype and values.shape == arr.shape
        assert values.tobytes() == arr.tobytes(), "payload must round-trip bit-exactly"

    def test_reload_after_session_reset_still_resolves(self, tmp_path):
        """A monitor opening the run later has no in-process session state."""
        arr = _payload()
        save_resource_source(str(tmp_path), "Step", "heatmap", "x",
                             HeatmapSource(arr, copy=False))
        reset_payload_session()
        got = load_resource_source(str(tmp_path), "Step", "heatmap", "x")
        assert np.array_equal(got.to_state()["matrix"], arr)


class TestMissingPayloadDegrades:
    def test_a_deleted_payload_yields_none_not_a_crash(self, tmp_path):
        arr = _payload()
        save_resource_source(str(tmp_path), "Step", "heatmap", "x",
                             HeatmapSource(arr, copy=False))
        root = resource_source_root(str(tmp_path))
        for p in root.rglob("*"):
            if p.is_file() and p.suffix == ".npy":
                p.unlink()
        reset_payload_session()
        assert load_resource_source(str(tmp_path), "Step", "heatmap", "x") is None
