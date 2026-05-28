"""Tests for nevresim compile profiling helpers."""

from __future__ import annotations

from mimarsinan.chip_simulation.nevresim.compile_cache import (
    NevresimCompileCache,
    cache_key,
    mapping_connectivity_hash,
    policy_hash,
)
from mimarsinan.chip_simulation.nevresim.profiling.compile_profile import (
    NevresimCompileProfile,
    correlate_compile_vs_metric,
)
from mimarsinan.chip_simulation.nevresim.profiling.synthetic_mapping import build_synthetic_mapping


def test_mapping_hash_stable() -> None:
    hcm, _ = build_synthetic_mapping("contiguous_input", 3, 8, 4)
    assert mapping_connectivity_hash(hcm) == mapping_connectivity_hash(hcm)


def test_mapping_hash_changes_with_topology() -> None:
    a, _ = build_synthetic_mapping("contiguous_input", 3, 8, 4)
    b, _ = build_synthetic_mapping("fragmented_crosscore", 3, 8, 4)
    assert mapping_connectivity_hash(a) != mapping_connectivity_hash(b)


def test_policy_hash_includes_connectivity_mode() -> None:
    base = dict(
        spiking_mode="lif",
        spike_generation_mode="Stochastic",
        firing_mode="Default",
        thresholding_mode="<=",
        weight_type_name="double",
        threshold_type_name="double",
        simulation_length=4,
        latency=0,
    )
    static = policy_hash(**base, connectivity_mode="compile_time")
    runtime = policy_hash(**base, connectivity_mode="runtime")
    assert static != runtime


def test_compile_cache_roundtrip(tmp_path) -> None:
    cache = NevresimCompileCache(tmp_path)
    key = cache_key("abc", "def")
    src = tmp_path / "src_bin"
    src.write_bytes(b"simulator")
    stored = cache.store_binary(key, src, metadata={"ok": True})
    assert cache.get_binary(key) == stored
    cache.invalidate(key)
    assert cache.get_binary(key) is None


def test_correlate_compile_vs_metric() -> None:
    rows = [
        NevresimCompileProfile(total_axon_slots=100, compile_s=1.0, compile_success=True),
        NevresimCompileProfile(total_axon_slots=200, compile_s=2.0, compile_success=True),
        NevresimCompileProfile(total_axon_slots=300, compile_s=3.0, compile_success=True),
    ]
    r = correlate_compile_vs_metric(rows, "total_axon_slots")
    assert r is not None
    assert abs(r - 1.0) < 1e-6
