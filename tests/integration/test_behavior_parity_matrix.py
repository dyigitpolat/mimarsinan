"""HCM reference parity matrix across simulators and behavior dimensions.

Each cell compares spike-count records from HCM (reference) against one
backend on the same single-core toy hybrid mapping:

  * **loihi** — ``LavaLoihiRunner.run_segments_from_reference``
  * **nevresim** — ``NevresimDriver`` compile + execute (rebuilt every run)
  * **sanafe** — ``SanafeRunner`` + ``SanafeRunRecord.to_hcm_subset()``
    (auto-rebuilds ``libmimarsinan_*.so`` when plugin sources are stale)

Backends that compile or link artifacts (``nevresim``, ``sanafe``) are
prepared via :func:`parity_harness.ensure_backend_ready`. Nevresim uses
deployment default ``nevresim_connectivity_mode=runtime``. Lava remains
opt-in (skipped when not installed).

These tests are **not** marked ``slow`` (~9s full matrix with runtime nevresim).
"""

from __future__ import annotations

import pytest

from integration.parity_harness import (
    SUPPORTED_FIRING_MODES,
    SUPPORTED_SPIKE_GENERATION_MODES,
    SUPPORTED_THRESHOLDING_MODES,
    BackendName,
    backend_requires_rebuild,
    default_behavior,
    ensure_backend_ready,
    run_backend_parity,
)


_BACKENDS: tuple[BackendName, ...] = ("loihi", "nevresim", "sanafe")


@pytest.mark.parametrize("firing_mode", SUPPORTED_FIRING_MODES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.integration
def test_firing_mode_backend_parity_matrix(firing_mode, backend):
    """HCM spike counts must match each backend for every firing mode."""
    ensure_backend_ready(backend)
    behavior = default_behavior(firing_mode=firing_mode)
    result = run_backend_parity(backend, behavior, T=4)
    assert result.ok, result.first_diff_message()


@pytest.mark.parametrize("thresholding_mode", SUPPORTED_THRESHOLDING_MODES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.integration
def test_thresholding_backend_parity_matrix(thresholding_mode, backend):
    """Comparison policy ``<`` vs ``<=`` must agree across all backends."""
    ensure_backend_ready(backend)
    behavior = default_behavior(thresholding_mode=thresholding_mode)
    result = run_backend_parity(backend, behavior, T=4)
    assert result.ok, result.first_diff_message()


@pytest.mark.parametrize("spike_generation_mode", SUPPORTED_SPIKE_GENERATION_MODES)
@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.integration
def test_spike_encode_backend_parity_matrix(spike_generation_mode, backend):
    """Spike-encoding modes must agree across all backends."""
    ensure_backend_ready(backend)
    behavior = default_behavior(spike_generation_mode=spike_generation_mode)
    result = run_backend_parity(backend, behavior, T=4)
    assert result.ok, result.first_diff_message()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.integration
def test_combined_novena_inclusive_backend_parity_matrix(backend):
    """Novena reset + inclusive compare smoke on every backend."""
    ensure_backend_ready(backend)
    behavior = default_behavior(firing_mode="Novena", thresholding_mode="<=")
    result = run_backend_parity(backend, behavior, T=4)
    assert result.ok, result.first_diff_message()


@pytest.mark.parametrize("backend", tuple(b for b in _BACKENDS if backend_requires_rebuild(b)))
@pytest.mark.integration
def test_rebuild_backends_refresh_stale_plugins(backend):
    """Rebuild-dependent backends succeed after an explicit stale refresh."""
    ensure_backend_ready(backend, rebuild_if_stale=True)
    behavior = default_behavior(firing_mode="Novena")
    result = run_backend_parity(backend, behavior, T=4)
    assert result.ok, result.first_diff_message()
