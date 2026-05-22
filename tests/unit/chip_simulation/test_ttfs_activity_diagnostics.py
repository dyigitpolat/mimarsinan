"""TTFS contract vs hardware vs event diagnostics."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.sanafe.records import (
    SanafeCoreRecord,
    SanafeEnergyBreakdown,
)
from mimarsinan.chip_simulation.sanafe.runner import (
    _build_spike_capture_warning,
    _compute_ttfs_activity_diagnostics,
)
from mimarsinan.chip_simulation.ttfs_recorder import CoreTtfsActivations


def _energy() -> SanafeEnergyBreakdown:
    return SanafeEnergyBreakdown(
        synapse_j=0.1, dendrite_j=0.1, soma_j=0.1, network_j=0.1, total_j=0.4,
    )


def test_compute_ttfs_activity_diagnostics_mismatch():
    contract = [
        CoreTtfsActivations(0, 1, np.array([0.5], dtype=np.float64)),
        CoreTtfsActivations(1, 1, np.array([0.0], dtype=np.float64)),
    ]
    per_core = [
        SanafeCoreRecord(
            core_index=0, n_neurons=1, n_axons_used=1, core_latency=0,
            has_hardware_bias=False, n_always_on_axons=0, spikes_fired=0,
            input_spike_count=np.zeros(1, dtype=np.int64),
            output_spike_count=np.zeros(1, dtype=np.int64),
            energy=_energy(),
            output_activation=np.array([0.25], dtype=np.float64),
        ),
        SanafeCoreRecord(
            core_index=1, n_neurons=1, n_axons_used=1, core_latency=0,
            has_hardware_bias=False, n_always_on_axons=0, spikes_fired=2,
            input_spike_count=np.zeros(1, dtype=np.int64),
            output_spike_count=np.array([2], dtype=np.int64),
            energy=_energy(),
            output_activation=np.array([0.0], dtype=np.float64),
        ),
    ]
    diag = _compute_ttfs_activity_diagnostics(contract, per_core)
    assert diag["ttfs_contract_active_cores"] == 1
    assert diag["ttfs_hardware_active_cores"] == 1
    assert diag["ttfs_event_active_cores"] == 1
    assert diag["ttfs_activation_event_mismatch_count"] == 1


def test_spike_capture_warning_ttfs_mismatch_priority():
    msg = _build_spike_capture_warning(
        chip_spike_count=1000,
        lif_spike_count=0,
        input_path_packets=10,
        spike_trace_parse_skipped=0,
        ttfs_hardware_active=5,
        ttfs_event_active=0,
        ttfs_mismatch_count=5,
    )
    assert msg is not None
    assert "TTFS activations" in msg
    assert "event-emission" in msg
