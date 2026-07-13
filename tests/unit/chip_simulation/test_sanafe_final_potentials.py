"""[C2/WS-V] SANA-FE final-membrane read: plugin somas expose ``get_potential``
via the potential trace; the runner lands end-of-window potentials for LIF
cores in the record structure (additive, default-off)."""

from __future__ import annotations

import numpy as np

from mimarsinan.chip_simulation.sanafe.analysis import read_final_core_potentials
from mimarsinan.chip_simulation.sanafe.records import SanafeCoreRecord
from mimarsinan.chip_simulation.sanafe.records.energy import SanafeEnergyBreakdown


class _FakeChip:
    """Chip stub with the ``mapped_neuron_groups`` surface the reader walks."""

    def __init__(self, groups):
        self.mapped_neuron_groups = groups


def _core_record(**overrides) -> SanafeCoreRecord:
    base = dict(
        core_index=0,
        n_neurons=2,
        n_axons_used=1,
        core_latency=0,
        has_hardware_bias=False,
        n_always_on_axons=0,
        spikes_fired=0,
        input_spike_count=np.zeros(1, dtype=np.int64),
        output_spike_count=np.zeros(2, dtype=np.int64),
        energy=SanafeEnergyBreakdown(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    base.update(overrides)
    return SanafeCoreRecord(**base)


class TestFinalPotentialRecordField:
    def test_final_potential_defaults_to_none(self):
        assert _core_record().final_potential is None

    def test_final_potential_is_additive(self):
        rec = _core_record(final_potential=np.asarray([0.25, -0.5]))
        assert rec.final_potential is not None
        np.testing.assert_array_equal(rec.final_potential, [0.25, -0.5])


class TestReadFinalCorePotentials:
    """The reader consumes the LAST ``potential_trace`` row (each soma idles
    outside its active window, so the last row IS the end-of-window state)."""

    def _chip_and_results(self):
        chip = _FakeChip({"core0": [0, 1], "core1": [0], "core0_in": [0]})
        results = {
            "potential_trace": [
                [0.0, 0.0, 0.0],
                [0.5, -0.25, 1.5],
            ],
        }
        return chip, results

    def test_reads_last_row_slice_for_core(self):
        chip, results = self._chip_and_results()
        np.testing.assert_allclose(
            read_final_core_potentials(chip, 0, 2, results), [0.5, -0.25],
        )
        np.testing.assert_allclose(
            read_final_core_potentials(chip, 1, 1, results), [1.5],
        )

    def test_missing_trace_returns_zeros(self):
        chip, _ = self._chip_and_results()
        np.testing.assert_array_equal(
            read_final_core_potentials(chip, 0, 2, {}), [0.0, 0.0],
        )


class TestRunnerFlagDefaultOff:
    def test_runner_read_final_potentials_default_off_and_armable(self):
        from types import SimpleNamespace

        from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner

        mapping = SimpleNamespace(stages=[])
        assert SanafeRunner(
            mapping=mapping, simulation_length=4,
        ).read_final_potentials is False
        assert SanafeRunner(
            mapping=mapping, simulation_length=4, read_final_potentials=True,
        ).read_final_potentials is True
