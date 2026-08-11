"""NevresimDriver returns its total-spikes read (W4.3) instead of dropping it.

The ``"  Total spikes: ..."`` print is a pinned legacy surface; the figure now
ALSO returns from ``_simulator_output_to_predictions`` / ``predict_spiking``
so the nevresim deployment-record fragment can carry the measurement.
"""

from types import SimpleNamespace

import numpy as np

from mimarsinan.chip_simulation.nevresim.nevresim_driver import NevresimDriver


_OUTPUT = [0.0, 1.0, 3.0, 2.0, 5.0, 0.0, 1.0, 0.0]  # 2 samples x 4 classes


def _bare_driver():
    """Driver shell without __init__ (no nevresim checkout / chip build)."""
    driver = NevresimDriver.__new__(NevresimDriver)
    driver.chip = SimpleNamespace(output_size=4)
    return driver


def test_predictions_and_total_spikes_return_with_the_print_pinned(capsys):
    driver = _bare_driver()
    predictions, total_spikes = driver._simulator_output_to_predictions(
        _OUTPUT, 4,
    )
    assert total_spikes == 12.0
    np.testing.assert_array_equal(predictions, np.array([2, 0]))
    # The existing stdout surface stays byte-identical.
    assert capsys.readouterr().out == "  Total spikes: 12.0\n"


def test_predict_spiking_returns_the_pair(capsys):
    driver = _bare_driver()
    driver._run_simulator = lambda *args, **kwargs: (list(_OUTPUT), 2)
    predictions, total_spikes = driver.predict_spiking(
        input_loader=[], simulation_length=8, latency=1,
    )
    assert total_spikes == 12.0
    np.testing.assert_array_equal(predictions, np.array([2, 0]))
    assert "  Total spikes: 12.0" in capsys.readouterr().out
