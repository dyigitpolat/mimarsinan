import numpy as np

from mimarsinan.chip_simulation.ttfs.ttfs_encoding import (
    ttfs_always_on_spike_times_1based,
    ttfs_input_spike_times_1based,
    ttfs_latched_spike_train,
    ttfs_spike_time,
)


def test_spike_time_matches_nevresim_formula():
    s = 8
    rates = np.array([[1.0, 0.5, 0.0]], dtype=np.float64)
    t = ttfs_spike_time(rates, s)
    assert t[0, 0] == 0
    assert t[0, 1] == 4
    assert t[0, 2] == 8


def test_latched_train_high_from_spike_time():
    rates = np.array([[0.5]], dtype=np.float64)
    train = ttfs_latched_spike_train(rates, 4)
    assert train.shape == (1, 1, 4)
    assert train[0, 0, 0] == 0
    assert train[0, 0, 1] == 0
    assert train[0, 0, 2] == 1
    assert train[0, 0, 3] == 1


def test_always_on_fires_cycle_zero_only():
    assert ttfs_always_on_spike_times_1based(4) == [1]


def test_input_spike_times_1based_latched():
    times = ttfs_input_spike_times_1based(0.5, 4)
    assert times == [3, 4]
