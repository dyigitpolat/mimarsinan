"""Host-side spike-train file I/O — Python writer matches the C++ reader layout."""

from __future__ import annotations

import os
import numpy as np
import pytest

from mimarsinan.common.file_utils import (
    save_spike_train_inputs_to_files,
    spike_train_to_file,
)


def _parse_spike_train_file(path: str):
    """Mirror of nevresim::SpikeTrainInputLoader: parses header + flat cycle-major spikes."""
    with open(path, "r") as f:
        tokens = f.read().split()
    target = int(tokens[0])
    batch_size = int(tokens[1])
    input_size = int(tokens[2])
    simulation_length = int(tokens[3])
    expected = input_size * simulation_length
    values = [float(v) for v in tokens[4 : 4 + expected]]
    return {
        "target": target,
        "batch_size": batch_size,
        "input_size": input_size,
        "simulation_length": simulation_length,
        "values": values,
    }


def test_spike_train_to_file_cycle_major_ordering(tmp_path) -> None:
    spikes = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )  # (T=2, D=3) cycle-major: cycle 0 then cycle 1
    target = 7
    path = str(tmp_path / "0.txt")
    spike_train_to_file(spikes, target, path, simulation_length=2)
    parsed = _parse_spike_train_file(path)
    assert parsed["target"] == 7
    assert parsed["batch_size"] == 1
    assert parsed["input_size"] == 3
    assert parsed["simulation_length"] == 2
    # Flat cycle-major: data[cycle*D + neuron] = spikes[cycle, neuron]
    assert parsed["values"] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_spike_train_to_file_accepts_dt_layout(tmp_path) -> None:
    """A ``(D, T)`` layout where ``T`` is the second axis is auto-transposed."""
    spikes_dt = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )  # (D=3, T=2)
    path = str(tmp_path / "0.txt")
    spike_train_to_file(spikes_dt, target=2, filename=path, simulation_length=2)
    parsed = _parse_spike_train_file(path)
    assert parsed["values"] == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_spike_train_to_file_rejects_ambiguous_shape(tmp_path) -> None:
    """An input where no axis matches ``simulation_length`` is a hard error."""
    spikes = np.zeros((4, 5), dtype=np.float64)
    path = str(tmp_path / "0.txt")
    with pytest.raises(ValueError, match="simulation_length"):
        spike_train_to_file(spikes, target=0, filename=path, simulation_length=3)


def test_save_spike_train_inputs_roundtrip(tmp_path) -> None:
    T = 3
    D = 2
    samples = [
        (np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64),
         np.array([0.0, 1.0, 0.0])),  # target = 1
        (np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
         np.array([1.0, 0.0, 0.0])),  # target = 0
    ]
    save_spike_train_inputs_to_files(
        str(tmp_path), iter(samples), input_count=2, simulation_length=T,
    )
    parsed0 = _parse_spike_train_file(str(tmp_path / "inputs" / "0.txt"))
    parsed1 = _parse_spike_train_file(str(tmp_path / "inputs" / "1.txt"))
    assert parsed0["target"] == 1
    assert parsed0["input_size"] == D
    assert parsed0["simulation_length"] == T
    assert parsed0["values"] == [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert parsed1["target"] == 0
    assert parsed1["values"] == [0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
