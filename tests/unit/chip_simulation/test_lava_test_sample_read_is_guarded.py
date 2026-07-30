"""W0.8b finding 2 -- the Lava runner's test-sample read becomes a DEPLOYED accuracy.

``LavaLoihiRunner._load_test_samples`` reads a raw ``create_test_loader()`` and the
samples it returns are the ones the Lava process graph is run on; ``run()`` divides
the matches by ``N`` and stores it as ``self._accuracy``. That is the exact sibling
of ``chip_simulation/simulation_runner/core.py``, which W0.8 guarded -- and it was
left unguarded and unlisted, so a poisoned batch is baked into the chip's inputs for
the whole simulation and comes back out as a hardware result.

It is also the one place where NOT owning the batch is unambiguously wrong even on
a clean read: ``x.detach().cpu().numpy()`` SHARES storage with the loader's tensor
whenever the batch is already on the host, so the numpy arrays the simulation runs
on are views into a buffer the producer refills.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.data_handling import batch_integrity

pytest.importorskip(
    "lava.magma.core.run_configs",
    reason="lava-nc not installed; LavaLoihiRunner is unavailable",
)

from mimarsinan.chip_simulation.lava_loihi import runner as runner_mod
from mimarsinan.chip_simulation.lava_loihi.runner import LavaLoihiRunner


class _Provider:
    def get_test_batch_size(self):
        return 2


class _Factory:
    def __init__(self, passes):
        self._passes = list(passes)
        self.passes_taken = 0

    def create_data_provider(self):
        return _Provider()

    def create_test_loader(self, batch_size, provider):
        return self

    def __iter__(self):
        batches = self._passes[min(self.passes_taken, len(self._passes) - 1)]
        self.passes_taken += 1
        return iter(batches)


def _clean(values):
    x = torch.tensor(values, dtype=torch.float32).reshape(len(values), 1)
    return x, torch.zeros(len(values), dtype=torch.long)


def _nan(n=2):
    x, y = _clean([0.0] * n)
    return torch.full_like(x, float("nan")), y


def _runner(passes, max_samples=2, monkeypatch=None):
    r = LavaLoihiRunner.__new__(LavaLoihiRunner)
    r._data_loader_factory = _Factory(passes)
    r.max_samples = max_samples
    if monkeypatch is not None:
        monkeypatch.setattr(runner_mod, "shutdown_data_loader", lambda loader: None)
    return r


class TestDeployedAccuracyIsNeverMeasuredOnPoison:
    def test_a_poisoned_read_is_redone_before_it_becomes_a_chip_input(self, monkeypatch):
        r = _runner([[_nan()], [_clean([1.0, 2.0])]], monkeypatch=monkeypatch)
        x_np, _ = r._load_test_samples()
        assert np.isfinite(x_np).all()
        assert r._data_loader_factory.passes_taken == 2

    def test_persistent_corruption_fails_loud_instead_of_simulating(self, monkeypatch):
        r = _runner([[_nan()]], monkeypatch=monkeypatch)
        with pytest.raises(batch_integrity.CorruptedBatchError):
            r._load_test_samples()

    def test_a_clean_read_costs_exactly_one_pass(self, monkeypatch):
        r = _runner([[_clean([1.0, 2.0])]], monkeypatch=monkeypatch)
        r._load_test_samples()
        assert r._data_loader_factory.passes_taken == 1


class TestSimulatedInputsAreSnapshots:
    def test_the_numpy_inputs_do_not_share_storage_with_the_loader(self, monkeypatch):
        """``.cpu().numpy()`` on a host tensor is a VIEW, not a copy."""
        batch = _clean([1.0, 2.0])
        r = _runner([[batch]], monkeypatch=monkeypatch)
        x_np, _ = r._load_test_samples()
        batch[0].fill_(float("nan"))
        assert np.isfinite(x_np).all()

    def test_values_and_the_sample_cap_are_unchanged(self, monkeypatch):
        r = _runner(
            [[_clean([1.0, 2.0]), _clean([3.0, 4.0])]], max_samples=3, monkeypatch=monkeypatch
        )
        x_np, y_np = r._load_test_samples()
        assert x_np.reshape(-1).tolist() == [1.0, 2.0, 3.0]
        assert y_np.shape == (3,)
