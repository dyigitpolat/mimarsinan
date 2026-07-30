"""W0.8b finding 3 -- the "own BEFORE verify" rule holds at EVERY guarded seam.

W0.8 wrote the rule down (``batch_integrity.own_batch``: snapshot first, because a
verdict about a live view into a producer's rotating buffer is a verdict about
bytes nobody later reads) and then violated it in three of the nine paths it
guarded: the NAS fitness pass, the deterministic test-subsample collector, and the
nevresim SimulationRunner's test-input read all called ``verify_batch`` on the
tensor the loader had just yielded and only afterwards took a copy -- or no copy at
all.

The window is exactly the TOCTOU the rule exists to close, so the test reproduces
the window rather than the outcome: ``is_corrupt_batch`` is the last thing that
touches the batch before the verdict is returned, so poisoning the loader's live
buffer from inside it lands the poison precisely between the check and the use.

  * own-first  -> the verified tensor is a private copy; the poison lands in a
                  buffer nobody reads, and the measurement is the true one.
  * verify-first -> the verified tensor IS the buffer; the poison lands in the very
                  tensor about to be measured, and NOTHING raises -- the check
                  already passed. A fabricated number is published silently.

This is the load-bearing correction to W0.7, so it is tested at each seam that
claims it, not once on the primitive.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mimarsinan.data_handling import batch_integrity
from mimarsinan.model_training import basic_trainer_subsample
from mimarsinan.search.evaluators.loader_accuracy import accuracy_over_loader

LABELS = [0, 3, 1, 0]


class _ConstantClassModel(torch.nn.Module):
    """Finite inputs score perfectly; a non-finite batch collapses to one class.

    The same shape as the measured defect: a NaN batch does not raise, it makes the
    model emit one constant class, so the "accuracy" lands at that class's share.
    """

    def __init__(self, n_classes=4):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x):
        flat = x.reshape(x.shape[0], -1)
        if not bool(torch.isfinite(flat).all()):
            logits = torch.zeros(x.shape[0], self.n_classes)
            logits[:, 0] = 1.0
            return logits
        labels = flat[:, 0].round().long().clamp(0, self.n_classes - 1)
        return torch.nn.functional.one_hot(labels, self.n_classes).float()

    def to(self, *args, **kwargs):
        return self


class _RefillableLoader:
    """A loader that hands out THE SAME buffer every pass, as FFCV's pool does.

    ``poison()`` is what a producer thread refilling that buffer looks like from
    the consumer's side: an in-place overwrite of bytes the consumer is holding a
    view of.
    """

    def __init__(self, labels=LABELS):
        y = torch.tensor(labels, dtype=torch.long)
        x = torch.zeros(len(labels), 1, 2, 2)
        x[:, 0, 0, 0] = y.float()
        self.buffer_x = x
        self.buffer_y = y
        self.passes_taken = 0

    def __iter__(self):
        self.passes_taken += 1
        return iter([(self.buffer_x, self.buffer_y)])

    def __len__(self):
        return 1

    def poison(self):
        self.buffer_x.fill_(float("nan"))


@pytest.fixture
def refill_between_check_and_use(monkeypatch):
    """Poison the loader's buffer in the window between the verdict and the use."""

    def _arm(loader):
        real = batch_integrity.is_corrupt_batch

        def _probe(x):
            verdict = real(x)
            loader.poison()
            return verdict

        monkeypatch.setattr(batch_integrity, "is_corrupt_batch", _probe)

    return _arm


class TestNasFitnessPass:
    """``search/evaluators/loader_accuracy`` -- the fitness that RANKS candidates."""

    def test_the_ranked_number_is_measured_on_the_verified_bytes(
        self, refill_between_check_and_use
    ):
        loader = _RefillableLoader()
        refill_between_check_and_use(loader)
        assert accuracy_over_loader(_ConstantClassModel(), loader, "cpu") == 1.0


class TestSubsampleCollector:
    """``basic_trainer_subsample`` -- what ``pipeline_metric`` progresses on."""

    class _Trainer:
        def __init__(self, loader):
            self.test_loader = loader

    def test_collected_samples_are_the_verified_bytes(
        self, refill_between_check_and_use
    ):
        loader = _RefillableLoader()
        refill_between_check_and_use(loader)
        xs, _ = basic_trainer_subsample._collect_all(self._Trainer(loader), "loader")
        assert all(bool(torch.isfinite(s).all()) for s in xs)

    def test_selected_samples_are_the_verified_bytes(
        self, refill_between_check_and_use
    ):
        loader = _RefillableLoader()
        refill_between_check_and_use(loader)
        xs, _ = basic_trainer_subsample._collect_selected(
            self._Trainer(loader), {0, 2}, "loader"
        )
        assert all(bool(torch.isfinite(s).all()) for s in xs)


class TestSimulationRunnerTestInputs:
    """``chip_simulation/simulation_runner/core`` -- the chip's actual inputs."""

    def test_the_simulated_inputs_are_the_verified_bytes(
        self, refill_between_check_and_use
    ):
        from mimarsinan.chip_simulation.simulation_runner import core

        loader = _RefillableLoader()
        refill_between_check_and_use(loader)
        runner = core.SimulationRunner.__new__(core.SimulationRunner)
        runner._preprocessor = torch.nn.Identity()
        data = core.load_verified_test_data(runner, loader, "the simulation test loader")
        assert np.isfinite(np.stack([x for x, _ in data])).all()


class TestOwnVerifiedBatchPrimitive:
    def test_the_verified_tensor_is_not_the_loader_tensor(self):
        x = torch.zeros(2, 2)
        owned, _ = batch_integrity.own_verified_batch(x, None, source="loader")
        x.fill_(float("nan"))
        assert bool(torch.isfinite(owned).all())

    def test_a_corrupt_batch_still_raises(self):
        with pytest.raises(batch_integrity.CorruptedBatchError):
            batch_integrity.own_verified_batch(
                torch.full((2, 2), float("nan")), None, source="loader"
            )

    def test_the_move_happens_before_the_verdict(self):
        """The verdict must be about the tensor the FORWARD PASS reads, so a
        device-moved batch is verified after the move, not before it."""
        seen = []
        real = batch_integrity.is_corrupt_batch

        def _probe(t):
            seen.append(t.device.type)
            return real(t)

        try:
            batch_integrity.is_corrupt_batch = _probe
            batch_integrity.own_verified_batch(
                torch.zeros(2, 2), None, source="loader", device="cpu",
            )
        finally:
            batch_integrity.is_corrupt_batch = real
        assert seen == ["cpu"]

    def test_non_tensor_targets_pass_through(self):
        _, y = batch_integrity.own_verified_batch(torch.zeros(2), "label", source="l")
        assert y == "label"


class TestEscalatingRepairPrimitive:
    """``restart_and_advance`` -- the one repair mechanism the retries escalate."""

    def test_zero_skip_is_a_plain_rewind(self):
        it = batch_integrity.restart_and_advance(lambda: iter([0, 1, 2]), 0)
        assert next(it) == 0

    def test_skip_lands_on_the_requested_position(self):
        it = batch_integrity.restart_and_advance(lambda: iter([0, 1, 2]), 2)
        assert next(it) == 2

    def test_a_short_loader_rewinds_instead_of_running_out(self):
        """Advancing past the end must yield a usable iterator, not a dead one."""
        it = batch_integrity.restart_and_advance(lambda: iter([0, 1]), 5)
        assert next(it) == 0
