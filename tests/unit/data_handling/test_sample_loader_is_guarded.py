"""W0.8b finding 1 -- the deterministic sample loader is a number-producing path.

``sample_loader.load_test_samples_by_index`` iterates a raw ``create_test_loader()``
and hands the samples it finds to shipped pipeline steps:
``loihi_simulation_step`` and ``sanafe_simulation_step`` feed them to a chip
simulator and compare the result against a torch reference. The number that comes
out is a PARITY verdict -- the deployment claim itself -- so a corrupted read here
does not print a wrong accuracy, it fails (or passes) a parity check on data that
was never the dataset.

W0.8's enumeration did not list it. Two defects follow from that:

  * no verification at all; and
  * ``xs[local : local + 1]`` is a VIEW into the loader's batch buffer, retained
    across the whole pass, so even a clean read can be overwritten by the producer
    before the caller ever looks at it -- the aliasing hazard W0.8 fixed one file
    over in ``test_on_subsample`` and left standing here.
"""

from __future__ import annotations

import pytest
import torch

from mimarsinan.data_handling import batch_integrity, sample_loader


class _Provider:
    def get_test_batch_size(self):
        return 2


class _Factory:
    """Stands in for ``DataLoaderFactory``; ``passes`` replays successive reads."""

    def __init__(self, passes):
        self._passes = list(passes)
        self.passes_taken = 0
        self.shutdowns = 0

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
    y = torch.zeros(len(values), dtype=torch.long)
    return x, y


def _nan(n=2):
    x, y = _clean([0.0] * n)
    return torch.full_like(x, float("nan")), y


@pytest.fixture
def factory(monkeypatch):
    made = {}

    def _install(passes):
        made["factory"] = _Factory(passes)
        monkeypatch.setattr(
            sample_loader, "DataLoaderFactory", lambda *a, **k: made["factory"]
        )
        monkeypatch.setattr(sample_loader, "shutdown_data_loader", lambda loader: None)
        return made["factory"]

    return _install


class TestCorruptedReadsNeverReachAParityCheck:
    def test_a_poisoned_pass_is_redone_rather_than_compared_against(self, factory):
        f = factory([[_nan(), _clean([2.0, 3.0])], [_clean([0.0, 1.0]), _clean([2.0, 3.0])]])
        samples = sample_loader.load_test_samples_by_index(object(), [0, 3])
        assert f.passes_taken == 2
        assert [float(s.reshape(-1)[0]) for s in samples] == [0.0, 3.0]

    def test_persistent_corruption_fails_loud_instead_of_returning_a_sample(self, factory):
        factory([[_nan()]])
        with pytest.raises(batch_integrity.CorruptedBatchError):
            sample_loader.load_test_samples_by_index(object(), [0])

    def test_the_single_sample_helper_is_guarded_too(self, factory):
        factory([[_nan()]])
        with pytest.raises(batch_integrity.CorruptedBatchError):
            sample_loader.load_test_sample_by_index(object(), 0)


class TestReturnedSamplesAreSnapshots:
    def test_samples_do_not_alias_the_loader_buffer(self, factory):
        """A retained view is a sample the producer is still free to overwrite."""
        batch = _clean([7.0, 8.0])
        factory([[batch]])
        samples = sample_loader.load_test_samples_by_index(object(), [0, 1])
        batch[0].fill_(float("nan"))
        assert all(bool(torch.isfinite(s).all()) for s in samples)

    def test_clean_reads_cost_exactly_one_pass(self, factory):
        f = factory([[_clean([1.0, 2.0])]])
        sample_loader.load_test_samples_by_index(object(), [0])
        assert f.passes_taken == 1

    def test_values_and_order_are_unchanged(self, factory):
        factory([[_clean([1.0, 2.0]), _clean([3.0, 4.0])]])
        samples = sample_loader.load_test_samples_by_index(object(), [3, 0])
        assert [float(s.reshape(-1)[0]) for s in samples] == [4.0, 1.0]

    def test_out_of_range_indices_still_raise_index_error(self, factory):
        factory([[_clean([1.0, 2.0])]])
        with pytest.raises(IndexError):
            sample_loader.load_test_samples_by_index(object(), [9])
