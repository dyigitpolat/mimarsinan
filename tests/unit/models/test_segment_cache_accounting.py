"""[F4] The retention budget must count what the entry actually holds.

The packed-cycle executor memoises its PackedStage into the SAME cache entry
the budget measures (``seg["packed"]``), and each bucket carries a DENSE
stack of the per-core weight views plus index tensors. None of it was
counted, so the 256 MiB budget under-reported by the stacked weights of
every packed stage — a budget that under-counts makes every eviction
decision unsound.
"""

import torch

from mimarsinan.models.spiking.hybrid.executors.packed_cycle import (
    PackedStage,
    _Bucket,
)
from mimarsinan.models.spiking.hybrid.segment_cache import segment_entry_nbytes


def _bucket(n_cores=4, n_ax=8, n_out=8):
    return _Bucket(
        latency=0, n_axons=n_ax, n_neurons=n_out,
        core_indices=list(range(n_cores)), neuron_start=0,
        neuron_end=n_cores * n_out,
        weights=torch.zeros(n_cores, n_ax, n_out, dtype=torch.float64),
        bias=torch.zeros(n_cores, n_out, dtype=torch.float64),
        on_dst=None, inp_dst=torch.arange(4), inp_src=torch.arange(4),
        buf_dst=None, buf_src=None,
    )


class TestPackedStageIsAccounted:
    def test_packed_weights_are_counted(self):
        bucket = _bucket()
        entry = {"core_params": [], "packed": PackedStage(
            total_neurons=32, buckets=[bucket],
            theta_flat=torch.zeros(32, dtype=torch.float64),
            neuron_offset={},
        )}
        counted = segment_entry_nbytes(entry)
        assert counted >= bucket.weights.numel() * 8, (
            f"stacked packed weights ({bucket.weights.numel() * 8} B) must be "
            f"counted; budget saw {counted} B"
        )

    def test_entry_without_a_packed_stage_is_unchanged(self):
        params = [torch.zeros(4, 4, dtype=torch.float64)]
        assert segment_entry_nbytes({"core_params": params}) == 4 * 4 * 8

    def test_shared_storage_is_not_double_counted(self):
        # The dedupe-by-storage rule must survive the new tensors.
        shared = torch.zeros(8, 8, dtype=torch.float64)
        entry = {"core_params": [shared, shared.view(-1)]}
        assert segment_entry_nbytes(entry) == 8 * 8 * 8
