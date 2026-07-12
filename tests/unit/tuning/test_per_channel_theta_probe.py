"""Clamp saturation counting must handle per-channel (vector) ceilings."""

import pytest
import torch

from mimarsinan.tuning.tuners.clamp_tuner import saturation_hit_count


class TestSaturationHitCount:
    def test_scalar_ceiling_counts_as_before(self):
        latest = torch.tensor([0.5, 0.999, 1.0, 0.2])
        assert saturation_hit_count(latest, torch.tensor(1.0)) == 2

    def test_vector_ceiling_broadcasts_channels_last(self):
        # 2 samples x 3 channels, flattened the way the decorator captures.
        latest = torch.tensor([1.0, 0.1, 3.0,
                               0.9, 2.0, 0.1])
        ceiling = torch.tensor([1.0, 2.0, 3.0])
        # hits: (1.0>=~1.0), (3.0>=~3.0), (2.0>=~2.0) -> 3
        assert saturation_hit_count(latest, ceiling) == 3

    def test_vector_ceiling_on_batched_tensor(self):
        latest = torch.rand(8, 4) + 10.0  # everything saturates
        ceiling = torch.ones(4)
        assert saturation_hit_count(latest, ceiling) == 32

    def test_non_divisible_flat_size_fails_loud(self):
        latest = torch.zeros(7)
        ceiling = torch.ones(3)
        with pytest.raises(RuntimeError):
            saturation_hit_count(latest, ceiling)

    def test_full_shape_channels_last_broadcasts(self):
        latest = torch.rand(16, 4) + 10.0
        assert saturation_hit_count(latest, torch.ones(4)) == 64

    def test_sampled_flat_capture_with_vector_ceiling_fails_loud(self):
        # The decorator's linspace subsample destroys channel structure.
        latest = torch.zeros(8192)
        with pytest.raises(RuntimeError, match="channels-last capture"):
            saturation_hit_count(latest, torch.ones(120))
