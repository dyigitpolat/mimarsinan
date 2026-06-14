"""The GPU val cache caps to the fixed decision subsample (the W8 scale fix)."""

import torch

from mimarsinan.model_training import basic_trainer_eval


class _StubTrainer:
    device = "cpu"

    def __init__(self, n_batches, cap=None):
        self.validation_loader = [
            (torch.randn(2, 1, 4, 4), torch.randint(0, 4, (2,))) for _ in range(n_batches)
        ]
        if cap is not None:
            self._val_cache_max_batches = cap


def test_cap_limits_cache_length():
    t = _StubTrainer(10, cap=3)
    basic_trainer_eval._build_gpu_val_cache(t)
    assert len(t._gpu_val_cache) == 3


def test_no_cap_materializes_whole_loader():
    t = _StubTrainer(10)
    basic_trainer_eval._build_gpu_val_cache(t)
    assert len(t._gpu_val_cache) == 10


def test_capped_cursor_still_serves_requested_batches():
    t = _StubTrainer(10, cap=3)
    batches = list(basic_trainer_eval.iter_validation_batches(t, 5))
    assert len(batches) == 5  # cursor rotates within the capped subsample
