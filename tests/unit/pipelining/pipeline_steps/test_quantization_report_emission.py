"""Quantization Verification: integer-grid diagnostics emission."""

import torch

from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    integer_grid_stats,
    quantization_grid_report,
)
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron


class TestIntegerGridStats:
    def test_counts_levels_zeros_and_clips(self):
        q_max = 3
        ints = torch.tensor([[0, 0, 1, -3], [3, 2, -1, 0]])
        stats = integer_grid_stats(ints, q_max)
        assert stats["n_weights"] == 8
        assert stats["zero_frac"] == 3 / 8
        assert stats["clip_frac"] == 2 / 8
        assert stats["effective_levels"] == 6  # {-3,-1,0,1,2,3}
        assert stats["int_min"] == -3 and stats["int_max"] == 3
        assert stats["q_max"] == 3

    def test_all_zero_layer(self):
        stats = integer_grid_stats(torch.zeros(4, 4, dtype=torch.long), 7)
        assert stats["zero_frac"] == 1.0
        assert stats["clip_frac"] == 0.0
        assert stats["effective_levels"] == 1


class TestQuantizationGridReport:
    def _quantized_perceptron(self, name: str, q_max: int) -> Perceptron:
        p = Perceptron(4, 3, name=name)
        ints = torch.randint(-q_max, q_max + 1, p.layer.weight.shape).float()
        scale = float(2 * q_max)
        with torch.no_grad():
            p.layer.weight.copy_(ints / scale)
            assert p.layer.bias is not None
            p.layer.bias.zero_()
        p.set_parameter_scale(torch.tensor(scale))
        return p

    def test_per_layer_rows_follow_the_verification_grid_convention(self):
        q_max = 7
        perceptrons = [
            self._quantized_perceptron("features_0", q_max),
            self._quantized_perceptron("features_1", q_max),
        ]
        report = quantization_grid_report(perceptrons, q_max)
        assert [row["name"] for row in report] == ["features_0", "features_1"]
        for row in report:
            assert row["n_weights"] == 12
            assert 1 <= row["effective_levels"] <= 2 * q_max + 1
            assert 0.0 <= row["clip_frac"] <= 1.0
            assert row["index"] in (0, 1)
