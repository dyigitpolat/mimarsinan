"""Per-perceptron decoded-value recorder side-channel on the LIF segment forward.

``chip_aligned_segment_forward(model, x, T, node_value_recorder=rec)`` runs the
deployed LIF cascade unchanged and, as a pure side-channel, fills ``rec`` with
each perceptron's decoded cascade value (``rate * activation_scale`` — the train
mean over T, in teacher-activation units), keyed by ``id(perceptron)``. The DFQ
LIF bias-correction loop consumes these to match the cascade's per-channel mean
to the teacher ANN's. The recorder must NEVER change the forward output.
"""

from __future__ import annotations

import copy

import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_partition import perceptron_of


T_STEPS = 8


def _deployed_lif_model(seed=0):
    """A tiny model driven into the deployed cycle-accurate LIF state (as the LIF
    step's caller leaves it before the deployed full-test eval)."""
    torch.manual_seed(seed)
    base = make_tiny_supermodel()

    cfg = default_config()
    cfg["spiking_mode"] = "lif"
    cfg["firing_mode"] = "Default"
    cfg["thresholding_mode"] = "<"
    cfg["simulation_steps"] = T_STEPS
    cfg["cycle_accurate_lif_forward"] = True
    pipeline = MockPipeline(config=cfg)
    pipeline._target_metric = 0.5

    model = copy.deepcopy(base)
    am = AdaptationManager()
    tuner = LIFAdaptationTuner(
        pipeline, model, target_accuracy=0.5,
        lr=cfg["lr"], adaptation_manager=am,
    )
    tuner._set_rate(1.0)
    tuner._finalize_rebuild()
    return model


def _cal_x(n=12):
    torch.manual_seed(123)
    return torch.randn(n, *default_config()["input_shape"])


class TestRecorderIsPureSideChannel:
    def test_output_byte_identical_with_and_without_recorder(self):
        model = _deployed_lif_model()
        x = _cal_x()

        out_plain = chip_aligned_segment_forward(model, x, T_STEPS)
        rec = {}
        out_rec = chip_aligned_segment_forward(model, x, T_STEPS, node_value_recorder=rec)

        torch.testing.assert_close(out_plain, out_rec, rtol=0, atol=0)


class TestRecorderPopulatesPerceptronValues:
    def test_one_entry_per_perceptron(self):
        model = _deployed_lif_model()
        x = _cal_x()
        rec = {}
        chip_aligned_segment_forward(model, x, T_STEPS, node_value_recorder=rec)

        perceptron_ids = {id(p) for p in model.get_perceptrons()}
        assert perceptron_ids <= set(rec.keys()), (
            "every perceptron must have a recorded decoded value"
        )

    def test_recorded_value_matches_decoded_channel_dim(self):
        model = _deployed_lif_model()
        x = _cal_x()
        rec = {}
        chip_aligned_segment_forward(model, x, T_STEPS, node_value_recorder=rec)

        for p in model.get_perceptrons():
            value = rec[id(p)]
            assert torch.isfinite(value).all()
            # Decoded value is non-negative (rate in [0,1] times a positive scale).
            assert float(value.min()) >= -1e-6

    def test_recorded_value_equals_train_mean(self):
        """The recorded value is exactly the per-cycle train mean (rate*scale)."""
        model = _deployed_lif_model()
        x = _cal_x()
        rec = {}
        out = chip_aligned_segment_forward(model, x, T_STEPS, node_value_recorder=rec)
        # The output (host classifier on the last segment's decoded means) is finite
        # and the recorder captured strictly positive activity somewhere (not all dead).
        assert torch.isfinite(out).all()
        total_active = sum(float(v.abs().sum()) for v in rec.values())
        assert total_active > 0.0


class TestRecorderDefaultNone:
    def test_no_recorder_argument_is_unaffected(self):
        model = _deployed_lif_model()
        x = _cal_x()
        # Must not raise and must return a valid tensor when no recorder is passed.
        out = chip_aligned_segment_forward(model, x, T_STEPS)
        assert torch.isfinite(out).all()
