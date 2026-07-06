"""P1' encoder-scale conservation: cascaded QAT trains under the deployed encoder pin.

The FT rebuild (``calibrate_scale_aware_boundaries``) pins the subsumed encoding
layer's decode scale to ``input_data_scale`` — so a ClampTuner that stamps the
freely-measured quantile onto the encoder trains a QAT function the deployment
contract then truncates (T4 §2c: theta_enc 4.67 trained, 1.0 deployed, 5-8 pp).
Conservation: under a cascaded-TTFS plan the ClampTuner stamps the deployed pin
from the start, so every trained parameter survives verbatim (§6b contract-2).
"""

from __future__ import annotations

import pytest

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)

from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)
from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner


_ENC_SCALE = 4.67
_HIDDEN_SCALE = 2.2


def _make_clamp_tuner(tmp_path, *, spiking_mode, schedule=None):
    cfg = default_config()
    cfg["spiking_mode"] = spiking_mode
    if schedule is not None:
        cfg["ttfs_cycle_schedule"] = schedule
    cfg["activation_quantization"] = True
    cfg["optimization_driver"] = "fast"
    cfg["clamp_fast_steps_per_rate"] = 0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [_ENC_SCALE, _HIDDEN_SCALE]
    stats = make_activation_scale_stats(model, scales)
    tuner = ClampTuner(pipeline, model, 0.5, cfg["lr"], manager, scales, stats)
    return tuner, model


class TestCascadedEncoderPin:
    def test_encoder_scale_pinned_to_input_data_scale(self, tmp_path):
        _, model = _make_clamp_tuner(
            tmp_path, spiking_mode="ttfs_cycle_based", schedule="cascaded",
        )
        encoder, hidden = model.get_perceptrons()
        assert getattr(encoder, "is_encoding_layer", False)
        assert float(encoder.activation_scale) == pytest.approx(1.0), (
            "cascaded QAT must clamp the subsumed encoder at the deployed "
            "input_data_scale pin (P1' conservation), not the measured quantile"
        )

    def test_non_encoding_scales_stamped_verbatim(self, tmp_path):
        _, model = _make_clamp_tuner(
            tmp_path, spiking_mode="ttfs_cycle_based", schedule="cascaded",
        )
        _, hidden = model.get_perceptrons()
        assert float(hidden.activation_scale) == pytest.approx(_HIDDEN_SCALE)

    def test_diagnostics_record_the_pinned_scale(self, tmp_path):
        tuner, _ = _make_clamp_tuner(
            tmp_path, spiking_mode="ttfs_cycle_based", schedule="cascaded",
        )
        assert tuner.scale_diagnostics[0]["scale"] == pytest.approx(1.0)


class TestNonCascadedUnchanged:
    @pytest.mark.parametrize("mode,schedule", [
        ("lif", None),
        ("ttfs_quantized", None),
        ("ttfs_cycle_based", "synchronized"),
    ])
    def test_encoder_scale_keeps_measured_quantile(self, tmp_path, mode, schedule):
        _, model = _make_clamp_tuner(tmp_path, spiking_mode=mode, schedule=schedule)
        encoder, _ = model.get_perceptrons()
        assert float(encoder.activation_scale) == pytest.approx(_ENC_SCALE), (
            "the deploy-side pin exists only on the cascaded path; other modes "
            "carry theta_enc through mapping and must keep the measured scale"
        )
