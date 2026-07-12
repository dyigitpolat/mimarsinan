"""[S3/R6] AQ-tuner seam: gate, float-reference capture, and hook ordering.

The fold is armed by ``sync_exact_qat_active AND sync_first_moment_fold``,
captures its float pre-activation reference at tuner INIT (before the grid
snap / half-step folds change the forward), and applies at the conversion
endpoint inside ``_post_stabilization_hook`` — after the deferred half-step
fold, BEFORE ``run_endpoint_recovery``.
"""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel


def _sync_cfg(**overrides):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "synchronized"
    cfg["activation_quantization"] = True
    cfg["thresholding_mode"] = "<="
    cfg["target_tq"] = 4
    cfg["simulation_steps"] = 4
    cfg["sync_exact_qat"] = True
    cfg.update(overrides)
    return cfg


def _tuner(tmp_path, cfg):
    from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
        create_adaptation_manager_for_model,
    )
    from mimarsinan.tuning.tuners.activation_quantization_tuner import (
        ActivationQuantizationTuner,
    )

    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return ActivationQuantizationTuner(pipeline, model, 4, 0.5, cfg["lr"], manager)


class TestGateAndCapture:
    def test_knob_on_captures_the_float_reference_at_init(self, tmp_path):
        tuner = _tuner(tmp_path, _sync_cfg(sync_first_moment_fold=True))
        try:
            assert tuner._first_moment_armed is True
            assert tuner._fm_cal_x is not None
            assert tuner._fm_float_preact
            n_hooked = sum(
                1 for p in tuner.model.get_perceptrons()
                if hasattr(p, "activation")
            )
            assert len(tuner._fm_float_preact) == n_hooked
            # Capture is a reference, never a fold: no hop is marked yet.
            from mimarsinan.spiking.sync_first_moment import SYNC_FIRST_MOMENT_FLAG

            assert not any(
                getattr(p, SYNC_FIRST_MOMENT_FLAG, False)
                for p in tuner.model.get_perceptrons()
            )
        finally:
            tuner.close()

    def test_knob_absent_is_off_and_captures_nothing(self, tmp_path):
        cfg = _sync_cfg()
        assert "sync_first_moment_fold" not in cfg
        tuner = _tuner(tmp_path, cfg)
        try:
            assert tuner._first_moment_armed is False
            assert tuner._fm_cal_x is None
            assert tuner._fm_float_preact is None
        finally:
            tuner.close()

    def test_non_sync_mode_never_arms(self, tmp_path):
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_quantized"
        cfg["activation_quantization"] = True
        cfg["optimization_driver"] = "fast"
        cfg["sync_first_moment_fold"] = True  # accidental knob: mode gate wins
        tuner = _tuner(tmp_path, cfg)
        try:
            assert tuner._first_moment_armed is False
        finally:
            tuner.close()


class TestHookOrdering:
    def test_fold_runs_after_half_step_before_endpoint_recovery(
        self, tmp_path, monkeypatch,
    ):
        import mimarsinan.tuning.tuners.activation_quantization_tuner as aqt

        calls = []
        monkeypatch.setattr(
            aqt, "apply_sync_first_moment_fold",
            lambda *a, **kw: calls.append("fold") or {
                "folded": 1, "skipped": 0, "mean_abs_delta": 0.0,
                "per_hop_abs_delta": {},
            },
        )
        monkeypatch.setattr(
            aqt, "run_endpoint_recovery",
            lambda *a, **kw: calls.append("recovery"),
        )
        tuner = _tuner(
            tmp_path,
            _sync_cfg(sync_first_moment_fold=True, sync_entry_half_step=True),
        )
        try:
            tuner._fixed_ladder_policy = True
            tuner._post_stabilization_hook()
            assert calls == ["fold", "recovery"]
        finally:
            tuner.close()

    def test_knob_off_hook_never_folds(self, tmp_path, monkeypatch):
        import mimarsinan.tuning.tuners.activation_quantization_tuner as aqt

        calls = []
        monkeypatch.setattr(
            aqt, "apply_sync_first_moment_fold",
            lambda *a, **kw: calls.append("fold") or {},
        )
        monkeypatch.setattr(
            aqt, "run_endpoint_recovery",
            lambda *a, **kw: calls.append("recovery"),
        )
        tuner = _tuner(tmp_path, _sync_cfg())
        try:
            tuner._fixed_ladder_policy = True
            tuner._post_stabilization_hook()
            assert calls == ["recovery"]
        finally:
            tuner.close()


class TestEndToEndAtTheSeam:
    def test_real_fold_marks_hops_and_prints_witness(self, tmp_path, capsys):
        from mimarsinan.spiking.sync_first_moment import SYNC_FIRST_MOMENT_FLAG

        tuner = _tuner(
            tmp_path,
            _sync_cfg(sync_first_moment_fold=True, sync_entry_half_step=True),
        )
        try:
            perceptrons = list(tuner.model.get_perceptrons())
            biases_before = [
                p.layer.bias.detach().clone() if p.layer.bias is not None else None
                for p in perceptrons
            ]
            tuner._apply_first_moment_fold()
            out = capsys.readouterr().out
            assert "[MBH-S3] sync first-moment fold" in out
            assert any(
                getattr(p, SYNC_FIRST_MOMENT_FLAG, False) for p in perceptrons
            )
            half = 1.0 / (2 * 4)  # theta = 1, S = 4
            for p, b0 in zip(perceptrons, biases_before):
                if b0 is None:
                    continue
                change = p.layer.bias.detach() - b0
                if getattr(p, "is_encoding_layer", False):
                    # The encoder hop has NO upstream error: its raw gap is
                    # exactly its own half-step, so the exclusion makes the
                    # fold a no-op — the seam-level sign-trap witness.
                    torch.testing.assert_close(
                        change, torch.zeros_like(change), rtol=0, atol=1e-5,
                    )
                else:
                    # Downstream hops may fold genuine boundary-snap
                    # systematics (either sign, per channel), but never the
                    # trap's signature: a whole-population shift of ~= -half.
                    assert float(change.mean()) > -0.5 * half
        finally:
            tuner.close()
