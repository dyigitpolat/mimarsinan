"""The pipeline D-hat high-water SSOT (MBH X3, P1'' target anchor).

One home for the running max deployed full-transform accuracy: a pipeline-level
cache entry written by the [MBH-GATE] probes (entry + every rung read) and read
fail-loud by the endpoint-recovery stage. The mark only ever rises — endpoint
targets anchor on the pipeline's best measured deployed accuracy, never on a
damaged local baseline (theory section 5g-v/5i).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from conftest import (
    MockPipeline,
    default_config,
    make_activation_scale_stats,
    make_tiny_supermodel,
)
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.tuning.orchestration import dhat_highwater, mbh_ledger
from mimarsinan.tuning.orchestration.adaptation_manager_factory import (
    create_adaptation_manager_for_model,
)


def _dict_pipeline():
    return SimpleNamespace(cache={})


def _cache_pipeline():
    return SimpleNamespace(cache=PipelineCache())


@pytest.mark.parametrize("make", [_dict_pipeline, _cache_pipeline],
                         ids=["dict-cache", "pipeline-cache"])
class TestObserveRequire:
    def test_absent_peek_is_none_and_require_fails_loud(self, make):
        pipeline = make()
        assert dhat_highwater.peek(pipeline) is None
        with pytest.raises(RuntimeError, match="high-water mark is absent"):
            dhat_highwater.require(pipeline)

    def test_observe_writes_and_require_reads(self, make):
        pipeline = make()
        assert dhat_highwater.observe(pipeline, 0.42) == pytest.approx(0.42)
        assert dhat_highwater.require(pipeline) == pytest.approx(0.42)

    def test_mark_only_ever_rises(self, make):
        pipeline = make()
        dhat_highwater.observe(pipeline, 0.5)
        assert dhat_highwater.observe(pipeline, 0.3) == pytest.approx(0.5)
        assert dhat_highwater.require(pipeline) == pytest.approx(0.5)
        assert dhat_highwater.observe(pipeline, 0.7) == pytest.approx(0.7)
        assert dhat_highwater.require(pipeline) == pytest.approx(0.7)


class TestGateWritesTheMark:
    def _clamp_tuner(self, tmp_path):
        from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_quantized"
        cfg["activation_quantization"] = True
        cfg["optimization_driver"] = "fast"
        cfg["clamp_fast_rates"] = [0.5, 1.0]
        cfg["clamp_fast_steps_per_rate"] = 1
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        model = make_tiny_supermodel()
        manager = create_adaptation_manager_for_model(cfg, model)
        scales = [1.0 for _ in model.get_perceptrons()]
        stats = make_activation_scale_stats(model, scales)
        return ClampTuner(pipeline, model, 0.5, cfg["lr"], manager, scales, stats)

    def test_gate_probes_ratchet_the_pipeline_mark(self, tmp_path, monkeypatch):
        seq = iter([0.55, 0.62])
        monkeypatch.setattr(
            mbh_ledger, "rung_measurements",
            lambda tuner: {
                "blended_fp32": 0.5, "full_acc": next(seq),
                "rho": 1.0, "grad_norm_t": 0.0,
            },
        )
        monkeypatch.setattr(
            mbh_ledger, "full_transform_measurement", lambda tuner: 0.4,
        )
        torch.manual_seed(0)
        tuner = self._clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        # entry 0.4, rungs 0.55 / 0.62 -> the mark carries the max.
        assert dhat_highwater.require(tuner.pipeline) == pytest.approx(0.62)

    def test_rejected_reads_still_observe(self, tmp_path, monkeypatch):
        # A rejected rung's D-hat is a genuine deployed read; the mark keeps the
        # max even though the state was rolled back (rejects read below best by
        # construction, so this can only matter within the tolerance band).
        monkeypatch.setattr(
            mbh_ledger, "rung_measurements",
            lambda tuner: {
                "blended_fp32": 0.5, "full_acc": 0.895,
                "rho": 1.0, "grad_norm_t": 0.0,
            },
        )
        monkeypatch.setattr(
            mbh_ledger, "full_transform_measurement", lambda tuner: 0.9,
        )
        torch.manual_seed(0)
        tuner = self._clamp_tuner(tmp_path)
        try:
            tuner.run()
        finally:
            tuner.close()
        assert dhat_highwater.require(tuner.pipeline) == pytest.approx(0.9)
