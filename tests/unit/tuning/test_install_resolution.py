"""[MBH-A6] install-resolution pre-flight gauges (5v): pure math + capture + emission.

A P1-family install at target resolution (S, T) is MBH-rampable only if every
hop of the longest unfragmented chain retains usable signal. The gauges are
static (computable from float activations BEFORE the install) and warn-only:
thresholds need matrix-wide calibration before they may refuse anything.
Fixtures mirror the study's known-crater (t0_21/t0_01 mixers) and known-clean
(t0_22 lenet5 / t0_05 simple_mlp) anatomy numbers.
"""

from __future__ import annotations

import torch

from conftest import make_tiny_supermodel
from mimarsinan.tuning.orchestration.install_resolution import (
    MIN_MEDIAN_EFFECTIVE_LEVELS,
    STARVED_MASS_WARN,
    ChannelStatsAccumulator,
    HopValueGauge,
    TemporalWindowGauge,
    ValueInstallGauge,
    collect_channel_stats,
    emit_temporal_gauge,
    emit_value_gauge,
    first_fire_delay,
    hop_value_gauge,
    median_effective_levels,
    starved_mass,
    temporal_window_gauge,
)


class TestMedianEffectiveLevels:
    def test_crater_layer_reads_under_two_levels(self):
        # t0_21 layer-1 anatomy: theta 8.19 (full-quantile inflated), q99 spread
        # far below it, Delta = theta/8 ~ 1.02 -> most channels under 2 levels.
        q99s = [3.19, 1.4, 1.1, 0.9, 0.0, 0.0]  # pruned channels read 0
        levels = median_effective_levels(q99s, theta=8.19, levels=8)
        assert levels < MIN_MEDIAN_EFFECTIVE_LEVELS

    def test_healthy_layer_reads_many_levels(self):
        # Healthy: theta = q99 of the layer -> a top channel spans ~S levels.
        q99s = [1.0, 0.8, 0.6, 0.5]
        levels = median_effective_levels(q99s, theta=1.0, levels=8)
        assert levels >= 4.0

    def test_dead_channels_are_excluded_not_counted_as_zero(self):
        # Pruning zeros channels; the median is over LIVE channels so a mostly
        # pruned but healthy-on-live-channels layer does not false-alarm.
        q99s = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        assert median_effective_levels(q99s, theta=1.0, levels=8) == 8.0

    def test_all_dead_hop_reads_zero(self):
        assert median_effective_levels([0.0, 0.0], theta=1.0, levels=8) == 0.0


class TestStarvedMass:
    def test_crater_mass_below_one_grid_step(self):
        # 93% of positive elements below one grid step (t0_21 layer 1).
        values = [0.1] * 93 + [3.0] * 7
        assert starved_mass(values, theta=8.19, levels=8) > 0.9

    def test_healthy_uniform_mass(self):
        values = [i / 100 for i in range(1, 101)]
        mass = starved_mass(values, theta=1.0, levels=8)
        assert abs(mass - 0.12) < 0.05

    def test_no_positive_mass_is_fully_starved(self):
        assert starved_mass([], theta=1.0, levels=8) == 1.0


class TestHopAndInstallGauge:
    def _crater_hop(self, depth=1):
        return hop_value_gauge(
            name=f"fc{depth}", depth=depth,
            per_channel_q99=[3.19, 1.4, 1.1, 0.9, 0.0],
            positive_values=[0.1] * 93 + [3.0] * 7,
            theta=8.19, levels=8,
        )

    def _clean_hop(self, depth=1):
        return hop_value_gauge(
            name=f"fc{depth}", depth=depth,
            per_channel_q99=[1.0, 0.8, 0.6, 0.5],
            positive_values=[i / 100 for i in range(1, 101)],
            theta=1.0, levels=8,
        )

    def test_crater_hop_is_starved(self):
        hop = self._crater_hop()
        assert isinstance(hop, HopValueGauge)
        assert hop.starved is True

    def test_clean_hop_is_not_starved(self):
        assert self._clean_hop().starved is False

    def test_crater_install_fails_and_names_starved_hops(self):
        gauge = ValueInstallGauge(
            hops=tuple(self._crater_hop(d) for d in range(9)), levels=8,
        )
        assert gauge.fails is True
        assert len(gauge.starved_hops) == 9

    def test_clean_install_passes(self):
        gauge = ValueInstallGauge(
            hops=tuple(self._clean_hop(d) for d in range(4)), levels=8,
        )
        assert gauge.fails is False
        assert gauge.starved_hops == ()


class TestTemporalWindowGauge:
    def test_mixer_t4_window_is_exhausted(self):
        # t0_01: ~1 cycle of first-fire delay per hop, L=9 > T=4.
        delays = {d: 1.0 for d in range(1, 10)}
        gauge = temporal_window_gauge(delays, window=4)
        assert isinstance(gauge, TemporalWindowGauge)
        assert gauge.total_delay == 9.0
        assert gauge.fails is True

    def test_shallow_chain_fits_the_window(self):
        # t0_05: L~3 on T=4 -> no crater.
        gauge = temporal_window_gauge({1: 1.0, 2: 1.0, 3: 0.8}, window=4)
        assert gauge.fails is False

    def test_deep_chain_with_wide_window_passes(self):
        # deep_mlp d8 at T=32 (X4 t0_04): sum ~ 9 << 32.
        gauge = temporal_window_gauge({d: 1.0 for d in range(1, 10)}, window=32)
        assert gauge.fails is False

    def test_first_fire_delay_is_theta_over_drive(self):
        assert first_fire_delay(theta=2.0, mean_drive=0.5) == 4.0

    def test_zero_drive_hop_reads_a_dead_window(self):
        # A hop with no drive can never fire: delay saturates far above any T.
        assert first_fire_delay(theta=1.0, mean_drive=0.0) >= 1e6


class TestChannelStatsAccumulator:
    def test_accumulates_per_channel_q99_and_positives(self):
        acc = ChannelStatsAccumulator()
        x = torch.tensor([[1.0, -1.0], [0.5, -0.5], [2.0, 0.25]])
        acc.output_transform(x)
        q99 = acc.per_channel_q99()
        assert len(q99) == 2
        assert q99[0] >= 1.0  # channel 0 positives: 1.0, 0.5, 2.0
        assert 0.0 < q99[1] <= 0.25  # channel 1 has one positive
        positives = acc.positive_values()
        assert all(v > 0 for v in positives)
        assert len(positives) == 4

    def test_conv_shaped_outputs_use_the_channel_axis(self):
        acc = ChannelStatsAccumulator()
        x = torch.zeros(2, 3, 4, 4)
        x[:, 1] = 1.0
        acc.output_transform(x)
        q99 = acc.per_channel_q99()
        assert len(q99) == 3
        assert q99[1] > 0 and q99[0] == 0.0 and q99[2] == 0.0

    def test_all_negative_channel_reads_zero(self):
        acc = ChannelStatsAccumulator()
        acc.output_transform(torch.full((4, 2), -1.0))
        assert acc.per_channel_q99() == [0.0, 0.0]
        assert acc.positive_values() == []


class TestCollectChannelStats:
    def test_collects_one_stat_per_perceptron_and_restores_activations(self):
        model = make_tiny_supermodel()
        before = [id(p.activation) for p in model.get_perceptrons()]
        batches = [torch.randn(4, 1, 8, 8) for _ in range(2)]
        stats = collect_channel_stats(model, batches, "cpu")
        after = [id(p.activation) for p in model.get_perceptrons()]
        assert len(stats) == len(list(model.get_perceptrons()))
        assert before == after, "capture must leave the activation stack untouched"
        for _, acc in stats:
            assert len(acc.per_channel_q99()) > 0

    def test_restores_activations_when_the_forward_raises(self):
        model = make_tiny_supermodel()
        before = [id(p.activation) for p in model.get_perceptrons()]

        class Boom:
            def __iter__(self):
                raise RuntimeError("boom")

        try:
            collect_channel_stats(model, Boom(), "cpu")
        except RuntimeError:
            pass
        after = [id(p.activation) for p in model.get_perceptrons()]
        assert before == after


class TestEmission:
    def _crater_gauge(self):
        hop = hop_value_gauge(
            name="fc1", depth=1,
            per_channel_q99=[3.19, 1.4, 0.0],
            positive_values=[0.1] * 93 + [3.0] * 7,
            theta=8.19, levels=8,
        )
        return ValueInstallGauge(hops=(hop,), levels=8)

    def test_value_gauge_warns_loud_but_never_raises(self, capsys):
        emit_value_gauge("ActivationAnalysisStep", self._crater_gauge())
        out = capsys.readouterr().out
        assert "[MBH-A6]" in out
        assert "verdict=FAIL" in out
        assert "warn-only" in out
        assert "fc1" in out  # the starved hop is named

    def test_clean_value_gauge_emits_a_pass_line(self, capsys):
        hop = hop_value_gauge(
            name="fc1", depth=1,
            per_channel_q99=[1.0, 0.8],
            positive_values=[i / 100 for i in range(1, 101)],
            theta=1.0, levels=8,
        )
        emit_value_gauge("ctx", ValueInstallGauge(hops=(hop,), levels=8))
        out = capsys.readouterr().out
        assert "verdict=PASS" in out

    def test_temporal_gauge_emits_window_arithmetic(self, capsys):
        gauge = temporal_window_gauge({d: 1.0 for d in range(1, 10)}, window=4)
        emit_temporal_gauge("LIFAdaptationTuner", gauge)
        out = capsys.readouterr().out
        assert "[MBH-A6]" in out
        assert "kind=temporal" in out
        assert "verdict=FAIL" in out
        assert "9.0" in out and "window=4" in out


class TestThresholdConstants:
    def test_a6_constants_are_the_study_values(self):
        assert MIN_MEDIAN_EFFECTIVE_LEVELS == 2.0
        assert STARVED_MASS_WARN == 0.5


class TestLifTunerCallSite:
    """A6(ii): the LIF install anchor logs the temporal window gauge at init."""

    def _lif_tuner(self, tmp_path):
        from conftest import MockPipeline, default_config
        from mimarsinan.tuning.orchestration.adaptation_manager import (
            AdaptationManager,
        )
        from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

        cfg = default_config()
        cfg["spiking_mode"] = "lif"
        cfg["firing_mode"] = "Default"
        cfg["thresholding_mode"] = "<"
        cfg["simulation_steps"] = 4
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        return LIFAdaptationTuner(
            pipeline, model=make_tiny_supermodel(), target_accuracy=0.5,
            lr=cfg["lr"], adaptation_manager=AdaptationManager(),
        )

    def test_init_emits_the_temporal_gauge(self, tmp_path, capsys):
        tuner = self._lif_tuner(tmp_path)
        try:
            out = capsys.readouterr().out
            assert "[MBH-A6] kind=temporal" in out
            assert "context=LIFAdaptationTuner" in out
            assert "window=4" in out
        finally:
            tuner.close()

    def test_gauge_capture_leaves_the_validation_cursor_untouched(self, tmp_path):
        tuner = self._lif_tuner(tmp_path)
        try:
            assert getattr(tuner.trainer, "_gpu_val_cursor", None) in (None, 0)
        finally:
            tuner.close()


class TestCascadedTunerCallSite:
    """A6 chain line: the cascaded install anchor names its hop-chain depth."""

    def test_configure_emits_the_chain_line(self, tmp_path, capsys):
        from conftest import MockPipeline, default_config
        from mimarsinan.tuning.orchestration.adaptation_manager import (
            AdaptationManager,
        )
        from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
            TTFSCycleAdaptationTuner,
        )

        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "cascaded"
        cfg["activation_quantization"] = True
        cfg["simulation_steps"] = 8
        cfg["ttfs_genuine_blend_ramp"] = True
        cfg["ttfs_genuine_blend_fast"] = True
        pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
        pipeline._target_metric = 0.0
        tuner = TTFSCycleAdaptationTuner(
            pipeline, model=make_tiny_supermodel(), target_accuracy=0.5,
            lr=cfg["lr"], adaptation_manager=AdaptationManager(),
        )
        try:
            out = capsys.readouterr().out
            assert "[MBH-A6] kind=chain" in out
            assert "max_intra_segment_depth=" in out
            assert "S=8" in out
        finally:
            tuner.close()
