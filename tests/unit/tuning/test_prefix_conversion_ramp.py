"""Prefix-conversion RampStrategy (T4/P4): the axis walks converted-prefix k.

Recipe-gated via ConversionPolicy for cascaded vehicles with >1 spike segment:
each rung advances the topological frontier k, runs a short keep-best DFQ
re-affine measured through the k-hybrid, trains ~steps_per_rate plain-CE steps
(no KD teacher), and the [MBH-GATE] D-hat read IS the k-hybrid; the P1''
endpoint closes at k=n. Single-segment vehicles keep GenuineBlendRamp.
"""

from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from conftest import MockPipeline, default_config

from mimarsinan.models.spiking.training.prefix_genuine_forward import (
    PrefixGenuineForward,
)
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.tuning.axes.blend_axis import PrefixConversionAxis
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager
from mimarsinan.tuning.orchestration.blend_ramp import PlainClassificationLoss
from mimarsinan.tuning.orchestration.conversion_policy import ConversionPolicy
from mimarsinan.tuning.orchestration.ttfs_adaptation_plan import TtfsAdaptationPlan
from mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner import (
    GenuineBlendRamp,
    PrefixConversionRamp,
    TTFSCycleAdaptationTuner,
    _SegmentSpikeForward,
)


class _ThreeSegTorch(nn.Module):
    """Three spike segments split by bounded host ops (input (1, 8, 8))."""

    def __init__(self):
        super().__init__()
        self.flat = nn.Flatten()
        self.enc = nn.Sequential(nn.Linear(64, 16), nn.ReLU())
        self.mid = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16, 4), nn.ReLU())

    def forward(self, x):
        z = self.enc(self.flat(x))
        z = z * 0.7 + 0.05
        z = self.mid(z)
        z = z * 0.9 + 0.01
        return self.head(z)


def _make_pipeline(tmp_path, **extra):
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_cycle_based"
    cfg["ttfs_cycle_schedule"] = "cascaded"
    cfg["activation_quantization"] = True
    cfg["tuning_budget_scale"] = 1.0
    cfg["simulation_steps"] = 8
    cfg["ttfs_genuine_blend_ramp"] = True
    cfg["ttfs_genuine_blend_fast"] = True
    cfg["ttfs_prefix_ramp"] = True
    cfg["ttfs_blend_fast_steps_per_rate"] = 1
    cfg["ttfs_distmatch_bias_iters"] = 1
    cfg["ttfs_prefix_stage_dfq_iters"] = 1
    cfg["endpoint_recovery_steps"] = 0
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    for key, value in extra.items():
        pipeline.config[key] = value
    return pipeline


def _multi_seg_model():
    torch.manual_seed(0)
    return convert_torch_model(_ThreeSegTorch(), (1, 8, 8), 4, device="cpu")


def _make_tuner(tmp_path, *, model=None, **extra):
    pipeline = _make_pipeline(tmp_path, **extra)
    model = model if model is not None else _multi_seg_model()
    tuner = TTFSCycleAdaptationTuner(
        pipeline, model=model, target_accuracy=0.0,
        lr=pipeline.config["lr"], adaptation_manager=AdaptationManager(),
    )
    return tuner


class TestRecipeGating:
    def test_cascaded_recipe_carries_the_prefix_knob(self):
        recipe = ConversionPolicy.derive("ttfs_cycle_based", "cascaded")
        assert recipe.knobs.get("ttfs_prefix_ramp") is True

    @pytest.mark.parametrize("mode,schedule", [
        ("ttfs_cycle_based", "synchronized"),
        ("ttfs_quantized", None),
        ("lif", None),
    ])
    def test_other_modes_do_not(self, mode, schedule):
        recipe = ConversionPolicy.derive(mode, schedule)
        assert "ttfs_prefix_ramp" not in recipe.knobs

    def test_plan_resolves_prefix_ramp(self):
        plan = TtfsAdaptationPlan.resolve(
            {
                "ttfs_prefix_ramp": True,
                "ttfs_genuine_blend_ramp": True,
                "ttfs_genuine_blend_fast": True,
            },
            synchronized=False,
        )
        assert plan.prefix_ramp is True

    def test_synchronized_never_prefixes(self):
        plan = TtfsAdaptationPlan.resolve(
            {
                "ttfs_prefix_ramp": True,
                "ttfs_genuine_blend_ramp": True,
                "ttfs_genuine_blend_fast": True,
            },
            synchronized=True,
        )
        assert plan.prefix_ramp is False

    def test_default_off(self):
        plan = TtfsAdaptationPlan.resolve(
            {"ttfs_genuine_blend_ramp": True, "ttfs_genuine_blend_fast": True},
            synchronized=False,
        )
        assert plan.prefix_ramp is False


class TestStrategySelection:
    def test_multi_segment_selects_prefix_ramp(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        assert isinstance(tuner._ramp, PrefixConversionRamp)
        assert tuner._prefix_ramp is True
        assert tuner._n_spike_segments == 3

    def test_single_segment_falls_back_to_blend_ramp(self, tmp_path):
        from conftest import make_tiny_supermodel

        tuner = _make_tuner(tmp_path, model=make_tiny_supermodel())
        assert isinstance(tuner._ramp, GenuineBlendRamp)
        assert tuner._prefix_ramp is False

    def test_flag_off_keeps_blend_ramp(self, tmp_path):
        tuner = _make_tuner(tmp_path, ttfs_prefix_ramp=False)
        assert isinstance(tuner._ramp, GenuineBlendRamp)


class TestPrefixWiring:
    def test_ladder_rates_walk_the_frontier(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        assert tuner._fixed_ladder_rates == pytest.approx([1 / 3, 2 / 3, 1.0])

    def test_axis_and_forward(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        assert isinstance(tuner._axis, PrefixConversionAxis)
        forward = tuner.model.__dict__.get("forward")
        assert isinstance(forward, PrefixGenuineForward)
        assert forward.rate == 0.0

    def test_set_rate_drives_prefix_k(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        tuner._set_rate(2 / 3)
        forward = tuner.model.__dict__.get("forward")
        assert forward.prefix_k == 2

    def test_loss_is_plain_ce_no_teacher(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        assert isinstance(tuner.trainer.loss_function, PlainClassificationLoss)

    def test_fast_loss_is_ce_through_the_hybrid(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        x = torch.randn(2, 1, 8, 8)
        y = torch.tensor([0, 1])
        loss = tuner._fast_loss(x, y)
        assert loss.requires_grad

    def test_finalize_forward_is_pure_cascade(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        assert isinstance(tuner._finalize_forward_for(tuner.model), _SegmentSpikeForward)


class TestGateReadsTheKHybrid:
    def test_full_transform_forward_is_prefix_at_live_rate(self, tmp_path):
        import copy

        tuner = _make_tuner(tmp_path)
        tuner._set_rate(1 / 3)
        clone = copy.deepcopy(tuner.model)
        forward = tuner._mbh_full_transform_forward(clone)
        assert isinstance(forward, PrefixGenuineForward)
        assert forward.model is clone
        assert forward.prefix_k == 1

    def test_without_installed_forward_reads_full_deployment(self, tmp_path):
        import copy

        tuner = _make_tuner(tmp_path)
        tuner._remove_forward()
        clone = copy.deepcopy(tuner.model)
        forward = tuner._mbh_full_transform_forward(clone)
        assert isinstance(forward, PrefixGenuineForward)
        assert forward.prefix_k == tuner._n_spike_segments


class TestStageReaffine:
    def test_fast_ramp_runs_keepbest_dfq_and_emits_ledger(self, tmp_path, capsys):
        tuner = _make_tuner(tmp_path)
        tuner._ensure_fast_optimizer()
        tuner._fast_ramp(1 / 3)
        out = capsys.readouterr().out
        lines = [l for l in out.splitlines() if l.startswith("[MBH-PREFIX]")]
        assert len(lines) == 1
        match = re.match(
            r"^\[MBH-PREFIX\] tuner=TTFSCycleAdaptationTuner k=1/3 rate=0\.3333\d+ "
            r"dfq_probe_entry=[0-9.]+ dfq_probe_best=[0-9.]+ dfq_iters=\d+$",
            lines[0],
        )
        assert match, lines[0]

    def test_blend_ramp_fast_ramp_unchanged(self, tmp_path, capsys):
        tuner = _make_tuner(tmp_path, ttfs_prefix_ramp=False)
        tuner._ensure_fast_optimizer()
        tuner._fast_ramp(0.5)
        out = capsys.readouterr().out
        assert "[MBH-PREFIX]" not in out


class TestStageLR:
    """The spanning pipeline LR (3e-3) measured destructive through the genuine
    k-hybrid (first-wave t0_19: every stage's training discarded by keep-best);
    prefix ladders build the shared fast optimizer at the arm-B stage LR."""

    def test_prefix_fast_optimizer_uses_arm_b_stage_lr(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        assert tuner.pipeline.config["lr"] == pytest.approx(0.001)
        tuner.pipeline_lr = 0.003  # a hotter pipeline LR must be capped
        tuner._fast_optimizer = None
        tuner._ensure_fast_optimizer()
        assert tuner._fast_optimizer.defaults["lr"] == pytest.approx(0.001)

    def test_cooler_pipeline_lr_wins(self, tmp_path):
        tuner = _make_tuner(tmp_path)
        tuner.pipeline_lr = 0.0005
        tuner._fast_optimizer = None
        tuner._ensure_fast_optimizer()
        assert tuner._fast_optimizer.defaults["lr"] == pytest.approx(0.0005)

    def test_blend_ramp_keeps_the_pipeline_lr(self, tmp_path):
        tuner = _make_tuner(tmp_path, ttfs_prefix_ramp=False)
        tuner.pipeline_lr = 0.003
        tuner._fast_optimizer = None
        tuner._ensure_fast_optimizer()
        assert tuner._fast_optimizer.defaults["lr"] == pytest.approx(0.003)


class TestStageKeepBest:
    """Arm-B stage semantics: rung training keeps the best k-hybrid probe state
    (never ends below its post-DFQ entry) and clips gradients — a destructive
    stage degenerates to its calibrated entry instead of poisoning the gate."""

    def _train_with_probe_script(self, tmp_path, monkeypatch, script):
        import itertools

        import mimarsinan.tuning.tuners.ttfs_cycle_adaptation_tuner as tuner_mod

        tuner = _make_tuner(tmp_path)
        tuner._ensure_fast_optimizer()
        tuner._fast_steps_per_rate = 2
        values = itertools.chain(script, itertools.repeat(script[-1]))
        monkeypatch.setattr(
            tuner_mod, "live_model_acc_fp32", lambda _tuner: next(values),
        )
        entry_state = {
            k: v.detach().clone() for k, v in tuner.model.state_dict().items()
        }
        tuner._prefix_stage_train(1 / 3)
        return tuner, entry_state

    def test_degrading_stage_restores_entry_state(self, tmp_path, monkeypatch):
        tuner, entry_state = self._train_with_probe_script(
            tmp_path, monkeypatch, [0.9, 0.1],
        )
        final = tuner.model.state_dict()
        for key, saved in entry_state.items():
            assert torch.equal(final[key], saved), key

    def test_improving_stage_keeps_the_trained_state(self, tmp_path, monkeypatch):
        tuner, entry_state = self._train_with_probe_script(
            tmp_path, monkeypatch, [0.1, 0.9],
        )
        final = tuner.model.state_dict()
        assert any(
            not torch.equal(final[key], saved) for key, saved in entry_state.items()
        ), "an improving stage must keep its trained parameters"

    def test_keep_best_probes_on_the_interval_and_the_last_step(self, tmp_path):
        # The shared rung loop probes entry, every interval-th step, and the
        # final step (deduplicated when they coincide).
        tuner = _make_tuner(tmp_path)
        tuner._ensure_fast_optimizer()
        tuner._fast_steps_per_rate = 5
        probes = []

        def probe():
            probes.append(len(probes))
            return 0.5

        tuner._fast_train_rung(
            1 / 3, keep_best_probe=probe, keep_best_interval=2,
        )
        # entry + steps 2, 4 (interval) + step 5 (last) = 4 probes.
        assert len(probes) == 4


class TestGateComposition:
    def test_run_walks_the_frontier_and_finalizes_at_k_n(self, tmp_path, capsys):
        tuner = _make_tuner(tmp_path)
        tuner.run()
        out = capsys.readouterr().out
        gate_lines = [l for l in out.splitlines() if l.startswith("[MBH-GATE]")]
        attempted = [
            float(m.group(1))
            for l in gate_lines
            for m in [re.search(r"(?:accept|reject) rung=\d+ (?:attempt=\d+ )?rate=([0-9.]+)", l)]
            if m
        ]
        assert attempted, out
        assert attempted[0] == pytest.approx(1 / 3)
        assert float(tuner._committed_rate) == 1.0
        # Finalize deploys the pure cascade (the k=n endpoint).
        forward = tuner.model.__dict__.get("forward")
        assert isinstance(forward, _SegmentSpikeForward)
        prefix_lines = [l for l in out.splitlines() if l.startswith("[MBH-PREFIX]")]
        assert len(prefix_lines) >= 3


class _DeepSingleSegTorch(nn.Module):
    """One unfragmented 6-hop chain (no host ops) — the t0_16 shape class."""

    def __init__(self, depth=6):
        super().__init__()
        self.flat = nn.Flatten()
        dims = [64] + [16] * (depth - 1) + [4]
        self.stages = nn.Sequential(*[
            nn.Sequential(nn.Linear(a, b), nn.ReLU())
            for a, b in zip(dims[:-1], dims[1:])
        ])

    def forward(self, x):
        return self.stages(self.flat(x))


def _deep_single_seg_model(depth=6):
    torch.manual_seed(0)
    return convert_torch_model(_DeepSingleSegTorch(depth), (1, 8, 8), 4, device="cpu")


class TestHopFrontierArming:
    """[5v B2] the frontier below segments arms on single-segment deep chains."""

    def test_recipe_carries_the_hop_prefix_knob(self):
        recipe = ConversionPolicy.derive("ttfs_cycle_based", "cascaded")
        assert recipe.knobs.get("ttfs_hop_prefix_ramp") is True

    def test_synchronized_never_carries_it(self):
        recipe = ConversionPolicy.derive("ttfs_cycle_based", "synchronized")
        assert "ttfs_hop_prefix_ramp" not in recipe.knobs

    def test_deep_single_segment_walks_the_hop_frontier(self, tmp_path):
        tuner = _make_tuner(
            tmp_path, model=_deep_single_seg_model(6), ttfs_hop_prefix_ramp=True,
        )
        try:
            assert tuner._n_spike_segments == 1
            assert tuner._hop_prefix_levels == 6
            assert tuner._prefix_ramp is True
            assert tuner._fixed_ladder_rates == pytest.approx(
                [i / 6 for i in range(1, 7)]
            )
            strategy = tuner._make_ramp_strategy()
            assert isinstance(strategy, PrefixConversionRamp)
            forward = strategy.ramp_forward(tuner, tuner.model)
            assert forward.hop_frontier is True
            assert forward.frontier_units == 6
        finally:
            tuner.close()

    def test_shallow_single_segment_keeps_the_blend_ramp(self, tmp_path):
        tuner = _make_tuner(
            tmp_path, model=_deep_single_seg_model(3), ttfs_hop_prefix_ramp=True,
        )
        try:
            assert tuner._hop_prefix_levels is None
            assert tuner._prefix_ramp is False
            assert isinstance(tuner._make_ramp_strategy(), GenuineBlendRamp)
        finally:
            tuner.close()

    def test_knob_off_is_bit_identical_blend_fallback(self, tmp_path):
        tuner = _make_tuner(
            tmp_path, model=_deep_single_seg_model(6), ttfs_hop_prefix_ramp=False,
        )
        try:
            assert tuner._hop_prefix_levels is None
            assert tuner._prefix_ramp is False
            assert isinstance(tuner._make_ramp_strategy(), GenuineBlendRamp)
        finally:
            tuner.close()

    def test_multi_segment_keeps_the_segment_frontier(self, tmp_path):
        tuner = _make_tuner(tmp_path, ttfs_hop_prefix_ramp=True)
        try:
            assert tuner._n_spike_segments == 3
            assert tuner._hop_prefix_levels is None
            assert tuner._prefix_ramp is True
            assert tuner._fixed_ladder_rates == pytest.approx([1 / 3, 2 / 3, 1.0])
            strategy = tuner._make_ramp_strategy()
            forward = strategy.ramp_forward(tuner, tuner.model)
            assert forward.hop_frontier is False
        finally:
            tuner.close()

    def test_probe_forward_carries_the_hop_mode(self, tmp_path):
        import copy

        tuner = _make_tuner(
            tmp_path, model=_deep_single_seg_model(6), ttfs_hop_prefix_ramp=True,
        )
        try:
            strategy = tuner._make_ramp_strategy()
            strategy.ramp_forward(tuner, tuner.model)
            tuner._prefix_forward.rate = 0.5
            clone = copy.deepcopy(tuner.model)
            probe_forward = tuner._mbh_full_transform_forward(clone)
            assert isinstance(probe_forward, PrefixGenuineForward)
            assert probe_forward.hop_frontier is True
            assert probe_forward.rate == pytest.approx(0.5)
        finally:
            tuner.close()
