"""E4 keystone activation — the real CascadeCharacterizer + the escalation flow.

The companion to ``test_conversion_policy.py`` (which pins the inert scaffolding):
here the keystone is SWITCHED ON with the real forward-only probes (cold-cascade
liveness / ramp monotonicity / staircase-vs-LIF ceiling / firing-gain), and the
propose → confirm → escalate flow is driven end to end on real converted cascade
flows:

1. an IN-distribution cascade ⇒ the characterizer matches ⇒ the proposed recipe
   stands (escalated=False);
2. an OFF-distribution model (a pruned-LIF-like / dead-at-depth cascade) ⇒ does
   NOT match ⇒ escalates to the controller fallback (escalated=True,
   driver=controller);
3. DEFAULT-OFF (``conversion_policy`` unset) ⇒ the characterizer is never
   constructed / consulted ⇒ byte-identical.

All probes are forward-only float64 toy, run on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cascade_fixtures import _SingleSegmentMLP, _calibrate_scales, install_ttfs_nodes

from mimarsinan.chip_simulation.spiking_mode_policy import policy_for_spiking_mode
from mimarsinan.tuning.orchestration.conversion_policy import (
    OPTIMIZATION_DRIVER_CONTROLLER,
    CharacterizationResult,
    ConversionPolicy,
    propose_recipe,
)
from mimarsinan.tuning.orchestration.characterization import (
    CascadeCharacterizer,
    CalibrationSource,
)


def _policy():
    return policy_for_spiking_mode("ttfs_cycle_based", "cascaded")


def _build_cascade(*, depth, width, in_dim, out_dim, seed, dead=False):
    """A converted TTFS cascade flow + its calibration inputs.

    ``dead=True`` prunes ~92% of every layer's weights to zero and kills the
    biases — the pruned-LIF-like / scrambled OFF-distribution fixture whose
    genuine single-spike cascade dies at depth (the deepest layers decode no
    value), so it must NOT match the cascaded fast recipe's assumptions.
    """
    from mimarsinan.torch_mapping.converter import convert_torch_model

    torch.manual_seed(seed)
    base = _SingleSegmentMLP(depth, width, in_dim, out_dim)
    for m in base.modules():
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.4, 0.4)
            nn.init.uniform_(m.bias, -0.05, 0.05)
    if dead:
        for m in base.modules():
            if isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight[torch.rand_like(m.weight) < 0.92] = 0.0
                    m.bias.mul_(0.0)

    flow = convert_torch_model(base, (in_dim,), out_dim, device="cpu")
    calib_x = torch.rand(128, in_dim, dtype=torch.float64)
    _calibrate_scales(flow, calib_x)
    install_ttfs_nodes(flow, 8)
    return flow, calib_x


class _FakeTrainer:
    """A calibration-batch source mimicking the trainer's iter_validation_batches."""

    def __init__(self, x, y=None):
        self._x = x
        self._y = torch.zeros(x.shape[0], dtype=torch.long) if y is None else y

    def iter_validation_batches(self, n_batches):
        for _ in range(int(n_batches)):
            yield self._x, self._y


# ── the calibration-batch seam (context plumbing) ────────────────────────────


class TestCalibrationSource:
    def test_context_supplies_batches(self):
        x = torch.rand(16, 8, dtype=torch.float64)
        src = CalibrationSource.from_context(_FakeTrainer(x), n_batches=2)
        batch = src.inputs()
        assert batch.shape[1] == 8
        # two yielded batches of 16 → concatenated to 32 calibration rows.
        assert batch.shape[0] == 32

    def test_explicit_inputs_pass_through(self):
        x = torch.rand(10, 8, dtype=torch.float64)
        src = CalibrationSource.from_inputs(x)
        assert torch.equal(src.inputs(), x)

    def test_none_context_is_falsy_until_inputs_supplied(self):
        src = CalibrationSource.from_context(None)
        assert src.inputs() is None


# ── the real CascadeCharacterizer probes (forward-only, CPU) ──────────────────


class TestCascadeCharacterizerProbes:
    def _char(self, flow, calib_x):
        return CascadeCharacterizer(
            context=_FakeTrainer(calib_x), n_batches=1, S=8,
        ), flow

    def test_in_distribution_cascade_is_live(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        char, _ = self._char(flow, x)
        assert char.cold_cascade_live(model=flow) is True

    def test_off_distribution_cascade_is_dead_at_depth(self):
        flow, x = _build_cascade(
            depth=6, width=16, in_dim=16, out_dim=6, seed=0, dead=True,
        )
        char, _ = self._char(flow, x)
        assert char.cold_cascade_live(model=flow) is False

    def test_in_distribution_ramp_is_monotone(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        char, _ = self._char(flow, x)
        assert char.ramp_monotone(model=flow) is True

    def test_off_distribution_firing_gain_is_collapsed(self):
        flow, x = _build_cascade(
            depth=6, width=16, in_dim=16, out_dim=6, seed=0, dead=True,
        )
        char, _ = self._char(flow, x)
        # the dead cascade's deepest firing gain collapses toward zero.
        assert char.firing_gain(model=flow) < 0.1

    def test_in_distribution_firing_gain_is_alive(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        char, _ = self._char(flow, x)
        assert char.firing_gain(model=flow) > 0.1

    def test_staircase_ceiling_is_positive(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        char, _ = self._char(flow, x)
        assert char.staircase_lif_ceiling(model=flow) > 0.0


class TestCharacterizeVerdict:
    def _char(self, flow, calib_x):
        return CascadeCharacterizer(context=_FakeTrainer(calib_x), n_batches=1, S=8)

    def test_in_distribution_matches(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        char = self._char(flow, x)
        recipe = propose_recipe(_policy())
        result = char.characterize(model=flow, recipe=recipe)
        assert isinstance(result, CharacterizationResult)
        assert result.matches is True
        assert result.probes  # the per-probe readings are archived

    def test_off_distribution_does_not_match_and_names_a_reason(self):
        flow, x = _build_cascade(
            depth=6, width=16, in_dim=16, out_dim=6, seed=0, dead=True,
        )
        char = self._char(flow, x)
        recipe = propose_recipe(_policy())
        result = char.characterize(model=flow, recipe=recipe)
        assert result.matches is False
        assert result.reason  # the mismatch is explained


# ── propose → confirm → escalate, driven through ConversionPolicy.resolve ──────


class TestEscalationFlow:
    def _resolve(self, flow, calib_x, *, enabled):
        cfg = {"conversion_policy": True} if enabled else {}
        char = CascadeCharacterizer(
            context=_FakeTrainer(calib_x), n_batches=1, S=8,
        )
        return ConversionPolicy.resolve(
            cfg,
            mode_policy=_policy(),
            model=flow,
            characterizer=char,
        )

    def test_in_distribution_recipe_stands(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        decision = self._resolve(flow, x, enabled=True)
        assert decision.enabled is True
        assert decision.characterized is True
        assert decision.escalated is False
        assert decision.driver == propose_recipe(_policy()).driver

    def test_off_distribution_escalates_to_controller(self):
        flow, x = _build_cascade(
            depth=6, width=16, in_dim=16, out_dim=6, seed=0, dead=True,
        )
        decision = self._resolve(flow, x, enabled=True)
        assert decision.enabled is True
        assert decision.characterized is True
        assert decision.escalated is True
        assert decision.driver == OPTIMIZATION_DRIVER_CONTROLLER
        assert decision.escalation_reason


class TestDefaultOffNeverConstructsCharacterizer:
    """(3) Default-off ⇒ the characterizer is never constructed / consulted."""

    def test_disabled_does_not_consult_a_supplied_characterizer(self):
        flow, x = _build_cascade(depth=6, width=16, in_dim=16, out_dim=6, seed=0)
        calls = {"n": 0}

        class _SpyCharacterizer(CascadeCharacterizer):
            def characterize(self, *, model, recipe, context=None):
                calls["n"] += 1
                return CharacterizationResult(matches=False, reason="should not run")

        spy = _SpyCharacterizer(context=_FakeTrainer(x), n_batches=1, S=8)
        decision = ConversionPolicy.resolve(
            {},  # conversion_policy unset ⇒ inert
            mode_policy=_policy(),
            model=flow,
            characterizer=spy,
        )
        assert calls["n"] == 0
        assert decision.enabled is False
        assert decision.escalated is False
        assert decision.driver == OPTIMIZATION_DRIVER_CONTROLLER
