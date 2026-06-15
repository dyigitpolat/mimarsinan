"""Tests for ``tuning_recipe_recovery`` (default-off byte-identical).

The STEP-based recovery historically hardcodes ``torch.optim.Adam(weight_decay=5e-5)``
with a warmup-then-CONSTANT LR, ignoring the configured ``tuning_recipe``. Behind
the new ``tuning_recipe_recovery`` flag the step recovery honors the recipe
(optimizer family, weight-decay, momentum/betas) and schedules the LR (warmup +
cosine over the budget) with the discovered LR as the PEAK. Flag off reproduces
the hardcoded Adam + constant-LR path exactly.
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import TrainingRecipe
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from conftest import MockDataProviderFactory


class _WrapperLoss:
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss()(model(x), y)


def _make_trainer(recipe=None, *, tuning_recipe_recovery=False,
                  num_classes=4, input_shape=(1, 8, 8)):
    dp_factory = MockDataProviderFactory(input_shape=input_shape, num_classes=num_classes)
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    in_features = 1
    for d in input_shape:
        in_features *= d
    model = nn.Sequential(nn.Flatten(), nn.Linear(in_features, num_classes))
    trainer = BasicTrainer(
        model, "cpu", dlf, _WrapperLoss(),
        recipe=recipe,
        tuning_recipe_recovery=tuning_recipe_recovery,
    )
    return trainer


def _adamw_recipe(weight_decay=1e-4):
    return TrainingRecipe(
        optimizer="adamw",
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        scheduler="cosine",
        warmup_ratio=0.2,
    )


def _sgd_recipe(momentum=0.85):
    return TrainingRecipe(
        optimizer="sgd",
        weight_decay=1e-4,
        momentum=momentum,
        scheduler="cosine",
        warmup_ratio=0.0,
    )


def _decay_group_wds(optimizer):
    """Weight-decay values of the param groups that DO decay (non-zero only).

    ``build_param_groups`` splits the (no-)decay groups, so the decay group is
    the one whose ``weight_decay`` is non-zero; with a uniform LR there is one.
    """
    return [g["weight_decay"] for g in optimizer.param_groups if g["weight_decay"] > 0]


# ── flag ON: recipe-driven optimizer + scheduled LR ──────────────────────────


class TestRecipeRecoveryFlagOn:
    def test_step_optimizer_is_recipe_adamw_with_weight_decay(self):
        """Flag on + adamw recipe → step optimizer is AdamW with the recipe wd
        (NOT the hardcoded Adam / 5e-5)."""
        trainer = _make_trainer(_adamw_recipe(weight_decay=1e-4), tuning_recipe_recovery=True)

        opt, _sched, _scaler = trainer._get_optimizer_and_scheduler_steps(0.01, total_steps=50)
        assert type(opt) is torch.optim.AdamW, "must be AdamW, not the hardcoded Adam"
        assert _decay_group_wds(opt) == [pytest.approx(1e-4)]

        owned = trainer.build_step_optimizer(0.01)
        assert type(owned) is torch.optim.AdamW
        assert _decay_group_wds(owned) == [pytest.approx(1e-4)]

    def test_step_optimizer_is_recipe_sgd_with_momentum(self):
        """Flag on + sgd recipe → step optimizer is SGD with the recipe momentum."""
        trainer = _make_trainer(_sgd_recipe(momentum=0.85), tuning_recipe_recovery=True)

        opt, _sched, _scaler = trainer._get_optimizer_and_scheduler_steps(0.01, total_steps=50)
        assert isinstance(opt, torch.optim.SGD)
        assert all(g["momentum"] == pytest.approx(0.85) for g in opt.param_groups)

        owned = trainer.build_step_optimizer(0.01)
        assert isinstance(owned, torch.optim.SGD)
        assert all(g["momentum"] == pytest.approx(0.85) for g in owned.param_groups)

    def test_scheduled_lr_warms_up_and_decays_peaking_at_supplied_lr(self):
        """Flag on → the per-cycle LR is scheduled (warms up from < peak then
        decays), peaking at the supplied LR. NOT constant."""
        peak = 0.01
        total_steps = 60
        trainer = _make_trainer(_adamw_recipe(), tuning_recipe_recovery=True)

        opt, sched, _scaler = trainer._get_optimizer_and_scheduler_steps(
            peak, total_steps=total_steps
        )
        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        assert lrs[0] < peak, "warmup must start below the peak"
        assert max(lrs) == pytest.approx(peak, rel=1e-3), "schedule must peak at the supplied LR"
        assert lrs[-1] < max(lrs), "cosine tail must decay below the peak"
        assert len(set(round(v, 8) for v in lrs)) > 2, "LR trajectory must not be constant"

    def test_scheduler_for_owned_optimizer_is_also_scheduled(self):
        """The externally-owned-optimizer scheduler path (persistent optimizer)
        is also recipe-scheduled when the flag is on."""
        peak = 0.01
        total_steps = 60
        trainer = _make_trainer(_adamw_recipe(), tuning_recipe_recovery=True)
        opt = trainer.build_step_optimizer(peak)

        sched, _scaler = trainer._scheduler_and_scaler_for_optimizer(
            opt, peak, total_steps
        )
        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        assert lrs[0] < peak
        assert max(lrs) == pytest.approx(peak, rel=1e-3)
        assert lrs[-1] < max(lrs)
        assert len(set(round(v, 8) for v in lrs)) > 2

    def test_train_steps_until_target_drives_scheduled_lr(self):
        """Through the public recovery API the reported LR trajectory is scheduled
        (warms up then decays), not constant."""
        trainer = _make_trainer(_adamw_recipe(), tuning_recipe_recovery=True)
        reported_lrs = []
        trainer.report_function = lambda name, val: (
            reported_lrs.append(val) if name == "LR" else None
        )
        trainer.validate_n_batches = lambda n: 0.0  # unreachable target → run all steps
        trainer.test = lambda: 0.0

        trainer.train_steps_until_target(
            lr=0.01,
            max_steps=40,
            target_accuracy=2.0,
            warmup_steps=0,
            validation_n_batches=1,
            check_interval=10,
            patience=100,
            min_steps=0,
        )
        assert reported_lrs, "LR must be reported each step"
        assert len(set(round(v, 8) for v in reported_lrs)) > 2, (
            "recipe-recovery LR must be scheduled, not constant"
        )
        assert max(reported_lrs) == pytest.approx(0.01, rel=5e-2)


# ── flag OFF: hardcoded Adam + constant LR (byte-identical) ───────────────────


class TestRecipeRecoveryFlagOff:
    def test_step_optimizer_is_hardcoded_adam_wd_5e5_even_with_recipe(self):
        """Flag off → the step optimizer is the hardcoded Adam(weight_decay=5e-5),
        ignoring the recipe (byte-identical to today)."""
        trainer = _make_trainer(_adamw_recipe(weight_decay=1e-4), tuning_recipe_recovery=False)

        opt, _sched, _scaler = trainer._get_optimizer_and_scheduler_steps(0.01, total_steps=50)
        assert type(opt) is torch.optim.Adam
        assert all(g["weight_decay"] == pytest.approx(5e-5) for g in opt.param_groups)

        owned = trainer.build_step_optimizer(0.01)
        assert type(owned) is torch.optim.Adam
        assert all(g["weight_decay"] == pytest.approx(5e-5) for g in owned.param_groups)

    def test_constant_lr_path_is_constant_after_warmup(self):
        """Flag off + constant_lr → warmup-then-constant exactly as today: the LR
        is constant at the peak after the short linear warmup."""
        peak = 0.01
        total_steps = 60
        trainer = _make_trainer(_adamw_recipe(), tuning_recipe_recovery=False)

        opt, sched, _scaler = trainer._get_optimizer_and_scheduler_steps(
            peak, total_steps=total_steps, constant_lr=True
        )
        lrs = []
        for _ in range(total_steps):
            lrs.append(opt.param_groups[0]["lr"])
            sched.step()

        tail = lrs[-10:]
        assert all(v == pytest.approx(peak) for v in tail), (
            "after the linear warmup the LR must be constant at the peak"
        )

    def test_default_off_matches_no_recipe_byte_identical(self):
        """No flag + recipe is byte-identical to no recipe at all on the step path:
        same optimizer type/wd and same LR trajectory."""
        peak = 0.01
        total_steps = 30
        t_recipe = _make_trainer(_adamw_recipe(), tuning_recipe_recovery=False)
        t_plain = _make_trainer(None, tuning_recipe_recovery=False)

        for trainer in (t_recipe, t_plain):
            torch.manual_seed(0)

        opt_r, sched_r, _ = t_recipe._get_optimizer_and_scheduler_steps(
            peak, total_steps=total_steps, constant_lr=True
        )
        opt_p, sched_p, _ = t_plain._get_optimizer_and_scheduler_steps(
            peak, total_steps=total_steps, constant_lr=True
        )
        assert type(opt_r) is type(opt_p) is torch.optim.Adam
        lrs_r, lrs_p = [], []
        for _ in range(total_steps):
            lrs_r.append(opt_r.param_groups[0]["lr"])
            lrs_p.append(opt_p.param_groups[0]["lr"])
            sched_r.step()
            sched_p.step()
        assert lrs_r == pytest.approx(lrs_p)

    def test_flag_default_false_in_defaults(self):
        from mimarsinan.config_schema.defaults import (
            DEFAULT_DEPLOYMENT_PARAMETERS,
            CONFIG_KEYS_SET,
        )

        assert DEFAULT_DEPLOYMENT_PARAMETERS["tuning_recipe_recovery"] is False
        assert "tuning_recipe_recovery" in CONFIG_KEYS_SET

    def test_trainer_defaults_flag_off(self):
        """A trainer built without the kwarg keeps recovery on the legacy path."""
        trainer = _make_trainer(_adamw_recipe())
        assert trainer.tuning_recipe_recovery is False
        opt = trainer.build_step_optimizer(0.01)
        assert type(opt) is torch.optim.Adam
        assert all(g["weight_decay"] == pytest.approx(5e-5) for g in opt.param_groups)


# ── recipe-vs-plateau composition ────────────────────────────────────────────


class TestRecipePlateauComposition:
    def test_plateau_reduction_scales_recipe_peak(self):
        """With BOTH flags on, the recipe schedule is the base LR trajectory and a
        plateau reduction multiplies the PEAK on top: after a reduction the
        scheduled peak is ``factor * original_peak`` (not a flat constant)."""
        peak = 0.1
        trainer = _make_trainer(_adamw_recipe(), tuning_recipe_recovery=True)

        opt_ref = [None]
        orig_optimize = trainer._optimize

        def capture(x, y, optimizer, scaler):
            opt_ref[0] = optimizer
            return orig_optimize(x, y, optimizer, scaler)

        trainer._optimize = capture
        observed_peaks = []
        observed_live = []

        def tracking_validate(_n):
            if opt_ref[0] is not None:
                g = opt_ref[0].param_groups[0]
                # ``initial_lr`` holds the recipe-schedule PEAK; ``lr`` is the live
                # (cosine-decayed) value. The plateau lowers the PEAK.
                observed_peaks.append(g.get("initial_lr", g["lr"]))
                observed_live.append(g["lr"])
            return 0.5  # never improves → plateau-driven

        trainer.validate_n_batches = tracking_validate
        trainer.test = lambda: 0.5

        trainer.train_steps_until_target(
            lr=peak,
            max_steps=200,
            target_accuracy=2.0,
            validation_n_batches=1,
            check_interval=1,
            patience=2,
            min_steps=0,
            min_improvement=1e-3,
            plateau_lr_factor=0.5,
            plateau_lr_reductions=2,
        )

        # The recipe PEAK is reduced multiplicatively: 0.1 → 0.05 → 0.025.
        assert min(observed_peaks) == pytest.approx(0.025), (
            f"plateau must scale the recipe peak down to 0.1*0.5*0.5=0.025, "
            f"got {min(observed_peaks):.4g}"
        )
        # The live LR is the cosine trajectory (warmup + decay), NOT a flat line —
        # this is what distinguishes recipe-recovery composition from the legacy
        # constant-LR plateau path.
        assert len(set(round(v, 7) for v in observed_live)) > 2
