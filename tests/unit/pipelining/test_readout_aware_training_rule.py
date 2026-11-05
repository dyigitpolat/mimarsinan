"""[U-RAW A-2] The readout-aware arm's training rule: under the deployed law, Weight Quantization's endpoint recovery
trains at the learning rate the toolchain's LR explorer finds non-destructive on the law-in-loop read, over its own
declared budget.

The defect (U-RAW's diagnosis at N = 1, C5 at adc_bits 4 on R5): the arm's recovery trained through the law at the
pipeline's learning rate ``lr`` (the member's float-training rate). One Adam step moves every weight by about the rate
whatever its gradient, and on a 4-bit weight grid that flips the codes of weights whose gradient is noise; the
law-in-loop read and the arm's own training loss got worse at every check, keep-best restored the entry weights and the
arm deployed the untrained count. The rule, one for every member and keyed by the declaration
(``readout_aware_training``; absent or false, every law, state and metric is R5's byte for byte):

* the recovery's learning rate is the toolchain's LR explorer's (``TunerBase._find_lr``, the rule the cycle recoveries
  already train at) run on the law-in-loop read, capped at the pipeline's ``lr``;
* an explorer that finds every probed rate destructive trains nothing: the entry state is kept;
* the recovery takes the funded convergence geometry over its declared budget (``endpoint_convergence_geometry``: the
  budget a ceiling, keep-best), never the unfunded stage's cheap patience stop.

The fixture is U-ADC's residual convolution net with executed lanes (L, E, P) at adc_bits 4, declared as the route (b)
rows declare their recovery (``endpoint_floor_steps`` 0, ``tuning_budget_scale`` 0: the unfunded, non-armed geometry),
at a learning rate where, under R5's rule, every law-in-loop check falls below the entry read."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

import pytest
import torch

import mimarsinan.pipelining.pipeline_steps.quantization.readout_aware_training as rat
import mimarsinan.tuning.orchestration.frontier.endpoint_recovery as er
import mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner as napq
from mimarsinan.model_training import basic_trainer_steps
from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer
from mimarsinan.model_training.training_recipe import build_optimizer, build_recipe
from mimarsinan.tuning.orchestration.recovery_engine import RecoveryEngine
from mimarsinan.tuning.orchestration.tuner_base import _RECOVERY_PATIENCE, TunerBase
from mimarsinan.tuning.orchestration.tuning_policy import endpoint_convergence_geometry
from unit.pipelining import readout_aware_programs as F

#: Each test takes the fixture through Weight Quantization's endpoint (about 3 to 45 s alone on one CPU): above the
#: suite's 60 s default under a loaded host.
pytestmark = pytest.mark.timeout(900)

KEY = rat.READOUT_AWARE_TRAINING
#: The route (b) rows' own recovery declarations (W34's points carry them): the unfunded, non-armed endpoint geometry.
ROWS = {"endpoint_floor_steps": 0, "tuning_budget_scale": 0.0}
#: The fixture: U-ADC's residual net with executed lanes, its recovery budget and the declared learning rate at which
#: R5's rule fails it.
NET, LANES, ADC_BITS, STEPS, LR = "resnet", ("L", "E", "P"), 4, 100, 0.01


def _prepared(tmp_path, *, key=None, **extra):
    declared = {} if key is None else {KEY: key}
    return F.prepared(tmp_path, adc_bits=ADC_BITS, net=NET, lanes=LANES, wq_endpoint_recovery_steps=STEPS, lr=LR,
                      **ROWS, **declared, **extra)


def _spies(monkeypatch):
    """Records every training leg the endpoint starts (its learning rate and geometry), every endpoint report with its
    trajectory, and every LR-explorer call (whether the law-in-loop context was open, and what it returned)."""
    seen = {"legs": [], "reports": [], "explorer": [], "recovery_kwargs": []}
    train = RecoveryEngine.train_to_target

    def leg(trainer, lr, target, **kwargs):
        seen["legs"].append({"lr": float(lr), "max_steps": kwargs.get("max_steps"), "patience": kwargs.get("patience"),
                             "min_steps": kwargs.get("min_steps"), "check_interval": kwargs.get("check_interval")})
        return train(trainer, lr, target, **kwargs)

    monkeypatch.setattr(RecoveryEngine, "train_to_target", staticmethod(leg))
    emit = er._emit

    def report(tuner, rep, trajectory):
        seen["reports"].append((rep, [tuple(t) for t in trajectory]))
        return emit(tuner, rep, trajectory)

    monkeypatch.setattr(er, "_emit", report)
    find = TunerBase._find_lr

    def explorer(self):
        loop = rat.current_loop(self)
        found = find(self)
        seen["explorer"].append({"loop_open": loop is not None, "found": found, "pipeline_lr": float(self.pipeline_lr)})
        return found

    monkeypatch.setattr(TunerBase, "_find_lr", explorer)
    recover = napq.run_endpoint_recovery

    def recovery(tuner, **kwargs):
        seen["recovery_kwargs"].append(sorted(kwargs))
        return recover(tuner, **kwargs)

    monkeypatch.setattr(napq, "run_endpoint_recovery", recovery)
    return seen


def _deploy(fx):
    ir = F.soft_core_mapping(fx)
    return ir, F.deployed(ir, fx.x)


class TestTheFixtureFailsUnderThePipelineRate:
    def test_at_the_pipeline_rate_every_law_in_loop_check_falls_below_the_entry_read(self, tmp_path, monkeypatch):
        """R5's rule on the fixture, read directly: inside the arm's law-in-loop context, the recovery's training leg at
        the pipeline's learning rate (the trainer's step schedule over the budget at the budget's check cadence, as
        R5's endpoint runs it, here without its patience stop) lowers the endpoint's own deployed-composition read
        (every validation input, through the law) at every check below its entry read (a guard of the fixture, on
        every tree)."""
        fx = _prepared(tmp_path, key=True)
        seen = {}

        def recovery(tuner, *, base_steps, target_floor=None, **_rule):
            checks = []
            entry = er._fp32_deployed_read(tuner)

            def read(step, acc, best, entry_acc):
                checks.append(er._fp32_deployed_read(tuner))
                return False

            RecoveryEngine.train_to_target(
                tuner.trainer, float(tuner.pipeline_lr), 1.0, max_steps=int(base_steps),
                hooks=tuner._recovery_training_hooks(1.0),
                validation_n_batches=tuner._budget.progress_eval_batches,
                check_interval=int(tuner._budget.check_interval), patience=int(base_steps), min_steps=0,
                min_improvement=tuner._budget.accuracy_se(), cosine_decay=True, return_steps=True,
                final_validation=False, on_check=read)
            seen.update(entry=entry, checks=checks, loop=rat.current_loop(tuner))

        monkeypatch.setattr(napq, "run_endpoint_recovery", recovery)
        F.weight_quantization(fx)
        assert isinstance(seen["loop"], rat.ReadoutLawInLoop)
        assert len(seen["checks"]) >= 4
        assert all(acc < seen["entry"] for acc in seen["checks"]), (seen["entry"], seen["checks"])


class TestTheArmRises:
    def test_where_the_pipeline_rate_made_it_fall_the_arm_deploys_above_its_untrained_pair(self, tmp_path,
                                                                                          monkeypatch):
        seen = _spies(monkeypatch)
        on = _prepared(tmp_path / "on", key=True)
        F.weight_quantization(on)
        _ir_on, out_on = _deploy(on)
        report, trajectory = seen["reports"][0]
        off = _prepared(tmp_path / "off")
        F.weight_quantization(off)
        _ir_off, out_off = _deploy(off)
        assert report.engaged and report.steps_used > 0
        assert report.exit > report.entry, (report, trajectory)            # the law-in-loop read rose
        assert F.count(out_on, on.y) > F.count(out_off, off.y)              # above the untrained pair
        assert F.deployed_loss(out_on, on.y) < F.deployed_loss(out_off, off.y)


class TestTheRule:
    def test_the_recovery_trains_at_the_explorer_s_rate_on_the_law_in_loop_read(self, tmp_path, monkeypatch):
        seen = _spies(monkeypatch)
        fx = _prepared(tmp_path, key=True)
        F.weight_quantization(fx)
        assert len(seen["explorer"]) == 1, seen["explorer"]
        call = seen["explorer"][0]
        assert call["loop_open"]                     # every probe trained and read the deployed law
        assert call["found"] is not None
        assert seen["legs"] and seen["legs"][0]["lr"] == min(float(call["found"]), LR)

    def test_the_explorer_s_rate_is_capped_at_the_pipeline_s(self, tmp_path, monkeypatch):
        seen = _spies(monkeypatch)
        monkeypatch.setattr(TunerBase, "_find_lr", lambda self: seen["explorer"].append("called") or 10.0 * LR)
        fx = _prepared(tmp_path, key=True)
        F.weight_quantization(fx)
        assert seen["explorer"] == ["called"]
        assert seen["legs"][0]["lr"] == LR

    def test_an_explorer_that_finds_every_rate_destructive_trains_nothing(self, tmp_path, monkeypatch):
        seen = _spies(monkeypatch)
        monkeypatch.setattr(TunerBase, "_find_lr", lambda self: seen["explorer"].append("called") or None)
        on = _prepared(tmp_path / "on", key=True)
        F.weight_quantization(on)
        ir_on, out_on = _deploy(on)
        report, _trajectory = seen["reports"][0]
        legs_on = list(seen["legs"])
        off = _prepared(tmp_path / "off")
        F.weight_quantization(off)
        ir_off, out_off = _deploy(off)
        assert seen["explorer"] == ["called"]
        assert legs_on == [] and report.engaged and report.steps_used == 0 and report.exit == report.entry
        assert F.law_digests(ir_on) == F.law_digests(ir_off)                 # the untrained program, deployed
        assert torch.equal(out_on, out_off)

    def test_the_recovery_takes_the_funded_geometry_over_its_declared_budget(self, tmp_path, monkeypatch):
        seen = _spies(monkeypatch)
        fx = _prepared(tmp_path, key=True)
        F.weight_quantization(fx)
        leg = seen["legs"][0]
        geometry = endpoint_convergence_geometry(STEPS, leg["check_interval"])
        assert leg["max_steps"] == STEPS
        assert (leg["min_steps"], leg["patience"]) == (geometry.min_steps, geometry.patience)


#: The route (b) rows' declared tuning recipe (C5's run config), with the warmup at a quarter so the fixture's 4-step
#: explorer probes take one warmup step as the rows' 20-step probes take two (round(0.1 x 20)).
WARMUP_RECIPE = {"optimizer": "adamw", "weight_decay": 0.01, "betas": [0.9, 0.999], "scheduler": "cosine",
                 "warmup_ratio": 0.25, "grad_clip_norm": 1.0, "layer_wise_lr_decay": 0.75, "label_smoothing": 0.0}


def _probe_rates(monkeypatch):
    """Every constant-rate step run (the LR explorer's probes): the rate it names and, per step, the largest rate of
    its optimizer's groups (the top layer's under layer-wise decay)."""
    probes = []
    run = basic_trainer_steps.train_n_steps

    def train_n_steps(trainer, lr, steps, warmup_steps=0, *, constant_lr=False, optimizer=None):
        record = {"named": float(lr), "constant": bool(constant_lr), "trained": [], "trainer": trainer}
        optimize = trainer._optimize

        def spy(x, y, opt, scaler):
            record["trained"].append(max(float(g["lr"]) for g in opt.param_groups))
            return optimize(x, y, opt, scaler)

        trainer._optimize = spy
        try:
            return run(trainer, lr, steps, warmup_steps, constant_lr=constant_lr, optimizer=optimizer)
        finally:
            del trainer._optimize
            probes.append(record)

    monkeypatch.setattr(basic_trainer_steps, "train_n_steps", train_n_steps)
    return probes


class TestTheExplorerProbesTrainAtTheRatesTheyName:
    def test_under_a_warmup_recipe_every_probe_of_the_arm_trains_at_its_named_rate(self, tmp_path, monkeypatch):
        """U-RAW's A-1: under the rows' warmup recipe, R5's constant-rate step run trains at the warmup's start factor
        (1e-3 of the rate the explorer names: the recipe's LinearLR, frozen by ConstantLR), so the explorer read rates
        it never trained at and handed the arm the pipeline's rate. Under the key, every probe trains at its rate."""
        seen = _spies(monkeypatch)
        probes = _probe_rates(monkeypatch)
        fx = _prepared(tmp_path, key=True, tuning_recipe=dict(WARMUP_RECIPE))
        F.weight_quantization(fx)
        constant = [probe for probe in probes if probe["constant"]]
        assert len(seen["explorer"]) == 1 and constant, (seen["explorer"], probes)
        for probe in constant:
            assert probe["trained"], probe
            assert all(rate == pytest.approx(probe["named"], rel=1e-12) for rate in probe["trained"]), probe
        # the mark lives only around the arm's explorer call
        assert not any(hasattr(probe["trainer"], basic_trainer_steps.CONSTANT_RATE_EXACT) for probe in probes)


class _StepTrainer:
    """The least a constant-rate step run reads: a transform trainer's step optimizer and schedule (its tuning recipe
    on its aux model), a training loader, and an ``_optimize`` that records each step's group rates."""

    device = "cpu"
    _get_optimizer_and_scheduler_steps = PerceptronTransformTrainer._get_optimizer_and_scheduler_steps

    def __init__(self, recipe):
        self.recipe = recipe
        self.aux_model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2))
        self.beta1, self.beta2 = 0.9, 0.99
        self.train_loader = [(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long))] * 4
        self.train_iter = iter(self.train_loader)
        self.rates = []

    def _optimize(self, x, y, optimizer, scaler):
        self.rates.append([float(g["lr"]) for g in optimizer.param_groups])
        optimizer.step()

    def _report(self, *_args):
        pass


def _recipe(warmup_ratio):
    return build_recipe({"tuning_recipe": dict(WARMUP_RECIPE, warmup_ratio=warmup_ratio)}, key="tuning_recipe")


class TestAConstantRateStepRun:
    """``basic_trainer_steps.train_n_steps(..., constant_lr=True)``, the explorer's probe, on a transform trainer."""

    def test_without_the_declaration_s_mark_a_warmup_recipe_run_keeps_r5_s_rates(self):
        """R5's rates, kept wherever the declaration does not mark the trainer: the warmup's start factor (1e-3) of
        every group's base, frozen for the whole run."""
        trainer = _StepTrainer(_recipe(0.1))
        basic_trainer_steps.train_n_steps(trainer, 1e-3, 20, constant_lr=True)
        bases = [float(g["lr"]) for g in build_optimizer(trainer.aux_model, 1e-3, trainer.recipe).param_groups]
        assert len(trainer.rates) == 20
        for rates in trainer.rates:
            assert rates == pytest.approx([1e-3 * base for base in bases], rel=1e-12)

    def test_a_marked_trainer_trains_at_the_rate_it_names_every_group_at_its_own_base(self):
        trainer = _StepTrainer(_recipe(0.1))
        setattr(trainer, basic_trainer_steps.CONSTANT_RATE_EXACT, True)
        basic_trainer_steps.train_n_steps(trainer, 1e-3, 20, constant_lr=True)
        bases = [float(g["lr"]) for g in build_optimizer(trainer.aux_model, 1e-3, trainer.recipe).param_groups]
        assert max(bases) == pytest.approx(1e-3, rel=1e-12)
        assert trainer.rates == [bases] * 20

    @pytest.mark.parametrize("marked", (False, True))
    def test_without_a_warmup_the_mark_changes_nothing(self, marked):
        trainer = _StepTrainer(_recipe(0.0))
        if marked:
            setattr(trainer, basic_trainer_steps.CONSTANT_RATE_EXACT, True)
        basic_trainer_steps.train_n_steps(trainer, 1e-3, 20, constant_lr=True)
        bases = [float(g["lr"]) for g in build_optimizer(trainer.aux_model, 1e-3, trainer.recipe).param_groups]
        assert trainer.rates == [bases] * 20


#: R5's bytes of the guard runs below (computed on the pristine R5 tree, e5f6bb37, plus this file, by its own
#: ``_spies``, ``_prepared`` and ``_guard_digest``: final_units/uraw/tests_first/guard_digests.py): per run, sha256 over
#: the Weight Quantization model's state, the deployed laws and the endpoint report with its trajectory.
R5_GUARD_DIGESTS = {
    "absent": "2e1c46e851f492f7df7ebc098178ca2e50eab3c9db925b6bf196128600b228e9",
    "false": "2e1c46e851f492f7df7ebc098178ca2e50eab3c9db925b6bf196128600b228e9",
    "absent_floor": "a9f1fb7e36276a45e6632f231e9dd22369caaabe2aba4cb71d20b028bb4648e0",
    "false_floor": "a9f1fb7e36276a45e6632f231e9dd22369caaabe2aba4cb71d20b028bb4648e0",
}


def _guard_digest(fx, seen) -> str:
    h = hashlib.sha256()
    for name, value in sorted(F.model_of(fx).state_dict().items()):
        h.update(name.encode())
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(f"{t.dtype}|{tuple(t.shape)}".encode())
            h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
    h.update(json.dumps(F.law_digests(F.soft_core_mapping(fx)), sort_keys=True).encode())
    for report, trajectory in seen["reports"]:
        h.update(json.dumps(asdict(report), sort_keys=True).encode())
        h.update(json.dumps([[float(v) for v in t] for t in trajectory]).encode())
    return h.hexdigest()


class TestTheKeyAbsentOrFalse:
    """Guards: without the key nothing of the rule runs, and every law, state and metric is R5's byte for byte."""

    @pytest.mark.parametrize("key", (None, False))
    @pytest.mark.parametrize("floor", (False, True))
    def test_the_recovery_is_r5_s(self, tmp_path, monkeypatch, key, floor):
        """The law-free recovery engages on this fixture (its read sits below its target), and with ``floor`` a
        declared target floor above every read lifts its target; either way its leg is R5's: the pipeline's rate and
        the unfunded patience stop, no explorer, the recovery called as R5 calls it."""
        seen = _spies(monkeypatch)
        extra = {"wq_endpoint_target_floor": 1.0} if floor else {}
        fx = _prepared(tmp_path, key=key, **extra)
        F.weight_quantization(fx)
        assert seen["explorer"] == [] and seen["recovery_kwargs"] == [["base_steps", "target_floor"]]
        report, _trajectory = seen["reports"][0]
        assert report.engaged and len(seen["legs"]) == 1
        assert seen["legs"][0]["lr"] == LR
        assert (seen["legs"][0]["min_steps"], seen["legs"][0]["patience"]) == (0, _RECOVERY_PATIENCE)
        name = ("absent" if key is None else "false") + ("_floor" if floor else "")
        assert _guard_digest(fx, seen) == R5_GUARD_DIGESTS[name]
