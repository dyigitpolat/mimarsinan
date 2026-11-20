"""[U-TGAUGE2] A step writes what it trains, and gradients never accumulate (the U-TGAUGE verify's m-2, by one rule).

U-TGAUGE (T1-C) made a transform trainer optimize exactly what its update writes: its trained copies, through its
projection. The fast ladder's rung loop runs its own step: it differentiates the tuner's model and steps the optimizer it
built. Without a tuning recipe that optimizer is the trainer's step optimizer, over the trained copies, which the rung's
backward never reaches: the rungs stepped nothing, and the backward's gradients piled up on the model, unclipped, until a
later trainer step read their sum. The weight trainer had the same pile-up in its own step: it handed each gradient to its
trained copy by reference, the optimizer cleared only the copy, and the next backward added into the tensor still held
by the deployed parameter.

The rule: before a step's backward, every gradient it may write starts at zero (the optimizer's parameters and every
parameter of the model it differentiates), and a step whose optimizer holds none of that model's parameters refuses by
name (``clear_step_gradients``); a trainer's gradient transfer moves each gradient (the deployed one is cleared). The class,
swept over every trainer and every step loop: the three trainers' own step, the fast ladder's rung (with and without a
recipe, a plain and a transform trainer), the deployed-risk loop and the search's two evaluators. Each test reads, inside
the backward itself, whether a gradient was already there.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from conftest import MockDataProviderFactory, MockPipeline, default_config, make_tiny_supermodel

from mimarsinan.config_schema.defaults import DEFAULT_TUNING_RECIPE
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer
from mimarsinan.model_training.training_recipe import TrainingRecipe
from mimarsinan.model_training.training_utilities import BasicClassificationLoss
from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.torch_mapping.converter import convert_torch_model
from mimarsinan.tuning.orchestration.adaptation_manager_factory import create_adaptation_manager_for_model

N_OUT = 6
RECIPE = TrainingRecipe(optimizer="adamw", weight_decay=0.01, betas=(0.9, 0.999), scheduler="cosine",
                        warmup_ratio=0.1, grad_clip_norm=1.0)


class _Spy:
    """Inside every backward, before any parameter's gradient is written: whether each parameter of ``model`` already
    held a gradient (a leftover the backward would add into). The hook sits on the loss, the backward's first node."""

    def __init__(self, model):
        self.model, self.leftovers = model, []

    def watch(self, loss):
        def hook(grad):
            self.leftovers.append(sorted(n for n, p in self.model.named_parameters() if p.grad is not None))
            return grad

        loss.register_hook(hook)
        return loss


# ---- the fast ladder's rung ---------------------------------------------------------------------------------------

def _napq_tuner(tmp_path, *, recipe):
    from mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner import (
        NormalizationAwarePerceptronQuantizationTuner,
    )

    cfg = default_config()
    cfg["weight_quantization"] = True
    cfg["optimization_driver"] = "fast"
    cfg["wq_fast_rates"] = [0.5, 1.0]
    cfg["wq_fast_steps_per_rate"] = 3
    cfg["tuning_recipe"] = dict(DEFAULT_TUNING_RECIPE) if recipe else None
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.0
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    return NormalizationAwarePerceptronQuantizationTuner(pipeline, model, cfg["weight_bits"], 0.5, cfg["lr"], manager)


def _clamp_tuner(tmp_path, *, recipe):
    from conftest import make_activation_scale_stats
    from mimarsinan.tuning.tuners.clamp_tuner import ClampTuner

    cfg = default_config()
    cfg.update(optimization_driver="fast", spiking_mode="ttfs_quantized", activation_quantization=True,
               clamp_fast_rates=[0.5], clamp_fast_steps_per_rate=3,
               tuning_recipe=dict(DEFAULT_TUNING_RECIPE) if recipe else None)
    pipeline = MockPipeline(config=cfg, working_directory=str(tmp_path))
    pipeline._target_metric = 0.5
    model = make_tiny_supermodel()
    manager = create_adaptation_manager_for_model(cfg, model)
    scales = [1.0 for _ in model.get_perceptrons()]
    return ClampTuner(pipeline, model, 0.5, 0.001, manager, scales, make_activation_scale_stats(model, scales))


def _spied_rung(tuner, rate=0.5):
    spy = _Spy(tuner.model)
    loss = tuner._fast_loss
    tuner._fast_loss = lambda x, y: spy.watch(loss(x, y))
    tuner._ensure_fast_optimizer()
    tuner._fast_train_rung(rate)
    return spy


def test_a_rung_whose_optimizer_holds_none_of_the_model_s_parameters_refuses_by_name(tmp_path):
    """The transform trainer without a recipe: the step optimizer holds its trained copies, which the rung's backward
    through the tuner's model never reaches. It used to step nothing while the model's gradients piled up."""
    tuner = _napq_tuner(tmp_path, recipe=False)
    try:
        spy = _Spy(tuner.model)
        with pytest.raises(Exception) as refused:
            loss = tuner._fast_loss
            tuner._fast_loss = lambda x, y: spy.watch(loss(x, y))
            tuner._ensure_fast_optimizer()
            tuner._fast_train_rung(0.5)
        assert type(refused.value).__name__ == "UnwrittenStepError", refused.value
        assert "fast ladder rung" in str(refused.value) and "holds none" in str(refused.value)
        assert not spy.leftovers, "the refusal comes before the first backward"
    finally:
        tuner.close()


@pytest.mark.parametrize("recipe", [True, False], ids=["recipe", "no_recipe"])
def test_a_plain_trainer_s_rung_starts_every_backward_from_zero_and_trains(tmp_path, recipe):
    tuner = _clamp_tuner(tmp_path, recipe=recipe)
    try:
        before = copy.deepcopy(tuner.model.state_dict())
        spy = _spied_rung(tuner)
        assert len(spy.leftovers) == 3 and all(not left for left in spy.leftovers), spy.leftovers
        assert any(not torch.equal(before[k], v) for k, v in tuner.model.state_dict().items())
    finally:
        tuner.close()


def test_a_transform_trainer_s_rung_with_a_recipe_starts_every_backward_from_zero(tmp_path):
    """With a recipe the rung's optimizer is built over the model it differentiates: it steps, and nothing piles up."""
    tuner = _napq_tuner(tmp_path, recipe=True)
    try:
        spy = _spied_rung(tuner)
        assert len(spy.leftovers) == 3 and all(not left for left in spy.leftovers), spy.leftovers
    finally:
        tuner.close()


# ---- the trainers' own step ------------------------------------------------------------------------------------------

class GainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, N_OUT)
        self.gain = nn.Parameter(torch.tensor(1.5))

    def forward(self, x):
        return self.layer(x.flatten(1)) * self.gain


class HostTail(nn.Module):
    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.bn, self.head = nn.Linear(8, 6), nn.Linear(6, 6), nn.BatchNorm1d(6), nn.Linear(6, N_OUT)
        self.r0, self.r1 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.head(self.bn(self.r1(self.fc1(self.r0(self.stem(x.flatten(1)))))))


def _loader():
    return DataLoaderFactory(MockDataProviderFactory(input_shape=(8,), num_classes=N_OUT, size=16), num_workers=0)


def _flow():
    torch.manual_seed(0)
    net = HostTail().eval()
    return convert_torch_model(net, net.shape, N_OUT, device="cpu", encoding_layer_placement="subsume").train()


TRAINERS = {
    "weight": lambda: WeightTransformTrainer(GainNet(), "cpu", _loader(), BasicClassificationLoss(), lambda t: t,
                                             recipe=RECIPE),
    "perceptron": lambda: PerceptronTransformTrainer(_flow(), "cpu", _loader(), BasicClassificationLoss(),
                                                     lambda perceptron: None, recipe=RECIPE),
    "plain": lambda: BasicTrainer(_flow(), "cpu", _loader(), BasicClassificationLoss(), recipe=RECIPE),
}


def _trained(trainer):
    return getattr(trainer, "aux_model", trainer.model)


@pytest.mark.parametrize("kind", sorted(TRAINERS))
def test_a_trainer_s_step_starts_every_backward_from_zero(kind):
    torch.manual_seed(0)
    trainer = TRAINERS[kind]()
    spy = _Spy(trainer.model)
    loss_function = trainer.loss_function
    trainer.loss_function = lambda model, x, y: spy.watch(loss_function(model, x, y))
    optimizer, _scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-3, 10)
    for _ in range(3):
        x, y = trainer.next_training_batch()
        trainer._optimize(x, y, optimizer, scaler)
    assert len(spy.leftovers) == 3 and all(not left for left in spy.leftovers), spy.leftovers


@pytest.mark.parametrize("kind", sorted(TRAINERS))
def test_a_trainer_s_step_reads_its_own_gradient_only(kind, monkeypatch):
    """The gradient the optimizer steps at the second step is that step's own: a fresh backward at the same parameters
    on the same batch (the clip spied, where it reads the gradients just before the optimizer does)."""
    torch.manual_seed(0)
    trainer = TRAINERS[kind]()
    optimizer, _scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-3, 10)
    batches = [trainer.next_training_batch() for _ in range(2)]
    trainer._optimize(*batches[0], optimizer, scaler)
    if hasattr(trainer, "_update_and_transform_model"):
        with torch.no_grad():
            trainer._update_and_transform_model()  # the deployed model the second step's forward reads
    probe = copy.deepcopy(trainer.model)
    probe.zero_grad(set_to_none=True)
    trainer.loss_function(probe, *batches[1]).backward()
    fresh = {n: p.grad.detach().clone() for n, p in probe.named_parameters() if p.grad is not None}
    seen = {}
    real = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        names = {id(p): n for n, p in _trained(trainer).named_parameters()}
        seen.update({names[id(p)]: p.grad.detach().clone() for p in parameters if p.grad is not None})
        return real(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    trainer._optimize(*batches[1], optimizer, scaler)
    assert seen and set(seen) <= set(fresh)
    for name, grad in seen.items():
        torch.testing.assert_close(grad, fresh[name], rtol=1e-6, atol=1e-7, msg=name)


# ---- the deployed-risk loop ------------------------------------------------------------------------------------------

def test_the_deployed_risk_loop_starts_every_backward_from_zero():
    from mimarsinan.tuning.orchestration.deployed_risk_finetune import run_deployed_risk_finetune

    torch.manual_seed(0)
    model = GainNet()
    teacher = copy.deepcopy(model).eval()
    spy = _Spy(model)
    g = torch.Generator().manual_seed(1)
    batches = [(torch.rand(8, 8, generator=g), torch.randint(0, N_OUT, (8,), generator=g)) for _ in range(3)]
    run_deployed_risk_finetune(model, teacher, lambda x: spy.watch(model(x)), train_batches=batches,
                               eval_genuine=lambda: 0.0, steps=3, lr=1e-3, eval_every=100)
    assert len(spy.leftovers) == 3 and all(not left for left in spy.leftovers), spy.leftovers


@pytest.mark.parametrize("kind", ["fast", "extrapolating"])
def test_a_search_evaluator_s_step_starts_every_backward_from_zero(kind):
    """The search's two evaluators run their own step on the candidate (its optimizer over every parameter): the spy
    sits on the model's output, the backward's first node they share."""
    from mimarsinan.search.evaluators.extrapolating_accuracy_evaluator import ExtrapolatingAccuracyEvaluator
    from mimarsinan.search.evaluators.fast_accuracy_evaluator import FastAccuracyEvaluator

    torch.manual_seed(0)
    model = GainNet()
    spy = _Spy(model)
    model.register_forward_hook(lambda _m, _i, out: spy.watch(out) if out.requires_grad else out)
    evaluator = {"fast": FastAccuracyEvaluator, "extrapolating": ExtrapolatingAccuracyEvaluator}[kind](
        MockDataProviderFactory(input_shape=(8,), num_classes=N_OUT, size=16), torch.device("cpu"), 1e-3)
    evaluator.evaluate(model)
    assert len(spy.leftovers) == 4 and all(not left for left in spy.leftovers), spy.leftovers


def test_the_step_clear_clears_every_parameter_of_the_model_and_refuses_an_unwritten_step():
    """``clear_step_gradients`` itself: an optimizer over part of the model still leaves no gradient anywhere on it; an
    optimizer over none of the model's parameters refuses by name."""
    from mimarsinan.model_training.training_recipe import UnwrittenStepError, clear_step_gradients

    model, other = GainNet(), GainNet()
    model(torch.rand(2, 8)).sum().backward()
    clear_step_gradients(torch.optim.SGD([model.gain], lr=0.1), model, step="a partial step")
    assert all(p.grad is None for p in model.parameters())
    with pytest.raises(UnwrittenStepError, match="a foreign step"):
        clear_step_gradients(torch.optim.SGD(other.parameters(), lr=0.1), model, step="a foreign step")
