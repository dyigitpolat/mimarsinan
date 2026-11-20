"""[U-TGAUGE T1-C] A trainer optimizes exactly the parameters its update writes to the deployed model.

The transform trainers train an aux copy and write it into the deployed model through their update: the perceptron
trainer writes each perceptron's own parameters (through the projection), the weight trainer every parameter named a
weight or a bias. The perceptron trainer optimized every aux parameter, the host ops' and host normalizations' with
them, and counted their gradients in its clip: a Weight Quantization endpoint trained every host value, none of which
ever reached the deployed model, and their gradients scaled the clip of the ones that did. The rule (the T1 read's supported
form): the optimizer's groups and the clip read one set, the parameters the update writes; every optimizer a trainer
builds (the recipe's, the plain one, the step optimizer) reads it.

h-1: every parameter in an optimizer's groups is written to the deployed model by one update, and every written one is
in the groups. h-2: the clip norm covers exactly those parameters. h-3: a host parameter's value in the deployed model is
unchanged by an endpoint leg (true before the rule too, by the update's copy), and its aux copy is no longer stepped.
[U-TGAUGE2] h-4 (the verify's m-4): a perceptron carries trainable parameters outside its layer (a normalization inside
it), which its update writes, so they are optimized; the form pins that half of h-1 on its own.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import MockDataProviderFactory

from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.perceptron_transform_trainer import PerceptronTransformTrainer
from mimarsinan.model_training.training_recipe import TrainingRecipe
from mimarsinan.model_training.training_utilities import BasicClassificationLoss
from mimarsinan.model_training.weight_transform_trainer import WeightTransformTrainer
from mimarsinan.torch_mapping.converter import convert_torch_model

N_OUT = 6
#: The WQ endpoint's declared recipe (AdamW, weight decay 0.01, cosine with warmup, clip 1.0).
RECIPE = TrainingRecipe(optimizer="adamw", weight_decay=0.01, betas=(0.9, 0.999), scheduler="cosine",
                        warmup_ratio=0.1, grad_clip_norm=1.0)


class HostTail(nn.Module):
    """A subsumed encoding and a core, then a host normalization and a host linear head (a classifier tail)."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.bn, self.head = nn.Linear(8, 6), nn.Linear(6, 6), nn.BatchNorm1d(6), nn.Linear(6, N_OUT)
        self.r0, self.r1 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.head(self.bn(self.r1(self.fc1(self.r0(self.stem(x.flatten(1)))))))


class NormTail(nn.Module):
    """A subsumed encoding, a core with a normalization inside its perceptron (trainable outside its layer), a host
    linear head."""

    shape = (8,)

    def __init__(self):
        super().__init__()
        self.stem, self.fc1, self.bn1, self.head = nn.Linear(8, 6), nn.Linear(6, 6), nn.BatchNorm1d(6), nn.Linear(6, N_OUT)
        self.r0, self.r1 = nn.ReLU(), nn.ReLU()

    def forward(self, x):
        return self.head(self.r1(self.bn1(self.fc1(self.r0(self.stem(x.flatten(1)))))))


class GainNet(nn.Module):
    """A layer and a free gain parameter (named neither weight nor bias)."""

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(8, N_OUT)
        self.gain = nn.Parameter(torch.tensor(1.5))

    def forward(self, x):
        return self.layer(x.flatten(1)) * self.gain


def _loader():
    return DataLoaderFactory(MockDataProviderFactory(input_shape=(8,), num_classes=N_OUT, size=16), num_workers=0)


def _flow(cls=HostTail):
    torch.manual_seed(0)
    net = cls().eval()
    return convert_torch_model(net, net.shape, N_OUT, device="cpu", encoding_layer_placement="subsume").train()


def _perceptron_trainer(recipe=RECIPE, cls=HostTail):
    return PerceptronTransformTrainer(_flow(cls), "cpu", _loader(), BasicClassificationLoss(), lambda perceptron: None,
                                      recipe=recipe)


def _norm_perceptron_trainer(recipe=RECIPE):
    return _perceptron_trainer(recipe, NormTail)


def _weight_trainer(recipe=RECIPE):
    torch.manual_seed(0)
    return WeightTransformTrainer(GainNet(), "cpu", _loader(), BasicClassificationLoss(), lambda t: t, recipe=recipe)


def _plain_trainer(recipe=RECIPE):
    return BasicTrainer(_flow(), "cpu", _loader(), BasicClassificationLoss(), recipe=recipe)


def _trained_model(trainer):
    return getattr(trainer, "aux_model", trainer.model)


def _written_names(trainer) -> set:
    """The trained model's parameters one update writes into the deployed model: each is perturbed, the update runs,
    and the deployed parameters that changed name the written ones."""
    deployed = dict(trainer.model.named_parameters())
    before = {n: p.detach().clone() for n, p in deployed.items()}
    if not hasattr(trainer, "aux_model"):
        return set(deployed)  # a plain trainer's update is the optimizer step on the deployed model itself
    with torch.no_grad():
        for p in trainer.aux_model.parameters():
            p.add_(0.37)
        trainer._update_and_transform_model()
    return {n for n, p in deployed.items() if not torch.equal(p.detach(), before[n])}


def _names(trainer, params) -> set:
    """The trained model's names of ``params``; a parameter of the deployed model itself (which an aux trainer's update
    overwrites) is named with a ``deployed:`` prefix."""
    by_id = {id(p): f"deployed:{n}" for n, p in trainer.model.named_parameters()}
    by_id.update({id(p): n for n, p in _trained_model(trainer).named_parameters()})
    return {by_id[id(p)] for p in params}


def _optimizer_names(trainer, optimizer) -> set:
    return _names(trainer, [p for g in optimizer.param_groups for p in g["params"]])


def _recovery_step_optimizer(trainer):
    """The step optimizer of the recipe-driven step recovery (``tuning_recipe_recovery``)."""
    trainer.tuning_recipe_recovery = True
    return trainer.build_step_optimizer(1e-3)


BUILDERS = {
    "steps": lambda t: t._get_optimizer_and_scheduler_steps(1e-3, 100)[0],
    "epochs": lambda t: t._get_optimizer_and_scheduler(1e-3, 2)[0],
    "step_optimizer": lambda t: t.build_step_optimizer(1e-3),
    "recovery_step_optimizer": _recovery_step_optimizer,
}
TRAINERS = {"perceptron": _perceptron_trainer, "perceptron_norm": _norm_perceptron_trainer, "weight": _weight_trainer,
            "plain": _plain_trainer}


@pytest.mark.parametrize("recipe", [RECIPE, None], ids=["recipe", "no_recipe"])
@pytest.mark.parametrize("builder", sorted(BUILDERS))
@pytest.mark.parametrize("kind", sorted(TRAINERS))
def test_h_1_every_optimized_parameter_is_written_by_the_update_and_every_written_one_is_optimized(kind, builder,
                                                                                                   recipe):
    trainer = TRAINERS[kind](recipe)
    optimized = _optimizer_names(trainer, BUILDERS[builder](trainer))
    written = _written_names(TRAINERS[kind](recipe))
    trainable = {n for n, p in _trained_model(trainer).named_parameters() if p.requires_grad}
    assert optimized <= written and (written & trainable) <= optimized, {
        "optimized_not_written": sorted(optimized - written), "written_not_optimized": sorted((written & trainable) - optimized)}


def test_h_1_the_perceptron_trainer_holds_every_host_parameter_out():
    trainer = _perceptron_trainer()
    optimized = _optimizer_names(trainer, BUILDERS["steps"](trainer))
    host = {n for n, _ in trainer.aux_model.named_parameters()} - {
        f"{name}.{pn}" for name, m in trainer.aux_model.named_modules()
        if type(m).__name__ == "Perceptron" for pn, _ in m.named_parameters()}
    assert host and not (host & optimized), sorted(host & optimized)


@pytest.mark.parametrize("kind", ["perceptron", "weight"])
def test_h_2_the_clip_norm_covers_exactly_the_written_parameters(kind, monkeypatch):
    trainer = TRAINERS[kind]()
    optimizer, _scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-3, 100)
    seen = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        seen.append(parameters)
        return real(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    x, y = trainer.next_training_batch()
    trainer._optimize(x, y, optimizer, scaler)
    assert len(seen) == 1
    assert _names(trainer, seen[0]) == _written_names(TRAINERS[kind]())


def _host_names(trainer) -> list:
    perceptron_params = {id(p) for _m, aux_p, _t in trainer._perceptron_slots for p in aux_p.parameters()}
    return [n for n, p in trainer.aux_model.named_parameters() if id(p) not in perceptron_params]


def _endpoint_leg(trainer, steps: int = 5) -> None:
    optimizer, scheduler, scaler = trainer._get_optimizer_and_scheduler_steps(1e-2, 50)
    for _ in range(steps):
        x, y = trainer.next_training_batch()
        trainer._optimize(x, y, optimizer, scaler)
        scheduler.step()
    with torch.no_grad():
        trainer._update_and_transform_model()


def test_h_3_a_host_parameter_of_the_deployed_model_is_unchanged_by_an_endpoint_leg():
    trainer = _perceptron_trainer()
    deployed = dict(trainer.model.named_parameters())
    host = _host_names(trainer)
    before = {n: deployed[n].detach().clone() for n in host}
    _endpoint_leg(trainer)
    assert host and all(torch.equal(deployed[n].detach(), before[n]) for n in host)


def test_h_3_a_host_parameter_s_trained_copy_is_not_stepped_by_an_endpoint_leg():
    trainer = _perceptron_trainer()
    aux = dict(trainer.aux_model.named_parameters())
    host = _host_names(trainer)
    before = {n: aux[n].detach().clone() for n in host}
    _endpoint_leg(trainer)
    moved = [n for n in host if not torch.equal(aux[n].detach(), before[n])]
    assert not moved, moved


def test_h_3_the_perceptron_parameters_are_still_trained_and_written():
    trainer = _perceptron_trainer()
    deployed = dict(trainer.model.named_parameters())
    written = sorted(_written_names(_perceptron_trainer()))
    before = {n: deployed[n].detach().clone() for n in written}
    _endpoint_leg(trainer)
    assert any(not torch.equal(deployed[n].detach(), before[n]) for n in written)


@pytest.mark.parametrize("recipe", [RECIPE, None], ids=["recipe", "no_recipe"])
@pytest.mark.parametrize("builder", sorted(BUILDERS))
def test_h_4_a_perceptron_s_trainable_parameters_outside_its_layer_are_optimized(builder, recipe):
    """[U-TGAUGE2] The verify's m-4: a perceptron's own trainable parameters outside its layer (here a normalization
    inside it) are written by the update, so every optimizer holds them."""
    trainer = _norm_perceptron_trainer(recipe)
    outside = {f"{name}.{pn}" for name, m in trainer.aux_model.named_modules() if type(m).__name__ == "Perceptron"
               for pn, p in m.named_parameters() if p.requires_grad and not pn.startswith("layer.")}
    assert outside, "the form carries trainable perceptron parameters outside the layer"
    assert outside <= _written_names(_norm_perceptron_trainer(recipe))
    optimized = _optimizer_names(trainer, BUILDERS[builder](trainer))
    assert outside <= optimized, sorted(outside - optimized)
