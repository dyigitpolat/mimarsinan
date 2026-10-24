"""Contracts for TrainingRecipe consumption: betas arity and recipe-required step builders."""

import pytest
import torch.nn as nn

from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import (
    TrainingRecipe,
    build_optimizer,
)
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from conftest import MockDataProviderFactory


def _tiny_model():
    return nn.Linear(4, 2)


class TestBetasArityContract:
    def test_two_betas_as_list_accepted(self):
        recipe = TrainingRecipe(optimizer="adam", betas=[0.9, 0.95])
        optimizer = build_optimizer(_tiny_model(), 1e-3, recipe)
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.95)

    def test_two_betas_as_tuple_accepted_for_adamw(self):
        recipe = TrainingRecipe(optimizer="adamw", betas=(0.9, 0.999))
        optimizer = build_optimizer(_tiny_model(), 1e-3, recipe)
        assert optimizer.param_groups[0]["betas"] == (0.9, 0.999)

    def test_three_betas_rejected_loudly(self):
        recipe = TrainingRecipe(optimizer="adam", betas=(0.9, 0.99, 0.999))
        with pytest.raises(ValueError):
            build_optimizer(_tiny_model(), 1e-3, recipe)

    def test_one_beta_rejected_loudly(self):
        recipe = TrainingRecipe(optimizer="adamw", betas=(0.9,))
        with pytest.raises(ValueError):
            build_optimizer(_tiny_model(), 1e-3, recipe)


class _WrapperLoss:
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss()(model(x), y)


def _make_recipeless_trainer():
    dp_factory = MockDataProviderFactory(input_shape=(1, 8, 8), num_classes=4)
    dlf = DataLoaderFactory(dp_factory, num_workers=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 4))
    return BasicTrainer(model, "cpu", dlf, _WrapperLoss(), recipe=None)


class TestRecipeRequiredForRecipeStepBuilders:
    def test_step_optimizer_without_recipe_raises_clear_error(self):
        trainer = _make_recipeless_trainer()
        with pytest.raises(RuntimeError, match="recipe"):
            trainer._build_recipe_step_optimizer(1e-3)

    def test_step_scheduler_without_recipe_raises_clear_error(self):
        trainer = _make_recipeless_trainer()
        recipe_trainer_optimizer = trainer.build_step_optimizer(1e-3)
        with pytest.raises(RuntimeError, match="recipe"):
            trainer._build_recipe_step_scheduler(recipe_trainer_optimizer, 10)


class TestOneReadingOfTheClipNorm:
    """[C4] ``grad_clip_norm`` has one reading for every trainer: an absent or non-positive norm is no clipping
    (``TrainingRecipe`` supplies 0 when a declared recipe omits the field; clipping to norm 0 zeroes every
    gradient, and a negative norm reverses it)."""

    @pytest.mark.parametrize("norm,expected", [(None, None), (0, None), (0.0, None), (-1.0, None),
                                               (0.25, 0.25), (1, 1.0)])
    def test_the_clip_norm_a_declaration_asks_for(self, norm, expected):
        from mimarsinan.model_training.training_recipe import clip_norm_of

        assert clip_norm_of(norm) == expected

    def test_the_trainer_reads_the_one_rule(self, monkeypatch):
        import torch
        from mimarsinan.model_training import basic_trainer

        seen = []
        monkeypatch.setattr(basic_trainer, "clip_norm_of", lambda norm: seen.append(norm) or None)
        trainer = _make_recipeless_trainer()
        trainer.recipe = TrainingRecipe(grad_clip_norm=0.5)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        x, y = torch.zeros(2, 1, 8, 8), torch.zeros(2, dtype=torch.long)
        trainer._optimize(x, y, optimizer, scaler)
        assert seen == [0.5]
