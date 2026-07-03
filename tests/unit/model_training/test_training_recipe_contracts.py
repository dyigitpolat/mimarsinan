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
