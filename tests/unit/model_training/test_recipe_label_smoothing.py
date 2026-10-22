"""TrainingRecipe.label_smoothing can reach the classification loss (opt-in).

The field used to be dead: every classification loss hardcoded 0.1. The losses
now take a smoothing value (``None`` = the legacy 0.1) and
``apply_recipe_label_smoothing`` rebinds a recipe's value onto a copy of the
loss. ``BasicTrainer`` applies it only with ``honor_recipe_label_smoothing=True``:
tracked configs declare 0 that was never read, so honoring it by default would
change how they train (every config's effective value is pinned by
``test_effective_label_smoothing.py``).
"""

import pytest
import torch
import torch.nn as nn

from mimarsinan.config_schema.defaults import (
    DEFAULT_TRAINING_RECIPE,
    DEFAULT_TUNING_RECIPE,
)
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory
from mimarsinan.model_training.basic_trainer import BasicTrainer
from mimarsinan.model_training.training_recipe import TrainingRecipe, build_recipe
from mimarsinan.model_training.training_utilities import (
    LEGACY_LABEL_SMOOTHING,
    BasicClassificationLoss,
    CustomClassificationLoss,
    apply_recipe_label_smoothing,
    resolve_label_smoothing,
)
from conftest import MockDataProviderFactory


def _batch():
    torch.manual_seed(0)
    model = nn.Linear(6, 5)
    x = torch.randn(16, 6)
    y = torch.randint(0, 5, (16,))
    return model, x, y


class TestResolution:
    def test_none_is_the_legacy_point_one(self):
        assert LEGACY_LABEL_SMOOTHING == 0.1
        assert resolve_label_smoothing(None) == 0.1

    def test_explicit_values_pass_through(self):
        assert resolve_label_smoothing(0) == 0.0
        assert resolve_label_smoothing(0.25) == 0.25

    @pytest.mark.parametrize("bad", [-0.1, 1.0, 1.5])
    def test_out_of_range_rejected(self, bad):
        with pytest.raises(ValueError):
            resolve_label_smoothing(bad)


class TestBasicClassificationLoss:
    def test_default_is_bit_identical_to_the_old_hardcoded_loss(self):
        model, x, y = _batch()
        got = BasicClassificationLoss()(model, x, y)
        want = nn.CrossEntropyLoss(label_smoothing=0.1)(model(x), y)
        assert torch.equal(got, want)

    def test_explicit_zero_is_plain_cross_entropy(self):
        model, x, y = _batch()
        got = BasicClassificationLoss(0.0)(model, x, y)
        assert torch.equal(got, nn.CrossEntropyLoss()(model(x), y))

    def test_with_label_smoothing_returns_a_new_loss(self):
        base = BasicClassificationLoss()
        other = base.with_label_smoothing(0.3)
        assert other is not base and isinstance(other, BasicClassificationLoss)
        assert base.label_smoothing == 0.1 and other.label_smoothing == 0.3

    def test_custom_loss_carries_the_same_seam(self):
        loss = CustomClassificationLoss().with_label_smoothing(0.0)
        assert isinstance(loss, CustomClassificationLoss)
        assert loss.label_smoothing == 0.0
        assert CustomClassificationLoss().label_smoothing == 0.1


class TestApplyRecipe:
    def test_no_recipe_or_none_field_keeps_the_loss_object(self):
        loss = BasicClassificationLoss()
        assert apply_recipe_label_smoothing(loss, None) is loss
        assert apply_recipe_label_smoothing(loss, TrainingRecipe()) is loss

    def test_declared_value_reaches_a_copy(self):
        loss = BasicClassificationLoss()
        got = apply_recipe_label_smoothing(loss, TrainingRecipe(label_smoothing=0.0))
        assert got is not loss
        assert got.label_smoothing == 0.0
        assert loss.label_smoothing == 0.1

    def test_loss_without_smoothing_term_is_unchanged(self):
        class _Mse:
            def __call__(self, model, x, y):
                return nn.MSELoss()(model(x), y)

        loss = _Mse()
        assert apply_recipe_label_smoothing(loss, TrainingRecipe(label_smoothing=0.2)) is loss


def _trainer(recipe, loss, **kwargs):
    dlf = DataLoaderFactory(MockDataProviderFactory(input_shape=(1, 8, 8), num_classes=4), num_workers=0)
    model = nn.Sequential(nn.Flatten(), nn.Linear(64, 4))
    return BasicTrainer(model, "cpu", dlf, loss, recipe=recipe, **kwargs)


class TestTrainerWiring:
    def test_default_trainer_keeps_the_loss_whatever_the_recipe_declares(self):
        loss = BasicClassificationLoss()
        trainer = _trainer(TrainingRecipe(label_smoothing=0.0), loss)
        assert trainer.honor_recipe_label_smoothing is False
        assert trainer.loss_function is loss
        assert trainer.loss_function.label_smoothing == 0.1

    def test_opted_in_trainer_uses_the_recipe_value(self):
        loss = BasicClassificationLoss()
        trainer = _trainer(TrainingRecipe(label_smoothing=0.0), loss, honor_recipe_label_smoothing=True)
        assert trainer.loss_function.label_smoothing == 0.0
        assert loss.label_smoothing == 0.1  # the shared pipeline loss is untouched

    def test_trainer_without_recipe_keeps_legacy(self):
        loss = BasicClassificationLoss()
        trainer = _trainer(None, loss, honor_recipe_label_smoothing=True)
        assert trainer.loss_function is loss
        assert trainer.loss_function.label_smoothing == 0.1


class TestDefaultsStayByteIdentical:
    @pytest.mark.parametrize("key,block", [
        ("training_recipe", DEFAULT_TRAINING_RECIPE),
        ("tuning_recipe", DEFAULT_TUNING_RECIPE),
    ])
    def test_default_recipes_train_at_the_legacy_point_one(self, key, block):
        recipe = build_recipe({key: dict(block)}, key=key)
        trainer = _trainer(recipe, BasicClassificationLoss())
        assert trainer.loss_function.label_smoothing == LEGACY_LABEL_SMOOTHING
