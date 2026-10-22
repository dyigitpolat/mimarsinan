from typing import Any

class AccuracyTracker:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def create_hook(self, y):
        def hook(module, input, output):
            _, predicted = output.max(1)
            self.total += float(y.size(0))
            self.correct += float(predicted.eq(y).sum().item())
        
        return hook

    def get_accuracy(self):
        return self.correct / self.total

    def reset(self):
        self.correct = 0
        self.total = 0

import torch.nn as nn

LEGACY_LABEL_SMOOTHING = 0.1
"""The label smoothing every classification loss trained with before the
recipe field reached the loss (it was hardcoded here). ``None`` in a recipe
resolves to it, so configs that never set the field stay byte-identical."""


def resolve_label_smoothing(label_smoothing) -> float:
    """``None`` -> the legacy 0.1; anything else -> its float value in [0, 1)."""
    if label_smoothing is None:
        return LEGACY_LABEL_SMOOTHING
    value = float(label_smoothing)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing!r}")
    return value


def apply_recipe_label_smoothing(loss_function, recipe):
    """The loss a trainer uses under ``recipe``: a copy of ``loss_function`` with
    ``recipe.label_smoothing`` when the recipe declares one and the loss is a
    label-smoothed classification loss (it exposes ``with_label_smoothing``).

    The caller's loss object is never mutated (pipelines share one loss across
    trainers built with different recipes). Losses without a label-smoothing
    term (regression, KD wrappers) are returned unchanged.
    """
    if recipe is None or getattr(recipe, "label_smoothing", None) is None:
        return loss_function
    rebind = getattr(loss_function, "with_label_smoothing", None)
    if rebind is None:
        return loss_function
    return rebind(recipe.label_smoothing)


class BasicClassificationLoss:
    def __init__(self, label_smoothing=None):
        self.label_smoothing = resolve_label_smoothing(label_smoothing)

    def with_label_smoothing(self, label_smoothing):
        return type(self)(label_smoothing)

    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(model(x), y)


import torch

from mimarsinan.models.nn.layers import SavedTensorDecorator

class CustomClassificationLoss:
    def __init__(self, label_smoothing=None):
        self.label_smoothing = resolve_label_smoothing(label_smoothing)
        self.main_loss_avg = None
        self.act_loss_avg = None

    def with_label_smoothing(self, label_smoothing):
        return type(self)(label_smoothing)

    def __call__(self, model, x, y):

        for perceptron in model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        model_out = model(x)
        classification_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(model_out, y)

        act_losses = torch.zeros(len(model.get_perceptrons()), device=model_out.device)

        for idx, perceptron in enumerate(model.get_perceptrons()):
            saved_tensor = perceptron.activation.pop_decorator()
            flat_acts = saved_tensor.latest_output.view(-1)

            act_dist = nn.ReLU()(flat_acts - 1)
            act_losses[idx] = torch.sum(act_dist * nn.Softmax(dim=0)(act_dist))

        act_loss = torch.sum(act_losses * nn.Softmax(dim=0)(act_losses))

        classification_loss = classification_loss + act_loss

        return classification_loss


class EpochArgTolerantScheduler:
    """Drop the epoch argument ``GradualWarmupScheduler`` passes downstream.

    The third-party warmup wrapper calls ``after_scheduler.step(None)``
    (``warmup_scheduler/scheduler.py``), but modern ``SequentialLR.step()``
    takes no argument — every from-scratch recipe whose warmup builds a
    SequentialLR dies with a TypeError. Dropping a ``None`` epoch is exactly
    the wrapper's intent; an EXPLICIT epoch has no modern equivalent and
    fails loud rather than silently moving the LR curve.
    """

    def __init__(self, scheduler: Any) -> None:
        self._scheduler = scheduler

    def step(self, epoch: Any = None, *args: Any, **kwargs: Any) -> Any:
        if epoch is not None:
            raise ValueError(
                f"epoch-indexed scheduler stepping (epoch={epoch!r}) is not "
                f"supported by {type(self._scheduler).__name__}: modern torch "
                f"schedulers advance implicitly. Step without an epoch."
            )
        return self._scheduler.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._scheduler, name)
