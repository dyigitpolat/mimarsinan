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
class BasicClassificationLoss:
    def __call__(self, model, x, y):
        return nn.CrossEntropyLoss(label_smoothing=0.1)(model(x), y)
    
import torch

from mimarsinan.models.nn.layers import SavedTensorDecorator

class CustomClassificationLoss:
    def __init__(self):
        self.main_loss_avg = None
        self.act_loss_avg = None

    def __call__(self, model, x, y):

        for perceptron in model.get_perceptrons():
            perceptron.activation.decorate(SavedTensorDecorator())

        model_out = model(x)
        classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(model_out, y)

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
