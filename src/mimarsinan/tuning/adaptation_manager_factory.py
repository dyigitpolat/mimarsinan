"""Create AdaptationManager instances for pipeline steps."""

from __future__ import annotations

from mimarsinan.tuning.adaptation_manager import AdaptationManager


def create_adaptation_manager_for_model(config, model) -> AdaptationManager:
    manager = AdaptationManager()
    if hasattr(model, "get_perceptrons"):
        for perceptron in model.get_perceptrons():
            manager.update_activation(config, perceptron)
    return manager
