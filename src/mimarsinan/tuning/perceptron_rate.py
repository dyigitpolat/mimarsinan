"""SSOT for applying a transformation rate across a model's perceptrons."""

from __future__ import annotations

from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation


def rebuild_activations(model, adaptation_manager, config) -> None:
    """Rebuild every perceptron's activation from the manager's current rates."""
    for perceptron in model.get_perceptrons():
        adaptation_manager.update_activation(config, perceptron)


def apply_manager_rate(model, adaptation_manager, config, rate_attr: str, rate) -> None:
    """Set ``adaptation_manager.<rate_attr> = rate`` and rebuild all activations.

    The SSOT for the ``AdaptationRateTuner`` family (``quantization_rate``,
    ``noise_rate``, …): a global manager field plus a full per-perceptron rebuild.
    """
    setattr(adaptation_manager, rate_attr, rate)
    rebuild_activations(model, adaptation_manager, config)


def set_blend_rate(model, rate: float) -> None:
    """Set the ``rate`` of every perceptron's ``BlendActivation`` (the ANN→SNN
    blend ramp). The SSOT for the ``KDBlendAdaptationTuner`` family, where the
    blend module *is* the activation (no decorator rebuild needed)."""
    r = float(rate)
    for perceptron in model.get_perceptrons():
        perceptron.base_activation.rate = r


def set_surrogate_alpha(model, a: float) -> None:
    """Set the spike-surrogate sharpness on every ``TTFSActivation`` under ``model``.

    ``alpha`` shapes only the ATan backward, never the ``pre > 0`` fire forward.
    """
    alpha = float(a)
    for module in model.modules():
        if isinstance(module, TTFSActivation):
            module.set_surrogate_alpha(alpha)
