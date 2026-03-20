"""Shared helpers for activation adaptation steps.

Used by ActivationAdaptationStep and ClampAdaptationStep to check whether
any chip-targeted perceptron has a non-ReLU base (GELU, LeakyReLU, etc.).

Contract: callers pass only perceptrons returned by ``model.get_perceptrons()``,
which already excludes host-side Identity perceptrons via
``owned_perceptron_groups()`` in every mapper.  No Identity special-casing is
needed here.
"""

# Base activation types that produce non-negative outputs and are
# compatible with the chip's hardcoded ReLU.
RELU_COMPATIBLE_TYPES = (
    "LeakyGradReLU",  # forward is pure ReLU; leaky only in backward
    "ReLU",
)


def needs_relu_adaptation(perceptron) -> bool:
    """True when a chip-targeted perceptron still needs ReLU adaptation.

    All perceptrons passed here are guaranteed chip-targeted by the mapper
    contract (``owned_perceptron_groups()`` never returns Identity perceptrons).
    This function only checks whether the activation is already ReLU-compatible.
    """
    base_name = type(perceptron.base_activation).__name__
    return base_name not in RELU_COMPATIBLE_TYPES


def has_non_relu_activations(model) -> bool:
    """True if any chip-targeted perceptron has a non-ReLU base (e.g. GELU, LeakyReLU)."""
    for p in model.get_perceptrons():
        if needs_relu_adaptation(p):
            return True
    return False
