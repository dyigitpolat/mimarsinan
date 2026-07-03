"""Shared helpers for activation adaptation steps."""

RELU_COMPATIBLE_TYPES = (
    "LeakyGradReLU",  # forward is pure ReLU; leaky only in backward
    "ReLU",
)


def needs_relu_adaptation(perceptron) -> bool:
    """True when a perceptron's base activation is not already ReLU-compatible."""
    base_name = type(perceptron.base_activation).__name__
    return base_name not in RELU_COMPATIBLE_TYPES


def has_non_relu_activations(model) -> bool:
    """True if any chip-targeted perceptron has a non-ReLU base (e.g. GELU, LeakyReLU)."""
    for p in model.get_perceptrons():
        if needs_relu_adaptation(p):
            return True
    return False
