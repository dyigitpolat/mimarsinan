"""Shared helpers for activation adaptation steps (clamp vs ReLU-only).

Used by ClampAdaptationStep and ActivationAdaptationStep to decide whether
any chip-targeted perceptron has a non-ReLU base (GELU, LeakyReLU, etc.).
"""

# Base activation types that produce non-negative outputs and are
# compatible with the chip's hardcoded ReLU.
RELU_COMPATIBLE_TYPES = (
    "LeakyGradReLU",  # forward is pure ReLU; leaky only in backward
    "ReLU",
)


def needs_clamp_adaptation(model) -> bool:
    """True if any chip-targeted perceptron has a non-ReLU base (e.g. GELU, LeakyReLU).

    Used by ActivationAdaptationStep (ReLU replacement when True) and by
    ClampAdaptationStep (short-path when False). Non-chip-supported perceptrons
    (e.g. Identity) are excluded from get_perceptrons().
    """
    for p in model.get_perceptrons():
        base = p.base_activation
        name = type(base).__name__
        if name not in RELU_COMPATIBLE_TYPES:
            return True  # chip-targeted but not directly ReLU-compatible → needs clamp
    return False
