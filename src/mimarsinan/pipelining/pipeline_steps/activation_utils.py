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

# Activations that become host-side ComputeOps — no adaptation needed.
HOST_SIDE_TYPES = ("Identity",)


def needs_clamp_adaptation(model) -> bool:
    """Check whether any chip-targeted perceptron needs clamp adaptation.

    Identity perceptrons are host-side pass-throughs and are excluded from
    get_perceptrons(); they do not appear here. Non-ReLU chip-targeted
    perceptrons (e.g. GELU, LeakyReLU) need clamp calibration before the
    activation is replaced by a chip-supported form (e.g. ReLU).
    """
    for p in model.get_perceptrons():
        base = p.base_activation
        name = type(base).__name__
        if name in HOST_SIDE_TYPES:
            continue  # redundant guard; Identity is already excluded by get_perceptrons()
        if name not in RELU_COMPATIBLE_TYPES:
            return True  # chip-targeted but not directly ReLU-compatible → needs clamp
    return False
