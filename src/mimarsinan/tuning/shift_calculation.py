def calculate_activation_shift(quantization_level, activation_scale):
    """Return the activation-space shift to subtract after ReLU for quantization.

    Callers baking this into the effective bias must add ``shift_amount / activation_scale``
    so the pre-activation rises by exactly ``shift_amount`` and the decorator subtraction cancels.
    """
    return (activation_scale * 0.5) / (quantization_level)