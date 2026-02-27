def calculate_activation_shift(quantization_level, activation_scale):
    """Return shift in activation/output space (amount to subtract after ReLU for quantization).

    Callers that bake this into the effective bias must add shift_amount / activation_scale
    to the effective bias so that the pre-activation (BN/layer output) increases by exactly
    shift_amount and the decorator's subtraction cancels correctly."""
    return (activation_scale * 0.5) / (quantization_level)