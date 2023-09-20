def calculate_activation_shift(quantization_level, activation_scale):
    return (activation_scale * 0.5) / (quantization_level)