def calculate_activation_shift(quantization_level, base_threshold):
    return (base_threshold * 0.5) / (quantization_level)