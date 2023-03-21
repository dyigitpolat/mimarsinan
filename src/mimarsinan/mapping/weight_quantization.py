import numpy as np

class WeightQuantization:
    def __init__(self, bits):
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1

    def quantize_weight_(self, w, min_w, max_w):
        if(w > 0):
            if(max_w == 0): return 0
            return round((self.q_max * w) / max_w)
        else:
            if(min_w == 0): return 0
            return round((self.q_min * w) / min_w)

    def quantize(self, weights):
        max_w = weights.max().item()
        min_w = weights.min().item()
        return np.array([
            [self.quantize_weight_(w, min_w, max_w) for w in weights.flatten()]
        ]).reshape(weights.shape)

    def quantize_without_scaling(self, weights):
        max_w = weights.max().item()
        min_w = weights.min().item()
        return np.where(
            weights > 0,
            np.round(((self.q_max) * (weights)) / (max_w)) / (self.q_max / max_w),
            np.round(((self.q_min) * (weights)) / (min_w)) / (self.q_min / min_w))
    
    def calculate_threshold(self, weights):
        max_w = weights.max().item()
        if(max_w == 0): max_w = 1.0
        return max(1, round((self.q_max * 1.0) / max_w))
    
    def scale(self, weights):
        max_w = weights.max().item()
        min_w = weights.min().item()
        if(max_w == 0): max_w = 1.0
        if(min_w == 0): min_w = 1.0

        return np.where(
            weights > 0,
            weights * (self.q_max / max_w),
            weights * (self.q_min / min_w))

def quantize_cores(cores, bits):
    quantizer = WeightQuantization(bits)

    for core in cores:
        core.threshold = quantizer.calculate_threshold(core.core_matrix)
        core.core_matrix = quantizer.quantize(core.core_matrix)
        
    return cores

def quantize_cores_without_scaling(cores, bits):
    quantizer = WeightQuantization(bits)

    for core in cores:
        core.threshold = quantizer.calculate_threshold(core.core_matrix)
        core.core_matrix = quantizer.quantize_without_scaling(core.core_matrix)
        
    return cores

def calculate_core_thresholds(cores, bits):
    quantizer = WeightQuantization(bits)
    for core in cores:
        core.threshold = quantizer.calculate_threshold(core.core_matrix)
    return cores

def scale_quantized_cores(cores, bits):
    quantizer = WeightQuantization(bits)
    for core in cores:
        core.core_matrix *= quantizer.scale(core.core_matrix)
    return cores