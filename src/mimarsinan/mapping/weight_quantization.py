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
    
    def calculate_threshold(self, weights):
        max_w = weights.max().item()
        if(max_w == 0): max_w = 1.0
        return max(1, round((self.q_max * 1.0) / max_w))

def quantize_cores(cores, bits):
    quantizer = WeightQuantization(bits)

    for core in cores:
        core.threshold = quantizer.calculate_threshold(core.core_matrix)
        core.core_matrix = quantizer.quantize(core.core_matrix)
        
    return cores