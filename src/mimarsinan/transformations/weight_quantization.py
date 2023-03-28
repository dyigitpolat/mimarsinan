import numpy as np
import torch

import math

class TensorQuantization:
    def __init__(self, bits):
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1

    def transform_np_array(self, weight_array, transformation):
        weight_tensor = torch.from_numpy(weight_array)
        quantized_weight_tensor = transformation(weight_tensor)
        return quantized_weight_tensor.detach().numpy()
    
    def get_scale(self, weights):
        if isinstance(weights, np.ndarray):
            weights = torch.from_numpy(weights)

        max_weight = torch.max(weights.flatten()).item()
        min_weight = torch.min(weights.flatten()).item()

        neg_scale = +math.inf
        if abs(min_weight) > 0: neg_scale = abs(self.q_max/min_weight)
        pos_scale = +math.inf
        if abs(max_weight) > 0: pos_scale = abs(self.q_max/max_weight)

        if neg_scale is +math.inf and pos_scale is +math.inf:
            return 1.0
        
        return min(neg_scale, pos_scale)
    
    def quantize(self, weights):
        if isinstance(weights, np.ndarray):
            return self.transform_np_array(weights, self.quantize)
        
        scale = self.get_scale(weights)
        return torch.round(weights * scale) / scale
