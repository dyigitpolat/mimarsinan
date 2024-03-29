from mimarsinan.transformations.transformation_utils import *

import numpy as np
import torch

import math

class TensorQuantization:
    def __init__(self, bits):
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
    
    def quantize(self, weights):
        if isinstance(weights, np.ndarray):
            return transform_np_array(weights, self.quantize)
        
        scale = self.q_max / torch.max(torch.abs(weights))
        return torch.round(weights * scale) / scale
    
    def scaled_quantize(self, weights):
        if isinstance(weights, np.ndarray):
            return transform_np_array(weights, self.scaled_quantize)
        
        # assert torch.max(torch.abs(weights)) <= 1.0, \
        #     f"{torch.max(torch.abs(weights))} > 1.0"
        
        scale = self.q_max / torch.max(torch.abs(weights))
        return torch.round(weights * scale)
