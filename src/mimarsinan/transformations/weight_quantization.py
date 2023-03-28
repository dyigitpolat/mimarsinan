import numpy as np
import torch

import math

def np_topk(a, k):
    if k == 0:
        return np.array([])
    elif k > 0:
        return np.partition(a, -k)[-k:]
    else:
        return np.partition(a, -k)[:-k]

class WeightQuantization:
    def __init__(self, bits, clipping_p=0.0):
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
        self.p = clipping_p
    
    def avg_top(self, w):
        p = self.p
        wf = w.flatten()
        count = len(wf)
        q = max(1, int(p * count))

        return np.mean(np_topk(wf, q))

    def avg_bottom(self, w):
        p = self.p
        wf = w.flatten()
        count = len(wf)
        q = max(1, int(p * count))

        return -np.mean(np_topk(-wf, q))    
    
    def smooth_max(self, w):
        return self.avg_top(w).item()

    def smooth_min(self, w):
        return self.avg_bottom(w).item()

    def quantize_weight_(self, w, scale):
        return np.round(w * scale)
    
    def get_scale(self, min_w, max_w):
        neg_scale = +math.inf
        if abs(min_w) > 0: neg_scale = abs(self.q_max/min_w)
        pos_scale = +math.inf
        if abs(max_w) > 0: pos_scale = abs(self.q_max/max_w)

        if neg_scale is +math.inf and pos_scale is +math.inf:
            return 1.0
        
        return min(neg_scale, pos_scale)

    def quantize(self, weights):
        max_w = self.smooth_max(weights)
        min_w = self.smooth_min(weights)

        scale = self.get_scale(min_w, max_w)
        clipped_weights = np.clip(weights, min_w, max_w)

        return self.quantize_weight_(clipped_weights, scale)
    
    def calculate_threshold(self, weights):
        max_w = self.smooth_max(weights)
        min_w = self.smooth_min(weights)
        return np.round(self.get_scale(min_w, max_w))
    
class TensorQuantization:
    def __init__(self, bits, clipping_p=0.0):
        self.bits = bits
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
        self.p = clipping_p

    def avg_top(self, weight_tensor):
        p = self.p
        q = max(1, int(p * weight_tensor.numel()))
        return torch.mean(torch.topk(weight_tensor.flatten(), q)[0])

    def avg_bottom(self, weight_tensor):
        p = self.p
        q = max(1, int(p * weight_tensor.numel()))
        return -torch.mean(torch.topk(-weight_tensor.flatten(), q)[0])
    
    def get_scale(self, weight_tensor):
        max_weight = self.avg_top(weight_tensor).item()
        min_weight = self.avg_bottom(weight_tensor).item()

        neg_scale = +math.inf
        if abs(min_weight) > 0: neg_scale = abs(self.q_max/min_weight)
        pos_scale = +math.inf
        if abs(max_weight) > 0: pos_scale = abs(self.q_max/max_weight)

        if neg_scale is +math.inf and pos_scale is +math.inf:
            return 1.0
        
        return min(neg_scale, pos_scale)

    def get_clipped_weights(self, weight_tensor):
        max_weight = self.avg_top(weight_tensor).item()
        min_weight = self.avg_bottom(weight_tensor).item()

        return torch.clamp(weight_tensor, min_weight, max_weight)

    def quantize_tensor(self, weight_tensor):
        scale = self.get_scale(weight_tensor)
        clipped_weights = self.get_clipped_weights(weight_tensor)

        return torch.round(clipped_weights * scale) / scale

def quantize_cores(cores, bits, clipping_p=0.0):
    quantizer = WeightQuantization(bits, clipping_p)

    for core in cores:
        core.threshold = quantizer.calculate_threshold(core.core_matrix)
        print(core.threshold)
        print(core.core_matrix.max().item())
        print(core.core_matrix.min().item())
        core.core_matrix = quantizer.quantize(core.core_matrix)
        
    return cores

def calculate_core_thresholds(cores, bits, clipping_p=0.0):
    quantizer = WeightQuantization(bits, clipping_p)
    for core in cores:
        core.threshold = quantizer.calculate_threshold(core.core_matrix)
    return cores