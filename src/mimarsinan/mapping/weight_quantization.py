import numpy as np

def np_topk(a, k):
    if k == 0:
        return np.array([])
    elif k > 0:
        return np.partition(a, -k)[-k:]
    else:
        return np.partition(a, -k)[:-k]

class WeightQuantization:
    def __init__(self, bits):
        self.q_min = -( 2 ** (bits - 1) )
        self.q_max = ( 2 ** (bits - 1) ) - 1
    
    def avg_top(self, w, p):
        wf = w.flatten()
        count = len(wf)
        q = max(1, int(p * count))
        return np.mean(np_topk(wf, q))

    def avg_bottom(self, w, p):
        wf = w.flatten()
        count = len(wf)
        q = max(1, int(p * count))
        return -np.mean(np_topk(-wf, q))    
    
    def smooth_max(self, w):
        return self.avg_top(w, 0.01).item()

    def smooth_min(self, w):
        return self.avg_bottom(w, 0.01).item()

    def quantize_weight_(self, w, scale):
        return np.round(w * scale)
    
    def get_scale(self, min_w, max_w):
        neg_scale = 1.0
        if abs(min_w) > 0: neg_scale = abs(self.q_max/min_w)
        pos_scale = 1.0
        if abs(max_w) > 0: pos_scale = abs(self.q_max/max_w)
        
        return min(neg_scale, pos_scale)

    def quantize(self, weights):
        max_w = self.smooth_max(weights)
        min_w = self.smooth_min(weights)

        scale = self.get_scale(min_w, max_w)
        clipped_weights = np.clip(weights, min_w, max_w)

        return self.quantize_weight_(clipped_weights, scale)

    def quantize_without_scaling(self, weights):
        max_w = self.smooth_max(weights)
        min_w = self.smooth_min(weights)

        scale = self.get_scale(min_w, max_w)
        clipped_weights = np.clip(clipped_weights, min_w, max_w)

        return self.quantize_weight_(clipped_weights, scale) / scale
    
    def calculate_threshold(self, weights):
        max_w = self.smooth_max(weights)
        min_w = self.smooth_min(weights)
        return self.get_scale(min_w, max_w)

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