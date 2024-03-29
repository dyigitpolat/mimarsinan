from mimarsinan.transformations.transformation_utils import *

import torch
import numpy as np

class SoftTensorClipping:
    def __init__(self, clipping_rate=0.01):
        self.rate = clipping_rate

    def avg_top(self, weight_tensor):
        if isinstance(weight_tensor, np.ndarray):
            return transform_np_array(weight_tensor, self.avg_top)
        
        p = self.rate
        wf = weight_tensor.flatten()
        wf = wf[torch.abs(wf) > 0]
        if wf.numel() == 0:
            return torch.tensor(0.0)
        q = max(1, int(p * wf.numel()))
        return torch.mean(torch.topk(wf, q)[0])

    def avg_bottom(self, weight_tensor):
        if isinstance(weight_tensor, np.ndarray):
            return transform_np_array(weight_tensor, self.avg_bottom)
        
        p = self.rate
        wf = weight_tensor.flatten()
        wf = wf[torch.abs(wf) > 0]
        if wf.numel() == 0:
            return torch.tensor(0.0)
        q = max(1, int(p * wf.numel()))
        return -torch.mean(torch.topk(-wf, q)[0])

    def get_clipped_weights(self, weight_tensor):
        if isinstance(weight_tensor, np.ndarray):
            return transform_np_array(weight_tensor, self.get_clipped_weights)
        
        max_weight = self.avg_top(weight_tensor).item()
        min_weight = self.avg_bottom(weight_tensor).item()

        return torch.clamp(weight_tensor, min_weight, max_weight)

def clip_core_weights(cores, clipping_rate=0.01):
    clipper = SoftTensorClipping(clipping_rate)
    for core in cores:
        core.core_matrix = clipper.get_clipped_weights(core.core_matrix)

def get_clipped_w_b(w, b, clipping_rate):
    clipper = SoftTensorClipping(clipping_rate)
    w_top = clipper.avg_top(w).item()
    w_bottom = clipper.avg_bottom(w).item()
    
    return torch.clamp(w, w_bottom, w_top), torch.clamp(b, w_bottom, w_top)
