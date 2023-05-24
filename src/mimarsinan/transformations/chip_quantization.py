from mimarsinan.transformations.weight_quantization import TensorQuantization
from mimarsinan.transformations.weight_clipping import SoftTensorClipping
import numpy as np

class ChipQuantization:
    def __init__(self, bits):
        self.quantizer = TensorQuantization(bits)
        self.target_minimum_threshold = self.quantizer.q_max

    def calculate_threshold(self, weights):
        max_w = np.max(weights)
        min_w = np.min(weights)

        return self.quantizer.get_scale(min_w, max_w)
    
    def scale_thresholds(self, cores):
        min_threshold = min([core.threshold for core in cores])
        target = self.target_minimum_threshold

        if(min_threshold < target):
            scale = target / min_threshold
        else:
            scale = 1.0

        for core in cores:
            core.threshold *= scale

        return scale
    
    def calculate_core_thresholds(self, cores):
        for core in cores:
            core.threshold = self.quantizer.get_scale(core.core_matrix)
    
    def quantize(self, cores):
        self.calculate_core_thresholds(cores)
        scale = self.scale_thresholds(cores)
            
        for core in cores:
            core.threshold = round(core.threshold * (1.0 - 1.0 / self.quantizer.q_max))
            core.core_matrix = self.quantizer.scaled_quantize(core.core_matrix)

        return scale