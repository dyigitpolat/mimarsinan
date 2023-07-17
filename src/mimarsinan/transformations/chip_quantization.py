from mimarsinan.transformations.weight_quantization import TensorQuantization
from mimarsinan.transformations.weight_clipping import SoftTensorClipping
import numpy as np

class ChipQuantization:
    def __init__(self, bits):
        self.quantizer = TensorQuantization(bits)
        self.target_minimum_threshold = self.quantizer.q_max
    
    def calculate_core_thresholds(self, cores):
        for core in cores:
            core.threshold = self.quantizer.q_max
    
    def quantize(self, cores):
        self.calculate_core_thresholds(cores)

        for core in cores:
            core.core_matrix = self.quantizer.scaled_quantize(core.core_matrix)
            assert np.max(np.abs(core.core_matrix)) <= self.quantizer.q_max, \
                f"{np.max(np.abs(core.core_matrix))} > {self.quantizer.q_max}"

        return 1.0