from mimarsinan.transformations.weight_quantization import TensorQuantization
import numpy as np

class ChipQuantization:
    def __init__(self, bits):
        self.quantizer = TensorQuantization(bits)

    def unscaled_quantize(self, cores):
        for core in cores:
            core.core_matrix = self.quantizer.quantize(core.core_matrix)
            
    def quantize(self, cores):
        for core in cores:
            threshold_scale = np.max(np.abs(core.core_matrix))

            core.core_matrix /= threshold_scale
            core.core_matrix = self.quantizer.scaled_quantize(core.core_matrix)
            
            assert np.max(np.abs(core.core_matrix)) == self.quantizer.q_max, \
                f"{np.max(np.abs(core.core_matrix))} > {self.quantizer.q_max}"
            
            core.threshold /= threshold_scale
            
