from mimarsinan.transformations.weight_quantization import TensorQuantization
import numpy as np

class ChipQuantization:
    def __init__(self, bits):
        self.quantizer = TensorQuantization(bits)

    def verify_quantization(self, core):
        scale = self.quantizer.q_max / np.max(np.abs(core.core_matrix))
        assert np.allclose(
            core.core_matrix * scale, np.round(core.core_matrix * scale),
            atol=1e-3, rtol=1e-3)

        assert np.max(np.abs(np.round(core.core_matrix * scale))) == self.quantizer.q_max, \
            f"{np.max(np.abs(np.round(core.core_matrix * scale)))} > {self.quantizer.q_max}"


    def unscaled_quantize(self, cores):
        for core in cores:
            core.core_matrix = self.quantizer.quantize(core.core_matrix)
            
    def quantize(self, cores):
        for core in cores:
            self.verify_quantization(core)
            threshold_scale = np.max(np.abs(core.core_matrix))

            print(core.threshold, threshold_scale)
            activation_scale = core.threshold * self.quantizer.q_max * 0.95
            adjusted_threshold = activation_scale / threshold_scale
            mixed_threshold = 0.9 * activation_scale + 0.1 * adjusted_threshold
            core.threshold = mixed_threshold * 0.95
            print(core.threshold)

            core.core_matrix *= self.quantizer.q_max / threshold_scale
            core.core_matrix = np.round(core.core_matrix)