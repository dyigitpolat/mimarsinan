from mimarsinan.transformations.weight_quantization import TensorQuantization
import numpy as np

class ChipQuantization:
    def __init__(self, bits):
        self.quantizer = TensorQuantization(bits)

    def verify_quantization(self, core):
        scale = core.parameter_scale.cpu().numpy()
        assert np.allclose(
            core.core_matrix * scale, np.round(core.core_matrix * scale),
            atol=1e-3, rtol=1e-3)

        assert np.max(np.abs(np.round(core.core_matrix * scale))) <= self.quantizer.q_max, \
            f"{np.max(np.abs(np.round(core.core_matrix * scale)))} > {self.quantizer.q_max}"


    def unscaled_quantize(self, cores):
        for core in cores:
            core.core_matrix = self.quantizer.quantize(core.core_matrix)
            
    def quantize(self, cores):
        for core in cores:
            self.verify_quantization(core)

            print("act scale = ", core.activation_scale.item())
            print("param scale = ", core.parameter_scale.item())
            core.threshold = core.parameter_scale.item() * 0.9
            print(core.threshold)

            core.core_matrix *= core.parameter_scale.item()
            core.core_matrix = np.round(core.core_matrix)