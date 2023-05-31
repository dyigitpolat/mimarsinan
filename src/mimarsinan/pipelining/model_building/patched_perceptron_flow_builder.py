from mimarsinan.models.perceptron_flow import PatchedPerceptronFlow

import math
class PatchedPerceptronFlowBuilder:
    def __init__(self, 
                 max_axons, max_neurons,
                 input_shape, output_shape,
                 model_complexity):
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_complexity = model_complexity

    def build(self):
        perceptron_flow = None

        print(f"Input shape: {self.input_shape}")
        for i in range(2, int(math.sqrt(self.max_axons)) + 1):
            patch_rows = max(self.input_shape[-2] // i, 1)
            patch_cols = max(self.input_shape[-1] // i, 1)

            print(f"Trying patch dimensions {i}: {patch_rows} x {patch_cols}")
            try:
                perceptron_flow = PatchedPerceptronFlow(
                    self.input_shape, self.output_shape,
                    self.max_axons - 1, self.max_neurons,
                    patch_cols, 
                    patch_rows, fc_depth=self.model_complexity)
                break
            except: 
                pass
        
        print(f"Patch dimensions {patch_rows} x {patch_cols}")
        assert perceptron_flow is not None, \
            "Could not find a valid patch size for the given input shape and max axons."
    
        return perceptron_flow