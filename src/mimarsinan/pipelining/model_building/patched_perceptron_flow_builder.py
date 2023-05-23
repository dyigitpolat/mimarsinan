from mimarsinan.models.perceptron_flow import PatchedPerceptronFlow

import math
class PatchedPerceptronFlowBuilder:
    def __init__(self, 
                 max_axons, max_neurons,
                 input_shape, output_shape):
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self):
        patch_cols = -1
        patch_rows = -1
        for i in range(1, int(math.sqrt(self.max_axons)) + 1):
            try:
                patch_rows = max(self.input_shape[-2] // i, 1)
                patch_cols = max(self.input_shape[-1] // i, 1)

                perceptron_flow = PatchedPerceptronFlow(
                    self.input_shape, self.output_shape,
                    self.max_axons - 1, self.max_neurons,
                    patch_rows, 
                    patch_cols, fc_depth=2)
                break
            except: 
                continue
        
        print(f"Patch dimensions {patch_cols} x {patch_cols}")
        assert patch_cols != -1 or patch_rows != -1, "Could not find a valid patch size for the given input shape and max axons."

    
        return perceptron_flow