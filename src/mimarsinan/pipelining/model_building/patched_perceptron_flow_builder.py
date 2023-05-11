from mimarsinan.models.perceptron_flow import PatchedPerceptronFlow

class PatchedPerceptronFlowBuilder:
    def __init__(self, 
                 max_axons, max_neurons,
                 input_shape, output_shape):
        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.input_shape = input_shape
        self.output_shape = output_shape

    def build(self):
        patch_rows = max(self.input_shape[-2] // 4, 1)
        patch_cols = max(self.input_shape[-1] // 4, 1)

        perceptron_flow = PatchedPerceptronFlow(
            self.input_shape, self.output_shape,
            self.max_axons - 1, self.max_neurons,
            patch_rows, 
            patch_cols, fc_depth=2)
    
        return perceptron_flow