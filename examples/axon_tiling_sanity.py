import sys
sys.path.append('./src')

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.mapping.mapping_utils import InputMapper, ModelRepresentation, PerceptronMapper, SoftCoreMapping
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

def main():
    # Value correctness test for tiling:
    # 1 accumulator, 2 tiles (pos/neg split).
    # Weights:
    #  tile 0: [2, -2] -> input [3, 4] -> 2*3 + (-2)*4 = 6 - 8 = -2
    #  tile 1: [-1, 1] -> input [5, 6] -> (-1)*5 + 1*6 = -5 + 6 = 1
    # Total = -2 + 1 = -1
    # Bias = 10
    # Final = 9 (before activation)

    max_axons = 6 # tile_size=6. inputs=4. Fits in 1 tile? No wait.
    # We want to FORCE tiling.
    # Input size must be > max_axons.
    # If max_axons=6, input must be >= 7.
    # Let's set in_features = 10.
    # tile_size = 6. Tiles: [0:6], [6:10]. (2 tiles)
    # Accumulator inputs = 2 tiles * 2 (pos/neg) + 1 (bias) = 5.
    # 5 <= max_axons=6. Fits!

    in_features = 10
    out_features = 1
    
    p = Perceptron(output_channels=out_features, input_features=in_features)
    with torch.no_grad():
        # Weights for 10 inputs.
        # Tile 0 (0-5): [1, -1, 1, -1, 0, 0] -> inputs 1..6 -> 1-2+3-4+0+0 = -2
        # Tile 1 (6-9): [0, 0, 0, 2] -> inputs 7..10 -> 0+0+0+20 = 20
        # Bias = 5
        # Total = -2 + 20 + 5 = 23.
        w = torch.zeros(1, 10)
        w[0, 0] = 1.0; w[0, 1] = -1.0; w[0, 2] = 1.0; w[0, 3] = -1.0
        w[0, 9] = 2.0
        p.layer.weight.data = w
        p.layer.bias.data = torch.tensor([5.0])

    p.set_parameter_scale(1.0)
    p.set_input_activation_scale(1.0)
    p.set_activation_scale(1.0)

    inp = InputMapper((in_features,))
    out = PerceptronMapper(inp, p)
    mr = ModelRepresentation(out)

    m = SoftCoreMapping(q_max=127, firing_mode='Default', max_axons=6, max_neurons=128, allow_axon_tiling=True)
    m.map(mr)

    # Input spikes: 1..10
    input_spikes = { (-2, i): float(i+1) for i in range(10) }
    # Check:
    # T0: 1*1 + (-1)*2 + 1*3 + (-1)*4 + 0 + 0 = 1 - 2 + 3 - 4 = -2.
    # T1: 0 + 0 + 0 + 2*10 = 20.
    # Bias: 5.
    # Sum: -2 + 20 + 5 = 23.

    core_outputs = {}

    print("Executing cores...")
    for core in m.cores:
        activation = 0.0
        for i, source in enumerate(core.axon_sources):
            if source.is_input_:
                val = input_spikes.get((source.core_, source.neuron_), 0.0)
            elif source.is_off_:
                val = 0.0
            elif getattr(source, 'is_always_on_', False):
                 val = 1.0
            else:
                val = core_outputs.get((source.core_, source.neuron_), 0.0)
            
            # SoftCore weights are stored as is (unscaled float in this test)
            w = core.core_matrix[i, 0]
            activation += val * w
        
        # Simulating standard ReLU neuron for all cores:
        output = max(0.0, activation)
        core_outputs[(core.id, 0)] = output
        print(f"Core {core.id} ({getattr(core, 'psum_role', 'standard')}): act={activation}, out={output}")

    last_core = m.cores[-1]
    final_val = core_outputs[(last_core.id, 0)]
    print(f"Final output: {final_val}")
    assert abs(final_val - 23.0) < 1e-5, f"Expected 23.0, got {final_val}"
    print("OK")

if __name__ == "__main__":
    main()
