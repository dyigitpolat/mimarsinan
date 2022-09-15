from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.simple_mlp import *

def simple_mlp_to_chip(
    simple_mlp_model: SimpleMLP):
    model = simple_mlp_model.cpu()

    neurons_per_core = max([l.weight.size(0) for l in model.layers])
    axons_per_core = max([l.weight.size(1) for l in model.layers])
    inputs = model.layers[0].weight.size(1)
    outputs = model.layers[-1].weight.size(0)

    biases = [None] * len(model.layers)

    cores_list: list[Core] = []
    connections_list: list[Connection] = []
    for i, layer in enumerate(model.layers):
        if(layer.bias is not None):
            biases[i] = layer.bias.cpu().tolist()
        
        cores_list.append(generate_core_weights(
            neurons_per_core, axons_per_core, 
            layer.weight.cpu(), layer.weight.size(0),
            1.0, biases[i] ))
        
        connections_list.append(generate_core_connection_info(
            axons_per_core, layer.weight.size(1), max(i - 1, 0), (i == 0)
        ))

    output_list: list[SpikeSource] = []
    for idx in range(outputs):
        output_list.append(SpikeSource(len(cores_list) - 1, idx))

    chip = ChipModel(axons_per_core, neurons_per_core, len(cores_list), 
        inputs, outputs, 0.075, connections_list, output_list, cores_list)
    
    return chip    
