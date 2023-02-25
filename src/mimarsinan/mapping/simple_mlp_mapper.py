from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.simple_mlp import *

def simple_mlp_to_chip(
    simple_mlp_model: SimpleMLP,
    leak = 0.0,
    quantize = False,
    weight_type = float):
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
        
        cores_list.append(generate_core_weights_legacy(
            neurons_per_core, axons_per_core, 
            layer.weight.cpu(), layer.weight.size(0),
            1.0, biases[i], quantize=quantize))
        
        connections_list.append(generate_core_connection_info(
            axons_per_core, layer.weight.size(1), max(i - 1, 0), (i == 0)
        ))

    output_list: list[SpikeSource] = []
    for idx in range(outputs):
        output_list.append(SpikeSource(len(cores_list) - 1, idx))

    chip = ChipModel(axons_per_core, neurons_per_core, len(cores_list), 
        inputs, outputs, leak, connections_list, output_list, cores_list, weight_type)

    # chip.load_from_json(chip.get_chip_json()) # sanity check
    
    return chip    

def simple_mlp_to_chip_v2(
simple_mlp_model: SimpleMLP,
    leak = 0.0,
    quantize = False,
    weight_type = float):
    model = simple_mlp_model.cpu()

    neurons_per_core = max([l.weight.size(0) for l in model.layers])
    axons_per_core = max([l.weight.size(1) for l in model.layers])
    inputs = model.layers[0].weight.size(1)
    outputs = model.layers[-1].weight.size(0)

    biases = [None] * len(model.layers)

    cores_list: list[Core] = []
    connections_list: list[Connection] = []
    for i, layer in enumerate(model.layers):
        weights = layer.weight.detach().numpy()

        threshold = 1.0
        if (quantize):
            weights = quantize_weight_tensor(weights)
            threshold = calculate_threshold(weights)

        if(layer.bias is not None):
            biases[i] = layer.bias.cpu().tolist()
        
        cores_list.append(generate_core_weights(
            neurons_per_core, axons_per_core, 
            weights, layer.weight.size(0),
            threshold, biases[i]))
        
        connections_list.append(generate_core_connection_info(
            axons_per_core, layer.weight.size(1), max(i - 1, 0), (i == 0)
        ))

    output_list: list[SpikeSource] = []
    for idx in range(outputs):
        output_list.append(SpikeSource(len(cores_list) - 1, idx))

    chip = ChipModel(axons_per_core, neurons_per_core, len(cores_list), 
        inputs, outputs, leak, connections_list, output_list, cores_list, weight_type)

    # chip.load_from_json(chip.get_chip_json()) # sanity check
    
    return chip    

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())