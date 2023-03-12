from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.mapping.weight_quantization import *
from mimarsinan.models.simple_mlp import *   

def simple_mlp_to_chip(
simple_mlp_model: SimpleMLP,
    leak = 0.0,
    quantize = False,
    weight_type = float):
    model = simple_mlp_model.cpu()

    mapping = Mapping()
    mapping.max_neurons = 256
    mapping.max_axons = 784

    input_size = model.layers[0].weight.size(1)

    input_shape = (1, input_size)
    input = InputMapper(input_shape)
    prev = input
    for layer in model.layers:
        prev = LinearMapper(prev, layer)
            
    output_list = []
    for source in prev.map(mapping).flatten():
        output_list.append(source)

    return to_chip(
        input_size, output_list, 
        mapping, mapping.max_axons, mapping.max_neurons,
        leak, quantize, weight_type)

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())