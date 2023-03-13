from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.mapping.weight_quantization import *
from mimarsinan.models.simple_mlp import *   
from mimarsinan.models.core_flow import *   

def get_simple_mlp_repr(input_shape, model):
    input = InputMapper(input_shape)
    prev = input
    for layer in model.layers:
        prev = LinearMapper(prev, layer)
    
    return ModelRepresentation(prev)

def simple_mlp_to_chip(
simple_mlp_model: SimpleMLP,
    leak = 0.0,
    quantize = False,
    weight_type = float):
    model = simple_mlp_model.cpu()
    neurons_per_core = 256
    axons_per_core = 785

    input_size = model.layers[0].weight.size(1)
    input_shape = (1, input_size)
    model_repr = get_simple_mlp_repr(input_shape, model)
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)

    if quantize:
        quantize_cores(soft_core_mapping.cores, bits=4)

    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)

    chip = hard_cores_to_chip(
        input_size, hard_core_mapping, axons_per_core, neurons_per_core,
        leak, weight_type)     

    return chip

def simple_mlp_to_core_flow(input_shape, simple_mlp_model, max_axons, max_neurons):
    model_repr = get_simple_mlp_repr(input_shape, simple_mlp_model)
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)

    return CoreFlow(input_shape, soft_core_mapping)


def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())