from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.mapping.weight_quantization import *
from mimarsinan.models.omihub_mlp_mixer import *
    
def polat_mlp_mixer_to_chip(
    omihub_mlp_mixer_model: MLPMixer,
    leak = 0.0,
    quantize = False,
    weight_type = float):
    model = omihub_mlp_mixer_model.cpu()

    neurons_per_core = 256
    axons_per_core = 256

    in_channels = model.img_dim_c
    in_height = model.img_dim_h
    in_width = model.img_dim_w

    input_size = in_channels * in_height * in_width
    
    mapping = Mapping()
    mapping.max_neurons = neurons_per_core
    mapping.max_axons = axons_per_core

    input_shape = (in_channels, in_height, in_width)
    layer_sources = prepare_input_sources(input_shape)

    fc_layers = []
    fc_layers.append(PatchEmbeddingLayer()) # input
    for i in range(model.num_layers):
        fc_layers.append(
            model.mixer_layers[i].mlp1.fc1)
        fc_layers.append(
            model.mixer_layers[i].mlp1.fc2)
        fc_layers.append(
            model.mixer_layers[i].mlp1.ln)
        
        fc_layers.append(
            model.mixer_layers[i].mlp2.fc1)
        fc_layers.append(
            model.mixer_layers[i].mlp2.fc2)
        fc_layers.append(
            model.mixer_layers[i].mlp2.ln)
    
    fc_layers.append(AvgPoolLayer())
    fc_layers.append(model.clf)

    fuse_layers(fc_layers)
        
    layer_sources_list = []
    for layer in fc_layers:
        layer_sources_list.append(layer_sources)

        if isinstance(layer, nn.Conv1d):
            layer_sources = map_conv1d(mapping, layer_sources, layer)
        elif isinstance(layer, nn.Linear):
            layer_sources = map_linear(mapping, layer_sources, layer)
        elif isinstance(layer, AvgPoolLayer):
            layer_sources = map_avg_pool(mapping, layer_sources)
        elif isinstance(layer, AddOp):
            layer_sources = map_add_op(mapping, 
                layer_sources_list[layer.source_idx_a], 
                layer_sources_list[layer.source_idx_b])
        elif isinstance(layer, PatchEmbeddingLayer):
            layer_sources = map_patch_embedding(mapping, layer_sources, model.patch_emb[0])
            
    output_list = []
    for source in layer_sources.flatten():
        output_list.append(source)
    
    return to_chip(
        input_size, output_list, 
        mapping, mapping.max_axons, mapping.max_neurons,
        leak, quantize, weight_type)     

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())