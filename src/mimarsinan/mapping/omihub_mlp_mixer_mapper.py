from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.omihub_mlp_mixer import *

def omihub_mlp_mixer_to_chip(
    omihub_mlp_mixer_model: MLPMixer,
    leak = 0.0,
    quantize = False,
    weight_type = float):
    model = omihub_mlp_mixer_model.cpu()

    neurons_per_core = 64
    axons_per_core = 256

    in_channels = model.img_dim_c
    in_height = model.img_dim_h
    in_width = model.img_dim_w

    input_size = in_channels * in_height * in_width
    
    kernel_features = model.patch_emb[0].weight.size(0)
    kernel_h = model.patch_emb[0].weight.size(2)
    kernel_w = model.patch_emb[0].weight.size(3)

    patch_rows = in_height // kernel_h
    patch_cols = in_width // kernel_w
    num_patches = patch_rows * patch_cols

    np_weights = model.patch_emb[0].weight.data.numpy()
    cores_list = []
    connections_list = []
    
    mapping = Mapping()
    mapping.neurons_per_core = neurons_per_core
    mapping.axons_per_core = axons_per_core
    mapping.cores = cores_list
    mapping.connections = connections_list

    # prepare input sources
    input_sources = []
    for i in range(in_channels):
        input_sources.append([])
        for j in range(in_height):
            input_sources[i].append([])
            for k in range(in_width):
                input_idx = i * in_height * in_width + j * in_width + k
                input_sources[i][j].append(
                    SpikeSource(0, input_idx, True, False))
    layer_sources = np.array(input_sources)

    fc_layers = []
    fc_layers.append(PatchEmbeddingLayer()) 
    for i in range(model.num_layers):
        fc_layers.append(
            model.mixer_layers[i].mlp1.fc1)
        fc_layers.append(
            model.mixer_layers[i].mlp1.fc2)
        fc_layers.append(
            AddOp(len(fc_layers), len(fc_layers)-2))
        fc_layers.append(
            model.mixer_layers[i].mlp2.fc1)
        fc_layers.append(
            model.mixer_layers[i].mlp2.fc2)
        fc_layers.append(
            AddOp(len(fc_layers), len(fc_layers)-2))
    
    fc_layers.append(AvgPoolLayer())
    fc_layers.append(model.clf)
        
    layer_sources_list = []
    for layer in fc_layers:
        layer_sources_list.append(layer_sources)
        if isinstance(layer, nn.Conv1d):
            layer_sources = map_conv1d(mapping, layer_sources, layer, quantize)
        elif isinstance(layer, nn.Linear):
            layer_sources = map_linear(mapping, layer_sources, layer, quantize)
        elif isinstance(layer, AvgPoolLayer):
            layer_sources = map_avg_pool(mapping, layer_sources, quantize)
        elif isinstance(layer, AddOp):
            layer_sources = map_add_op(mapping, 
                layer_sources_list[layer.source_idx_a], 
                layer_sources_list[layer.source_idx_b])
        elif isinstance(layer, PatchEmbeddingLayer):
            layer_sources = map_patch_embedding(mapping, layer_sources, model.patch_emb[0], quantize)
            
    output_list = []
    for source in layer_sources.flatten():
        output_list.append(source)
    
    chip = ChipModel(axons_per_core, neurons_per_core, len(mapping.cores), 
        input_size, len(output_list), leak, mapping.connections, output_list, mapping.cores, weight_type)

    chip.load_from_json(chip.get_chip_json()) # sanity check
    return chip    

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())