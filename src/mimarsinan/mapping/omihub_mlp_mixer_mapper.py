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
    for j in range(num_patches):
        threshold = 1.0
        if (quantize):
            threshold = calculate_threshold(core_matrix)
            core_matrix = quantize_weight_tensor(core_matrix)

        core_matrix = np.zeros([axons_per_core, neurons_per_core])
        core_source_neurons = -np.ones([axons_per_core], dtype=np.int32) # -1 means no connection
        
        for ch_ in range(in_channels):
            for h_ in range(kernel_h):
                for w_ in range(kernel_w):
                    axon_id = ch_ * kernel_h * kernel_w + h_ * kernel_w + w_
                    weights = np_weights[:, ch_, h_, w_].flatten()
                    assert len(weights) == kernel_features
                    core_matrix[axon_id, 0:kernel_features] = weights
                    
                    img_row = kernel_h * (j // patch_rows) + h_
                    img_col = kernel_w * (j % patch_rows) + w_
                    core_source_neurons[axon_id] = \
                        ch_ * in_height * in_width + img_row * in_width + img_col
                    
        cores_list.append(
            generate_core_weights(
                neurons_per_core, axons_per_core, core_matrix.transpose(), 
                neurons_per_core, threshold))
        
        is_off = lambda i: core_source_neurons[i] == -1
        connections_list.append(
            Connection([
                SpikeSource(0, core_source_neurons[i], not is_off(i), is_off(i)) 
                for i in range(axons_per_core)
            ]))

    prev_core_count = len(cores_list) - num_patches
    print("prev_core_count", prev_core_count)
    
    mapping = Mapping()
    mapping.neurons_per_core = neurons_per_core
    mapping.axons_per_core = axons_per_core
    mapping.cores = cores_list
    mapping.connections = connections_list

    # prepare input sources
    input_sources = []
    for i in range(num_patches):
        input_sources.append([])
        for j in range(kernel_features):
            input_sources[i].append(
                SpikeSource(i, j, True, False))
    layer_sources = np.array(input_sources)

    fc_layers = []
    fc_layers.append(0) # input
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
            
    print(layer_sources.shape)
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