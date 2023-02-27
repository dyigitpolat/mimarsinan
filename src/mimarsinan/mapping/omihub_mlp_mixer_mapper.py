from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.omihub_mlp_mixer import *

class AvgPoolLayer:
    pass
class AddOp:
    def __init__(self, source_idx_a, source_idx_b):
        self.source_idx_a = source_idx_a
        self.source_idx_b = source_idx_b

class Mapping:
    def __init__(self):
        self.cores = []
        self.connections = []

        self.neurons_per_core = 64
        self.axons_per_core = 64
        pass

    def map_fc(self, 
        input_tensor_sources,  # 
        output_shape, # 
        fc_weights, 
        quantize,
        threshold = 1.0): # 

        w_rows = fc_weights.shape[-2]
        w_cols = fc_weights.shape[-1]
        x_rows = input_tensor_sources.shape[-2]
        x_cols = input_tensor_sources.shape[-1]
        o_rows = output_shape[-2]
        o_cols = output_shape[-1]

        assert o_rows == w_rows
        assert o_cols == x_cols
        assert x_rows == w_cols

        new_cores_count = o_cols
        out_neurons_count = o_rows
        input_axons_count = x_rows

        core_matrix = np.zeros([self.axons_per_core, self.neurons_per_core])
        core_matrix[0:input_axons_count, 0:out_neurons_count] = fc_weights.transpose()

        for i in range(new_cores_count): 
            if (quantize):
                threshold = calculate_threshold(core_matrix)
                core_matrix = quantize_weight_tensor(core_matrix)

            self.cores.append(
                generate_core_weights(
                    self.neurons_per_core, self.axons_per_core, core_matrix.transpose(),
                    self.neurons_per_core, threshold))

            spike_sources = []
            for j in range(input_axons_count):
                source_core = input_tensor_sources[j, i].core_
                source_neuron = input_tensor_sources[j, i].neuron_
                spike_sources.append(SpikeSource(source_core, source_neuron))
            
            for j in range(self.axons_per_core - input_axons_count):
                spike_sources.append(SpikeSource(0, 0, is_input=False, is_off=True))
            
            self.connections.append(Connection(spike_sources))

        layer_sources = []
        core_offset = len(self.cores) - new_cores_count
        for neuron_idx in range(o_rows):
            layer_sources.append(
                [SpikeSource(core_offset + core_idx, neuron_idx) for core_idx in range(o_cols)])
        
        return np.array(layer_sources)

def map_mm(mapping, layer_sources, layer_weights, quantize, threshold = 1.0):
    layer_output_shape = np.array([layer_weights.shape[-2], layer_sources.shape[-1]])
    return mapping.map_fc(
        layer_sources, 
        layer_output_shape, 
        layer_weights,
        quantize,
        threshold)   
    
def map_conv1d(mapping, layer_sources, layer, quantize):
    layer_weights = layer.weight.data.numpy().squeeze()
    return map_mm(mapping, layer_sources, layer_weights, quantize)
      
def map_linear(mapping, layer_sources, layer, quantize):
    layer_sources = layer_sources.transpose()
    layer_sources = map_conv1d(mapping, layer_sources, layer, quantize)
    layer_sources = layer_sources.transpose()
    return layer_sources

def map_avg_pool(mapping, layer_sources, quantize):
    factor =  1.0 / layer_sources.shape[0]
    threshold = 1.0

    if quantize:
        factor = 1.0
        threshold = layer_sources.shape[0]

    weights = np.ones([1, layer_sources.shape[0]]) * factor
    return map_mm(mapping, layer_sources, weights, False, threshold)

def map_add_op(mapping, layer_sources_a, layer_sources_b):
    assert layer_sources_a.shape == layer_sources_b.shape

    x_rows = layer_sources_a.shape[-2]
    layer_sources = np.concatenate([layer_sources_a, layer_sources_b], axis=0)
    weights = np.concatenate([np.eye(x_rows), np.eye(x_rows)], axis=0).transpose()
    return map_mm(mapping, layer_sources, weights, quantize=False)


    
    
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
                    core_source_neurons[axon_id] = img_row * in_width + img_col
                    
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