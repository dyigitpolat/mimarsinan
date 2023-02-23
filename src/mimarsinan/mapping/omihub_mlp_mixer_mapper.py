from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.models.omihub_mlp_mixer import *

class Mapping:
    def __init__(self):
        self.cores = []
        self.connections = []
        self.quantize = False

        self.neurons_per_core = 64
        self.axons_per_core = 64
        pass

    def map_fc(self, 
        input_tensor_sources,  # numpy array of SpikeSource [num_patches, kernel_features]
        output_shape, # [hidden_s, kernel_features] 
        fc_weights): # numpy array of weights [hidden_s, num_patches]
        assert input_tensor_sources.shape[-1] == output_shape[-1]
        new_cores_count = output_shape[-1]
        out_neurons_count = output_shape[-2]
        input_axons_count = input_tensor_sources.shape[-2]

        core_matrix = np.zeros([self.axons_per_core, self.neurons_per_core])
        core_matrix[0:input_axons_count, 0:out_neurons_count] = fc_weights.transpose()

        for i in range(new_cores_count): #out_cores
            self.cores.append(
                generate_core_weights(
                    self.neurons_per_core, self.axons_per_core, core_matrix.transpose(),
                    self.neurons_per_core, 0.0, quantize=self.quantize))

            spike_sources = []
            for j in range(input_axons_count):
                source_core = input_tensor_sources[j, i].core_
                source_neuron = input_tensor_sources[j, i].neuron_
                spike_sources.append(SpikeSource(source_core, source_neuron))
            
            for j in range(self.axons_per_core - input_axons_count):
                spike_sources.append(SpikeSource(0, 0, is_input=False, is_off=True))
            
            self.connections.append(Connection(spike_sources))

def omihub_mlp_mixer_to_chip(
    omihub_mlp_mixer_model: MLPMixer,
    leak = 0.0,
    quantize = False,
    weight_type = float):
    model = omihub_mlp_mixer_model.cpu()

    neurons_per_core = 64
    axons_per_core = 64

    in_channels = model.img_dim_c
    in_height = model.img_dim_h
    in_width = model.img_dim_w

    input_size = in_channels * in_height * in_width
    output_size = model.num_classes
    
    kernel_features = model.patch_emb[0].weight.size(0)
    kernel_h = model.patch_emb[0].weight.size(2)
    kernel_w = model.patch_emb[0].weight.size(3)

    patch_rows = in_height // kernel_h
    patch_cols = in_width // kernel_w
    num_patches = patch_rows * patch_cols

    patch_size = in_channels * kernel_h * kernel_w

    # patch_emb, in_ch = in_channels, out_ch = kernel_featuresm

    np_weights = model.patch_emb[0].weight.data.numpy()
    cores_list = []
    connections_list = []
    for j in range(num_patches):
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
                neurons_per_core, 0.0, quantize=quantize))
        
        is_off = lambda i: core_source_neurons[i] == -1
        connections_list.append(
            Connection([
                SpikeSource(0, core_source_neurons[i], not is_off(i), is_off(i)) 
                for i in range(axons_per_core)
            ]))

    prev_core_count = len(cores_list) - num_patches
    print("prev_core_count", prev_core_count)

    input_sources = []
    for i in range(num_patches):
        input_sources.append([])
        for j in range(kernel_features):
            input_sources[i].append(
                SpikeSource(i, j, True, False))
    input_sources = np.array(input_sources)
    layer_output_shape = np.array([model.hidden_s, kernel_features])
    
    mapping = Mapping()
    mapping.neurons_per_core = neurons_per_core
    mapping.axons_per_core = axons_per_core
    mapping.cores = cores_list
    mapping.connections = connections_list
    mapping.map_fc(
        input_sources, 
        layer_output_shape, 
        model.mixer_layers[0].mlp1.fc1.weight.data[:, :, 0].numpy())
        
    output_list = []
    for i in range(kernel_features):
        for j in range(model.hidden_s):
            output_list.append(SpikeSource(num_patches+i, j))
    
    chip = ChipModel(axons_per_core, neurons_per_core, len(mapping.cores), 
        input_size, output_size, leak, mapping.connections, output_list, mapping.cores, weight_type)

    # chip.load_from_json(chip.get_chip_json()) # sanity check
    return chip    

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())