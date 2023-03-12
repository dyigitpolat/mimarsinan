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
    max_cores = 43

    in_channels = model.img_dim_c
    in_height = model.img_dim_h
    in_width = model.img_dim_w

    input_size = in_channels * in_height * in_width
    
    mapping = Mapping()
    mapping.max_neurons = neurons_per_core
    mapping.max_axons = axons_per_core

    input_shape = (in_channels, in_height, in_width)

    input = InputMapper(input_shape)
    patch_emb = PatchEmbeddingMapper(input, model.patch_emb[0])
    prev = patch_emb
    for i in range(model.num_layers):
        mixer_m1_fc1 = Conv1DMapper(prev, model.mixer_layers[i].mlp1.fc1)
        mixer_m1_fc2 = Conv1DMapper(mixer_m1_fc1, model.mixer_layers[i].mlp1.fc2)
        mixer_m1_fc2_norm = NormalizerMapper(mixer_m1_fc2, model.mixer_layers[i].mlp1.ln)
        
        mixer_m2_fc1 = LinearMapper(mixer_m1_fc2_norm, model.mixer_layers[i].mlp2.fc1)
        mixer_m2_fc2 = LinearMapper(mixer_m2_fc1, model.mixer_layers[i].mlp2.fc2)
        mixer_m2_fc2_norm = NormalizerMapper(mixer_m2_fc2, model.mixer_layers[i].mlp2.ln)

        prev = mixer_m2_fc2_norm
    
    avg_pool = AvgPoolMapper(prev)
    classifier = LinearMapper(avg_pool, model.clf)
            
    output_list = []
    for source in classifier.map(mapping).flatten():
        output_list.append(source)
    
    chip = to_chip(
        input_size, output_list, 
        mapping, mapping.max_axons, mapping.max_neurons,
        leak, quantize, weight_type)     
    
    assert chip.core_count <= max_cores

    return chip

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())