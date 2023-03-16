from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.mapping.weight_quantization import *
from mimarsinan.models.omihub_mlp_mixer import *

def get_omihub_repr(input_shape, model):
    input_mapper = InputMapper(input_shape)
    patch_emb_mapper = PatchEmbeddingMapper(input_mapper, model.patch_emb[0])
    prev_mapper = patch_emb_mapper
    for i in range(model.num_layers):
        mixer_m1_fc1 = Conv1DMapper(prev_mapper, model.mixer_layers[i].mlp1.fc1)
        mizer_m1_ln = NormalizerMapper(mixer_m1_fc1, model.mixer_layers[i].mlp1.ln)
        
        mixer_m1_fc2 = Conv1DMapper(mizer_m1_ln, model.mixer_layers[i].mlp1.fc2)
        mixer_m1_add = AddMapper(prev_mapper, mixer_m1_fc2)

        mixer_m2_fc1 = LinearMapper(mixer_m1_add, model.mixer_layers[i].mlp2.fc1)
        mixer_m2_ln = NormalizerMapper(mixer_m2_fc1, model.mixer_layers[i].mlp2.ln)
        
        mixer_m2_fc2 = LinearMapper(mixer_m2_ln, model.mixer_layers[i].mlp2.fc2)
        mixer_m2_add = AddMapper(mixer_m1_add, mixer_m2_fc2)
        prev_mapper = mixer_m2_add
        
    avg_pool_mapper = AvgPoolMapper(prev_mapper)
    classifier_mapper = LinearMapper(avg_pool_mapper, model.clf)

    return ModelRepresentation(classifier_mapper)

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
    input_shape = (in_channels, in_height, in_width)

    model_repr = get_omihub_repr(input_shape, model)
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

def export_json_to_file(chip, filename):
    with open(filename, 'w') as f:
        f.write(chip.get_chip_json())