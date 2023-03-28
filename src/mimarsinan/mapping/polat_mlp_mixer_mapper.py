from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.mapping.mapping_utils import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.models.polat_mlp_mixer import *
    
def get_polat_mlp_repr(input_shape, model):
    out = InputMapper(input_shape)
    out = PatchEmbeddingMapper(out, model.patch_emb[0])
    for i in range(model.num_layers):
        out = Conv1DMapper(out, model.mixer_layers[i].mlp1.fc1)
        out = Conv1DMapper(out, model.mixer_layers[i].mlp1.fc2)
        #out = BatchNormMapper(out, model.mixer_layers[i].mlp1.ln)
        
        out = LinearMapper(out, model.mixer_layers[i].mlp2.fc1)
        out = LinearMapper(out, model.mixer_layers[i].mlp2.fc2)
        out = BatchNormMapper(out, model.mixer_layers[i].mlp2.ln)
    
    out = AvgPoolMapper(out)
    out = LinearMapper(out, model.clf)

    return ModelRepresentation(out)


def polat_mlp_mixer_to_chip(
    omihub_mlp_mixer_model: PolatMLPMixer,
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
    input_shape = (in_channels, in_height, in_width)

    model_repr = get_polat_mlp_repr(input_shape, model)
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