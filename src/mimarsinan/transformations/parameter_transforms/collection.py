from mimarsinan.transformations.weight_quantization import TensorQuantization
from mimarsinan.transformations.weight_clipping import SoftTensorClipping

import torch

pipeline_clipping_rate = 0.01

def add_noise(w, amount = 0.01):
    return w + torch.rand_like(w) * amount

def special_decay(w):
    return torch.clamp(w, -1, 1)
    #return torch.sin(torch.tanh(w))

def decay_param(param_data):
    out = special_decay(param_data)
    return out

def decay_and_quantize_param(param_data):
    quantizer = TensorQuantization(bits=4)

    out = special_decay(param_data)
    out = quantizer.quantize(out)
    return out

def noisy_clip_decay_and_quantize_param(param_data):
    clipper = SoftTensorClipping(pipeline_clipping_rate)
    quantizer = TensorQuantization(bits=4)

    out = clipper.get_clipped_weights(param_data)
    out = add_noise(out)
    out = special_decay(out)
    out = quantizer.quantize(out)
    return out

def noisy_clip_and_decay_param(param_data):
    clipper = SoftTensorClipping(pipeline_clipping_rate)

    out = clipper.get_clipped_weights(param_data)
    out = add_noise(out)
    out = special_decay(out)
    return out

def clip_and_decay_param(param_data):
    clipper = SoftTensorClipping(pipeline_clipping_rate)

    out = clipper.get_clipped_weights(param_data)
    out = special_decay(out)
    return out