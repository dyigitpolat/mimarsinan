from mimarsinan.code_generation.cpp_chip_model import *
import math
import numpy as np

q_max = 7
q_min = -8

def quantize_weight(w, max_w, min_w):
    if(w > 0):
        if(max_w == 0): return 0
        return round((q_max * w) / max_w)
    else:
        if(min_w == 0): return 0
        return round((q_min * w) / min_w)

def do_quantize(w, max_w, min_w, switch=False):
    if(switch):
        return quantize_weight(w, max_w, min_w)
    else:
        return w
    
def quantize_weight_tensor(weight_tensor):
    max_w = weight_tensor.max().item()
    min_w = weight_tensor.min().item()
    return np.array([
        [quantize_weight(w.item(), max_w, min_w) for w in row] \
        for row in weight_tensor
    ])

def calculate_threshold(weight_tensor):
    max_w = weight_tensor.max().item()
    if(max_w == 0): max_w = 1.0
    return round((q_max * 1.0) / max_w)

def generate_core_weights(
    neurons_count, axons_count, weight_tensor, outs, 
    thresh, bias_tensor = None):

    neurons: list[Neuron] = []
    for idx in range(neurons_count):
        if(idx < outs):
            neuron_ws = [w for w in weight_tensor[idx]]

            for _ in range(axons_count - weight_tensor[idx].shape[0]):
                neuron_ws.append(int(0))
        else:
            neuron_ws = [int(0) for _ in range(axons_count)]

        bias = 0.0
        if(bias_tensor is not None) and (idx < outs): bias = bias_tensor[idx]
        
        neurons.append(Neuron(neuron_ws, thresh, bias))
    
    return Core(neurons)

def generate_core_weights_legacy(
    neurons_count, axons_count, weight_tensor, outs, 
    thresh, bias_tensor = None, quantize = False):
    
    max_w = weight_tensor.max().item()
    min_w = weight_tensor.min().item()
    if(quantize):
        if(max_w == 0): max_w = 1.0
        thresh = round(q_max * thresh / max_w)

    neurons: list[Neuron] = []
    for idx in range(neurons_count):
        if(idx < outs):
            neuron_ws = [ 
                do_quantize(w.item(), max_w, min_w, quantize) \
                for w in weight_tensor[idx] ]

            for _ in range(axons_count - weight_tensor[idx].shape[0]):
                neuron_ws.append(int(0))
        else:
            neuron_ws = [int(0) for _ in range(axons_count)]

        bias = 0.0
        if(bias_tensor is not None) and (idx < outs): bias = bias_tensor[idx]
        
        neurons.append(Neuron(neuron_ws, thresh, bias))
    
    return Core(neurons)

def generate_core_connection_info(
    axons_count, ins, core, is_input_core):
    axon_sources = [SpikeSource(core, i, is_input_core) for i in range(ins)]
    for _ in range(axons_count - ins):
        axon_sources.append(SpikeSource(core, 0, False, True)) 
    
    return Connection(axon_sources)