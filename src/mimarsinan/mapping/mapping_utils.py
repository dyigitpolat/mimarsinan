from mimarsinan.code_generation.cpp_chip_model import *

q_max = 7
q_min = -8

def quantize_weight_to_int(
    weight, max_weight, min_weight):
    
    if(weight > 0):
        return int(
            ((q_max) * (weight)) / (max_weight))
    if(weight < 0): 
        return int(
            ((q_min) * (weight)) / (min_weight))
    
    return 0

def do_quantize_4bits(w, max, min, switch=False):
    if(switch):
        return quantize_weight_to_int(w, max, min)
    else:
        return w

def generate_core_weights(
    neurons_count, axons_count, weight_tensor, outs, 
    thresh, bias_tensor = None, quantize = False):

    max_w = weight_tensor.max().item()
    min_w = weight_tensor.min().item()
    if(quantize):
        thresh = int(q_max * thresh / max_w)

    neurons: list[Neuron] = []
    for idx in range(neurons_count):
        if(idx < outs):
            neuron_ws = [ 
                do_quantize_4bits(w.item(), max_w, min_w, quantize) \
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