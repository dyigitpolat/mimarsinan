from mimarsinan.code_generation.cpp_chip_model import *

def generate_core_weights(
    neurons_count, axons_count, weight_tensor, outs, 
    thresh, bias_tensor = None):

    neurons: list[Neuron] = []
    for idx in range(neurons_count):
        if(idx < outs):
            neuron_ws = [ w.item() for w in weight_tensor[idx] ]

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