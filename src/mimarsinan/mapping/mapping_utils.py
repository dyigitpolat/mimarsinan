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

def generate_core_connection_info(
    axons_count, ins, core, is_input_core):
    axon_sources = [SpikeSource(core, i, is_input_core) for i in range(ins)]
    for _ in range(axons_count - ins):
        axon_sources.append(SpikeSource(core, 0, False, True)) 
    
    return Connection(axon_sources)


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