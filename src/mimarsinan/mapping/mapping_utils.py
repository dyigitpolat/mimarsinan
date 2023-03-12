from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.models.layers import *
from mimarsinan.mapping.softcore_mapping import *
from mimarsinan.mapping.weight_quantization import *

import numpy as np

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

class Mapping:
    def __init__(self):
        self.soft_cores = []

        self.max_neurons = 64
        self.max_axons = 64
        pass

    def map_fc(self, 
        input_tensor_sources,  # 
        output_shape, # 
        fc_weights,
        fc_biases = None): # 

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

        if(fc_biases is None):
            core_matrix = np.zeros([input_axons_count, out_neurons_count])
        else:
            core_matrix = np.zeros([input_axons_count+1, out_neurons_count])
            core_matrix[-1, :] = fc_biases.flatten()

        core_matrix[:input_axons_count, :] = fc_weights.transpose()

        for i in range(new_cores_count): 
            spike_sources = []
            for j in range(input_axons_count):
                source_core = input_tensor_sources[j, i].core_
                source_neuron = input_tensor_sources[j, i].neuron_
                source_is_input = input_tensor_sources[j, i].is_input_
                source_is_off = input_tensor_sources[j, i].is_off_
                spike_sources.append(SpikeSource(
                    source_core, 
                    source_neuron,
                    source_is_input,
                    source_is_off))
            
            if(fc_biases is not None):
                spike_sources.append(SpikeSource(0, 0, False, False, True))
            
            self.soft_cores.append(
                SoftCore(core_matrix, spike_sources.copy(), len(self.soft_cores)))

        layer_sources = []
        core_offset = len(self.soft_cores) - new_cores_count
        for neuron_idx in range(o_rows):
            layer_sources.append(
                [SpikeSource(core_offset + core_idx, neuron_idx) for core_idx in range(o_cols)])
        
        return np.array(layer_sources)

def map_mm(mapping, layer_sources, layer_weights, layer_biases = None):
    layer_output_shape = np.array([layer_weights.shape[-2], layer_sources.shape[-1]])
    return mapping.map_fc(
        layer_sources, 
        layer_output_shape, 
        layer_weights,
        layer_biases)   

class InputMapper:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.sources = None
    
    def map(self, _):
        if self.sources is not None:
            return self.sources
    
        input_length = 1
        for dim in self.input_shape:
            input_length *= dim
        
        input_sources = []
        for input_idx in range(input_length):
            input_sources.append(
                SpikeSource(0, input_idx, True, False))
        
        self.sources = np.array(input_sources).reshape(self.input_shape)
        return self.sources

class Conv1DMapper:
    def __init__(self, source_mapper, layer):
        self.layer = layer
        self.source_mapper = source_mapper
        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_weights = self.layer.weight.data.numpy().squeeze()
        if self.layer.bias is not None:
            layer_biases = self.layer.bias.data.numpy().squeeze()
        else:
            layer_biases = None

        self.sources = map_mm(mapping, self.source_mapper.map(mapping), layer_weights, layer_biases)
        return self.sources

class LinearMapper:
    def __init__(self, source_mapper, layer):
        self.layer = layer
        self.source_mapper = source_mapper
        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_weights = self.layer.weight.data.numpy().squeeze()
        if self.layer.bias is not None:
            layer_biases = self.layer.bias.data.numpy().squeeze()
        else:
            layer_biases = None
        
        layer_sources = self.source_mapper.map(mapping)
        layer_sources = layer_sources.transpose()
        layer_sources = map_mm(mapping, layer_sources, layer_weights, layer_biases)
        layer_sources = layer_sources.transpose()

        self.sources = layer_sources
        return self.sources

class AvgPoolMapper:
    def __init__(self, source_mapper):
        self.source_mapper = source_mapper
        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_sources = self.source_mapper.map(mapping)

        factor =  1.0 / layer_sources.shape[0]
        weights = np.ones([1, layer_sources.shape[0]]) * factor

        self.sources = map_mm(mapping, layer_sources, weights)
        return self.sources 
    
class NormalizerMapper:
    def __init__(self, source_mapper, layer):
        self.source_mapper = source_mapper
        self.layer = layer
        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        self.source_mapper.layer.weight.data *= self.layer.get_factor()

        self.sources = self.source_mapper.map(mapping)
        return self.sources
class AddMapper:
    def __init__(self, source_mapper_a, source_mapper_b):
        self.source_mapper_a = source_mapper_a
        self.source_mapper_b = source_mapper_b
        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_sources_a = self.source_mapper_a.map(mapping)
        layer_sources_b = self.source_mapper_b.map(mapping)

        assert layer_sources_a.shape == layer_sources_b.shape

        x_rows = layer_sources_a.shape[-2]
        layer_sources = np.concatenate([layer_sources_a, layer_sources_b], axis=0)
        weights = np.concatenate([np.eye(x_rows), np.eye(x_rows)], axis=0).transpose()

        self.sources = map_mm(mapping, layer_sources, weights)
        return self.sources
    
class PatchEmbeddingMapper:
    def __init__(self, source_mapper, layer):
        self.source_mapper = source_mapper
        self.layer = layer
        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        kernel_weights = self.layer.weight.data.numpy()
        layer_sources = self.source_mapper.map(mapping)

        in_channels = layer_sources.shape[-3]
        in_height = layer_sources.shape[-2]
        in_width = layer_sources.shape[-1]

        kernel_h = kernel_weights.shape[2]
        kernel_w = kernel_weights.shape[3]

        patch_rows = in_height // kernel_h
        patch_cols = in_width // kernel_w

        kernel_weights = kernel_weights.reshape(
            kernel_weights.shape[0], in_channels * kernel_h * kernel_w)

        output_sources = []
        for i in range(patch_rows):
            for j in range(patch_cols):
                patch_sources = \
                    layer_sources[:, i*kernel_h:(i+1)*kernel_h, j*kernel_w:(j+1)*kernel_w]
                
                patch_sources = patch_sources.reshape(
                    in_channels * kernel_h * kernel_w, 1)
                
                output_sources.append(
                    map_mm(mapping, patch_sources, kernel_weights))
                
        self.sources =  np.array(output_sources).reshape(
            patch_rows * patch_cols, kernel_weights.shape[0])
        
        return self.sources

def to_chip(input_size, output_sources, softcore_mapping, axons_per_core, neurons_per_core, leak, quantize, weight_type):
    if quantize:
        quantize_softcores(softcore_mapping.soft_cores, bits=4)

    hardcore_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hardcore_mapping.map(softcore_mapping.soft_cores, output_sources)

    hardcores = [
        generate_core_weights(
            neurons_per_core, 
            axons_per_core, 
            hardcore.core_matrix.transpose(),
            neurons_per_core,
            hardcore.threshold)
        for hardcore in hardcore_mapping.hardcores
    ]

    hardcore_connections = \
        [ Connection(hardcore.axon_sources) for hardcore in hardcore_mapping.hardcores ]
        
    chip = ChipModel(
        axons_per_core, neurons_per_core, len(hardcores), input_size,
        len(output_sources), leak, hardcore_connections, output_sources, hardcores, 
        weight_type
    )

    chip.load_from_json(chip.get_chip_json()) # sanity check
    
    return chip
