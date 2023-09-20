from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.models.layers import *
from mimarsinan.mapping.softcore_mapping import *
from mimarsinan.transformations.weight_quantization import *

import einops
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
        axon_sources.append(SpikeSource(-1, 0, False, True)) 
    
    return Connection(axon_sources)

def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    else:
        return tensor_or_array.detach().cpu().numpy()

class SoftCoreMapping:
    def __init__(self):
        self.cores = []
        self.output_sources = []

        self.max_neurons = 64
        self.max_axons = 64
        pass

    def map(self, model_representation):
        self.output_sources = np.array(model_representation.map(self)).flatten().tolist()

    def map_fc(self, 
        input_tensor_sources,  # 
        output_shape, # 
        fc_weights,
        fc_biases = None,
        threshold = 1.0): # 

        w_rows = fc_weights.shape[-2]
        w_cols = fc_weights.shape[-1]
        x_rows = input_tensor_sources.shape[-2]
        x_cols = input_tensor_sources.shape[-1]
        o_rows = output_shape[-2]
        o_cols = output_shape[-1]

        assert o_rows == w_rows, "o_rows: {}, w_rows: {}".format(o_rows, w_rows)
        assert o_cols == x_cols, "o_cols: {}, x_cols: {}".format(o_cols, x_cols)
        assert x_rows == w_cols, "x_rows: {}, w_cols: {}".format(x_rows, w_cols)

        new_cores_count = o_cols
        out_neurons_count = o_rows
        input_axons_count = x_rows

        if(fc_biases is None):
            core_matrix = np.zeros([input_axons_count, out_neurons_count])
        else:
            core_matrix = np.zeros([input_axons_count+1, out_neurons_count])
            core_matrix[-1, :] = to_numpy(fc_biases.flatten())

        core_matrix[:input_axons_count, :] = to_numpy(fc_weights).T

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
                spike_sources.append(SpikeSource(-3, 0, False, False, True))
            
            assert len(spike_sources) == core_matrix.shape[0]
            self.cores.append(
                SoftCore(core_matrix, spike_sources.copy(), len(self.cores), threshold))

        layer_sources = []
        core_offset = len(self.cores) - new_cores_count
        for neuron_idx in range(o_rows):
            layer_sources.append(
                [SpikeSource(core_offset + core_idx, neuron_idx) for core_idx in range(o_cols)])
        
        return np.array(layer_sources)

def map_mm(mapping, layer_sources, layer_weights, layer_biases = None, threshold = 1.0):
    layer_output_shape = np.array([layer_weights.shape[-2], layer_sources.shape[-1]])
    return mapping.map_fc(
        layer_sources, 
        layer_output_shape, 
        layer_weights,
        layer_biases,
        threshold)   

class InputMapper:
    def __init__(self, input_shape):
        if len(input_shape) == 1:
            input_shape = (1, input_shape[0])
            
        if isinstance(input_shape, int):
            input_shape = (1, input_shape)

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
                SpikeSource(-2, input_idx, True, False))
        
        self.sources = np.array(input_sources).reshape(self.input_shape)
        return self.sources
    
class ReshapeMapper:
    def __init__(self, source_mapper, output_shape):
        self.source_mapper = source_mapper
        self.output_shape = output_shape
        self.sources = None
    
    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        self.sources = self.source_mapper.map(mapping).reshape(self.output_shape)
        return self.sources
    
class EinopsRearrangeMapper:
    def __init__(self, source_mapper, einops_str, *einops_args, **einops_kwargs):
        self.source_mapper = source_mapper
        self.einops_str = einops_str
        self.einops_args = einops_args
        self.einops_kwargs = einops_kwargs
        self.sources = None
    
    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_sources = self.source_mapper.map(mapping)
        layer_sources = einops.einops.rearrange(
            layer_sources, self.einops_str, *self.einops_args, **self.einops_kwargs)

        self.sources = layer_sources
        return self.sources
    
class StackMapper:
    def __init__(self, source_mappers):
        self.source_mappers = source_mappers
        self.sources = None
    
    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_sources_list = []
        for mapper in self.source_mappers:
            sources = mapper.map(mapping)
            layer_sources_list.append(sources)
        layer_sources = np.stack(layer_sources_list).squeeze()
        
        self.sources = layer_sources
        return self.sources

def get_fused_weights(linear_layer, bn_layer):
    l = linear_layer
    bn = bn_layer

    w = l.weight.data
    b = l.bias.data if l.bias is not None else torch.zeros(w.shape[0]).to(w.device)

    new_w = w.clone()
    new_b = b.clone()

    gamma = bn.weight.data.to(w.device)
    beta = bn.bias.data.to(w.device)
    var = bn.running_var.data.to(w.device)
    mean = bn.running_mean.data.to(w.device)
    u = gamma / torch.sqrt(var + bn.eps)

    new_w[:,:] = w * u.unsqueeze(1)
    new_b[:] = (b - mean) * u + beta
    
    return new_w, new_b

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

class SubscriptMapper:
    def __init__(self, source_mapper, index):
        self.source_mapper = source_mapper
        self.index = index

        self.sources = None

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_sources = self.source_mapper.map(mapping)[self.index]

        self.sources = layer_sources
        return self.sources

class PerceptronMapper:
    def __init__(self, source_mapper, perceptron):
        self.perceptron = perceptron
        self.source_mapper = source_mapper
        self.sources = None

    def fuse_normalization(self):
        if isinstance(self.perceptron.normalization, nn.Identity):
            layer = self.perceptron.layer
            w = layer.weight.data
            b = layer.bias.data if layer.bias is not None else None
            return w, b
        
        assert isinstance(self.perceptron.normalization, FrozenStatsNormalization)
        assert self.perceptron.normalization.affine

        return get_fused_weights(
            linear_layer=self.perceptron.layer, 
            bn_layer=self.perceptron.normalization)

    def map(self, mapping):
        if self.sources is not None:
            return self.sources
        
        layer_weights, layer_biases = self.fuse_normalization()
        layer_weights.detach().cpu().numpy()
        layer_biases.detach().cpu().numpy()

        layer_sources = self.source_mapper.map(mapping)
        layer_sources = layer_sources.transpose()
        layer_sources = map_mm(
            mapping, 
            layer_sources, 
            layer_weights, 
            layer_biases, 
            self.perceptron.activation_scale)
        layer_sources = layer_sources.transpose()

        self.sources = layer_sources
        return self.sources
    
class ModelRepresentation:
    def __init__(self, output_layer_mapper):
        self.output_layer_mapper = output_layer_mapper

    def map(self, mapping):
        return self.output_layer_mapper.map(mapping)

def hard_cores_to_chip(input_size, hardcore_mapping, axons_per_core, neurons_per_core, leak, weight_type):
    output_sources = hardcore_mapping.output_sources

    hardcores = [
        generate_core_weights(
            neurons_per_core, 
            axons_per_core, 
            hardcore.core_matrix.transpose(),
            neurons_per_core,
            hardcore.threshold)
        for hardcore in hardcore_mapping.cores
    ]

    hardcore_connections = \
        [ Connection(hardcore.axon_sources) for hardcore in hardcore_mapping.cores ]
        
    chip = ChipModel(
        axons_per_core, neurons_per_core, len(hardcores), input_size,
        len(output_sources), leak, hardcore_connections, output_sources, hardcores, 
        weight_type
    )

    chip.load_from_json(chip.get_chip_json()) # sanity check
    
    return chip