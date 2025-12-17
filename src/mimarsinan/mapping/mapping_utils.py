from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.models.layers import *
from mimarsinan.mapping.softcore_mapping import *
from mimarsinan.transformations.weight_quantization import *

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer

import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

def generate_core_weights(
    neurons_count, axons_count, weight_tensor, outs, 
    thresh, latency, bias_tensor = None):
    
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
    
    return Core(neurons, latency)

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
    def __init__(self, q_max = 1.0, firing_mode = "Default"):
        self.cores = []
        self.output_sources = []

        self.q_max = q_max
        self.firing_mode = firing_mode
        
        assert firing_mode in ["Default", "Novena"]

    def map(self, model_representation):
        self.output_sources = np.array(model_representation.map(self)).flatten().tolist()

    def map_fc(self, 
        input_tensor_sources,  # 
        output_shape, # 
        fc_weights,
        fc_biases = None,
        activation_scale = torch.tensor(1.0), 
        parameter_scale = torch.tensor(1.0),
        input_activation_scale = torch.tensor(1.0)): # 

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
                SoftCore(core_matrix.copy(), spike_sources.copy(), len(self.cores), activation_scale, parameter_scale, input_activation_scale))

        layer_sources = []
        core_offset = len(self.cores) - new_cores_count
        for neuron_idx in range(o_rows):
            layer_sources.append(
                [SpikeSource(core_offset + core_idx, neuron_idx) for core_idx in range(o_cols)])
        
        return np.array(layer_sources)

def map_mm(
    mapping, 
    layer_sources, 
    layer_weights, 
    layer_biases = None, 
    activation_scale = torch.tensor(1.0), 
    parameter_scale = torch.tensor(1.0), 
    input_activation_scale = torch.tensor(1.0)):

    layer_output_shape = np.array([layer_weights.shape[-2], layer_sources.shape[-1]])
    return mapping.map_fc(
        layer_sources, 
        layer_output_shape, 
        layer_weights,
        layer_biases,
        activation_scale,
        parameter_scale,
        input_activation_scale)   

class Mapper(nn.Module):
    def __init__(self, source_mapper=None):
        super(Mapper, self).__init__()
        # Use a private list to hide the reference from PyTorch submodule registration
        self._source_mapper_container = [source_mapper]
        self.sources = None
        self._cached_mapping_id = None
        self._cached_output = None
        self._cached_input_id = None
    
    @property
    def source_mapper(self):
        return self._source_mapper_container[0]
    
    def clear_cache(self):
        self._cached_output = None
        self._cached_input_id = None
    
    def map(self, mapping):
        # IMPORTANT: sources are only valid for the specific `mapping` instance they
        # were created against. We keep a 1-entry cache keyed by `id(mapping)` so
        # rerunning mapping (e.g. pipeline resume / re-map with different q_max)
        # cannot accidentally reuse stale SpikeSources.
        if self.sources is not None and self._cached_mapping_id == id(mapping):
            return self.sources
        
        self.sources = self._map(mapping)
        self._cached_mapping_id = id(mapping)
        return self.sources
    
    def _map(self, mapping):
        raise NotImplementedError

    def forward(self, x):
        if self._cached_input_id == id(x) and self._cached_output is not None:
            return self._cached_output
        
        out = self._forward_impl(x)
        self._cached_input_id = id(x)
        self._cached_output = out
        return out

    def _forward_impl(self, x):
        raise NotImplementedError

class InputMapper(Mapper):
    def __init__(self, input_shape):
        super(InputMapper, self).__init__()
        if len(input_shape) == 1:
            input_shape = (1, input_shape[0])
            
        if isinstance(input_shape, int):
            input_shape = (1, input_shape)

        self.input_shape = input_shape
    
    def _map(self, mapping):
        input_length = 1
        for dim in self.input_shape:
            input_length *= dim
        
        input_sources = []
        for input_idx in range(input_length):
            input_sources.append(
                SpikeSource(-2, input_idx, True, False))
        
        return np.array(input_sources).reshape(self.input_shape)

    def _forward_impl(self, x):
        return x
    
class ReshapeMapper(Mapper):
    def __init__(self, source_mapper, output_shape):
        super(ReshapeMapper, self).__init__(source_mapper)
        self.output_shape = output_shape
    
    def _map(self, mapping):
        return self.source_mapper.map(mapping).reshape(self.output_shape)

    def _forward_impl(self, x):
        val = x
        return val.view(val.shape[0], *self.output_shape)
    
class DelayMapper(Mapper):
    def __init__(self, source_mapper, delay):
        super(DelayMapper, self).__init__(source_mapper)
        self.delay = delay
    
    def _map(self, mapping):
        layer_sources = self.source_mapper.map(mapping)
        for _ in range(self.delay):
            # passthrough
            layer_sources = map_mm(mapping, layer_sources, np.eye(layer_sources.shape[-2]), parameter_scale=torch.tensor(mapping.q_max))

        return layer_sources

    def _forward_impl(self, x):
        return x
    
class EinopsRearrangeMapper(Mapper):
    def __init__(self, source_mapper, einops_str, *einops_args, **einops_kwargs):
        super(EinopsRearrangeMapper, self).__init__(source_mapper)
        self.einops_str = einops_str
        self.einops_args = einops_args
        self.einops_kwargs = einops_kwargs
    
    def _map(self, mapping):
        layer_sources = self.source_mapper.map(mapping)
        return einops.einops.rearrange(
            layer_sources, self.einops_str, *self.einops_args, **self.einops_kwargs)

    def _forward_impl(self, x):
        return einops.einops.rearrange(
            x, self.einops_str, *self.einops_args, **self.einops_kwargs)
    
class StackMapper(Mapper):
    def __init__(self, source_mappers):
        super(StackMapper, self).__init__()
        # Use python list to hide inputs from PyTorch
        self._source_mappers_list = list(source_mappers)
    
    @property
    def source_mappers(self):
        return self._source_mappers_list

    def _map(self, mapping):
        layer_sources_list = []
        for mapper in self.source_mappers:
            sources = mapper.map(mapping)
            layer_sources_list.append(sources)
        return np.stack(layer_sources_list).squeeze()

    def _forward_impl(self, x):
        # `x` is expected to be an iterable/tuple of precomputed branch tensors.
        outputs = list(x)
        return torch.stack(outputs, dim=1).squeeze(1)
    
class AddMapper(Mapper):
    def __init__(self, source_mapper_a, source_mapper_b):
        super(AddMapper, self).__init__()
        self._source_mapper_a_container = [source_mapper_a]
        self._source_mapper_b_container = [source_mapper_b]

    @property
    def source_mapper_a(self):
        return self._source_mapper_a_container[0]
    
    @property
    def source_mapper_b(self):
        return self._source_mapper_b_container[0]

    def _map(self, mapping):
        layer_sources_a = self.source_mapper_a.map(mapping)
        layer_sources_b = self.source_mapper_b.map(mapping)

        assert layer_sources_a.shape == layer_sources_b.shape

        x_rows = layer_sources_a.shape[-2]
        layer_sources = np.concatenate([layer_sources_a, layer_sources_b], axis=0)
        weights = np.concatenate([np.eye(x_rows), np.eye(x_rows)], axis=0).transpose()

        return map_mm(mapping, layer_sources, weights, parameter_scale=torch.tensor(mapping.q_max))

    def _forward_impl(self, x):
        # `x` is expected to be a 2-tuple of precomputed tensors.
        a, b = x
        return a + b

class SubscriptMapper(Mapper):
    def __init__(self, source_mapper, index):
        super(SubscriptMapper, self).__init__(source_mapper)
        self.index = index

    def _map(self, mapping):
        return self.source_mapper.map(mapping)[self.index]

    def _forward_impl(self, x):
        return x.select(1, self.index)

class PerceptronMapper(Mapper):
    def __init__(self, source_mapper, perceptron):
        super(PerceptronMapper, self).__init__(source_mapper)
        self.perceptron = perceptron

    def _map(self, mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        layer_weights.detach().cpu().numpy()
        layer_biases.detach().cpu().numpy()

        layer_sources = self.source_mapper.map(mapping)
        layer_sources = layer_sources.transpose()
        layer_sources = map_mm(
            mapping, 
            layer_sources, 
            layer_weights, 
            layer_biases, 
            self.perceptron.activation_scale,
            self.perceptron.parameter_scale,
            self.perceptron.input_activation_scale)
        layer_sources = layer_sources.transpose()

        return layer_sources

    def _forward_impl(self, x):
        return self.perceptron(x)
    
class ModuleMapper(Mapper):
    """
    Forward-only module application mapper.
    For mapping (chip compilation), this acts as identity (passes sources through).
    """
    def __init__(self, source_mapper, module: nn.Module):
        super(ModuleMapper, self).__init__(source_mapper)
        self.module = module

    def _map(self, mapping):
        # Module effects (e.g. activations) are either already fused into perceptrons
        # or not representable in the linear mapping stage. Treat as identity.
        return self.source_mapper.map(mapping)

    def _forward_impl(self, x):
        return self.module(x)

class MergeLeadingDimsMapper(Mapper):
    """
    Flatten all leading dims except the last feature dim.

    Forward:
      (B, N, F) -> (B*N, F)
      (B, C, N, F) -> (B*C*N, F)
      If already 2D, returns as-is.

    Mapping:
      Usually identity because mapping tensors do not include batch.
    """
    def __init__(self, source_mapper):
        super(MergeLeadingDimsMapper, self).__init__(source_mapper)

    def _map(self, mapping):
        # Mapping sources generally have no batch dimension; keep shape.
        return self.source_mapper.map(mapping)

    def _forward_impl(self, x):
        if x.dim() <= 2:
            return x
        return x.reshape(-1, x.shape[-1])

class SplitLeadingDimMapper(Mapper):
    """
    Inverse of MergeLeadingDimsMapper for the common (B*N, F) -> (B, N, F) case.

    Forward:
      (B*N, F) -> (B, N, F)  where N = second_dim_size

    Mapping:
      Identity.
    """
    def __init__(self, source_mapper, second_dim_size: int):
        super(SplitLeadingDimMapper, self).__init__(source_mapper)
        self.second_dim_size = int(second_dim_size)

    def _map(self, mapping):
        return self.source_mapper.map(mapping)

    def _forward_impl(self, x):
        # If already 3D (e.g., caller didn't merge), keep as-is.
        if x.dim() != 2:
            return x
        n = self.second_dim_size
        assert x.shape[0] % n == 0, f"Cannot split leading dim {x.shape[0]} by {n}"
        b = x.shape[0] // n
        return x.view(b, n, x.shape[1])

class Ensure2DMapper(Mapper):
    """
    Ensure a 2D (Instances, Features) layout for mapping and a (Batch, Features)
    layout for forward.

    Mapping:
      (F,) -> (1, F)
      (N, F) -> unchanged

    Forward:
      (B, F) -> unchanged
      (F,) -> (1, F)
    """
    def __init__(self, source_mapper):
        super(Ensure2DMapper, self).__init__(source_mapper)

    def _map(self, mapping):
        sources = self.source_mapper.map(mapping)
        if len(sources.shape) == 1:
            return sources.reshape(1, -1)
        return sources

    def _forward_impl(self, x):
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

class Conv1DMapper(Mapper):
    def __init__(self, source_mapper, conv_layer):
        super(Conv1DMapper, self).__init__(source_mapper)
        self.conv_layer = conv_layer
    
    def _map(self, mapping):
        weights = self.conv_layer.weight
        bias = self.conv_layer.bias
        
        input_sources = self.source_mapper.map(mapping)
        
        C_in = input_sources.shape[-2]
        L_in = input_sources.shape[-1]
        
        if self.conv_layer.padding[0] > 0:
            pad = self.conv_layer.padding[0]
            zero_source = SpikeSource(-1, 0, False, True) 
            
            zeros = np.full((C_in, pad), zero_source, dtype=object)
            input_sources_padded = np.concatenate([zeros, input_sources, zeros], axis=-1)
        else:
            input_sources_padded = input_sources
            
        L_out = (L_in + 2 * self.conv_layer.padding[0] - self.conv_layer.dilation[0] * (self.conv_layer.kernel_size[0] - 1) - 1) // self.conv_layer.stride[0] + 1
        
        unfolded_sources_list = []
        K = self.conv_layer.kernel_size[0]
        S = self.conv_layer.stride[0]
        D = self.conv_layer.dilation[0]
        
        for i in range(L_out):
            start = i * S
            end = start + K * D
            window_indices = np.arange(start, end, D)
            window = input_sources_padded[:, window_indices]
            unfolded_sources_list.append(window.flatten())
            
        unfolded_sources = np.stack(unfolded_sources_list, axis=1)
        
        mm_weights = self.conv_layer.weight.view(self.conv_layer.out_channels, -1)
        mm_bias = self.conv_layer.bias
        
        mapped_sources = map_mm(
            mapping,
            unfolded_sources,
            mm_weights,
            mm_bias,
            torch.tensor(1.0), 
            torch.tensor(mapping.q_max)
        )
        
        return mapped_sources

    def _forward_impl(self, x):
        # Forward is the plain PyTorch op; compilation uses source_mapper via _map().
        return self.conv_layer(x)

class Conv2DMapper(Mapper):
    def __init__(self, source_mapper, conv_layer):
        super(Conv2DMapper, self).__init__(source_mapper)
        self.conv_layer = conv_layer

    def _map(self, mapping):
        input_sources = self.source_mapper.map(mapping)
        
        if len(input_sources.shape) != 3:
             raise ValueError(f"Conv2DMapper expects 3D input sources (C, H, W), got {input_sources.shape}")
             
        C_in, H_in, W_in = input_sources.shape
        
        K_h, K_w = self.conv_layer.kernel_size
        S_h, S_w = self.conv_layer.stride
        P_h, P_w = self.conv_layer.padding
        D_h, D_w = self.conv_layer.dilation
        
        H_out = (H_in + 2 * P_h - D_h * (K_h - 1) - 1) // S_h + 1
        W_out = (W_in + 2 * P_w - D_w * (K_w - 1) - 1) // S_w + 1
        
        zero_source = SpikeSource(-1, 0, False, True)
        if P_h > 0 or P_w > 0:
            pad_width = ((0, 0), (P_h, P_h), (P_w, P_w))
            input_sources_padded = np.pad(
                input_sources, 
                pad_width, 
                mode='constant', 
                constant_values=zero_source
            )
        else:
            input_sources_padded = input_sources
            
        unfolded_sources_list = []
        
        for h in range(H_out):
            for w in range(W_out):
                h_start = h * S_h
                w_start = w * S_w
                
                patch = []
                for c in range(C_in):
                    for kh in range(K_h):
                        for kw in range(K_w):
                            r = h_start + kh * D_h
                            c_idx = w_start + kw * D_w
                            patch.append(input_sources_padded[c, r, c_idx])
                
                unfolded_sources_list.append(patch)
                
        unfolded_sources = np.array(unfolded_sources_list).transpose()
        
        mm_weights = self.conv_layer.weight.reshape(self.conv_layer.out_channels, -1)
        mm_bias = self.conv_layer.bias
        
        mapped_sources = map_mm(
            mapping,
            unfolded_sources,
            mm_weights,
            mm_bias,
            torch.tensor(1.0),
            torch.tensor(mapping.q_max)
        )
        
        return mapped_sources.reshape(self.conv_layer.out_channels, H_out, W_out)

    def _forward_impl(self, x):
        # Forward is the plain PyTorch op; compilation uses source_mapper via _map().
        return self.conv_layer(x)

class ModelRepresentation:
    def __init__(self, output_layer_mapper):
        self.output_layer_mapper = output_layer_mapper
        self.pytorch_module = nn.Identity()
        self._exec_order = None
        self._deps = None

    def map(self, mapping):
        return self.output_layer_mapper.map(mapping)
    
    def construct_pytorch_module(self, module, next):
        return self.output_layer_mapper.construct_pytorch_module(self.pytorch_module)

    def __call__(self, x):
        """
        Execute the mapper graph as a single source of truth for forward.
        """
        def deps_of(node):
            if isinstance(node, StackMapper):
                return [m for m in node.source_mappers if m is not None]
            if isinstance(node, AddMapper):
                return [m for m in [node.source_mapper_a, node.source_mapper_b] if m is not None]
            if hasattr(node, "source_mapper") and node.source_mapper is not None:
                return [node.source_mapper]
            return []

        # Build a reusable topological execution order once (postorder: deps first).
        if self._exec_order is None or self._deps is None:
            deps_map = {}
            order = []
            visited = set()
            stack = [(self.output_layer_mapper, False)]

            while stack:
                node, expanded = stack.pop()
                if node is None:
                    continue
                if expanded:
                    order.append(node)
                    continue
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, True))
                d = deps_of(node)
                deps_map[node] = d
                # push deps (reverse to preserve natural left-to-right order)
                for dep in reversed(d):
                    if dep is not None and dep not in visited:
                        stack.append((dep, False))

            self._exec_order = order
            self._deps = deps_map

        values = {}
        for node in self._exec_order:
            d = self._deps.get(node, [])
            if len(d) == 0:
                # InputMapper (or a degenerate node) consumes the real input tensor.
                values[node] = node.forward(x)
            elif len(d) == 1:
                values[node] = node.forward(values[d[0]])
            else:
                values[node] = node.forward(tuple(values[dep] for dep in d))

        return values[self.output_layer_mapper]

def hard_cores_to_chip(input_size, hardcore_mapping, axons_per_core, neurons_per_core, leak, weight_type):
    output_sources = hardcore_mapping.output_sources

    hardcores = [
        generate_core_weights(
            neurons_per_core, 
            axons_per_core, 
            hardcore.core_matrix.transpose(),
            neurons_per_core,
            hardcore.threshold,
            hardcore.latency)
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
