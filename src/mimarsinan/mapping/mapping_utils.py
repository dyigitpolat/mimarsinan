from mimarsinan.code_generation.cpp_chip_model import *
from mimarsinan.models.layers import *
from mimarsinan.mapping.softcore_mapping import *
from mimarsinan.transformations.weight_quantization import *

from mimarsinan.transformations.perceptron_transformer import PerceptronTransformer
from mimarsinan.models.perceptron_mixer.perceptron import Perceptron

import einops
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

# Import IR types (late binding to avoid circular imports)
def _get_ir_types():
    from mimarsinan.mapping.ir import IRSource
    return IRSource

def _create_ir_input_source(idx):
    IRSource = _get_ir_types()
    return IRSource(node_id=-2, index=idx)

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
    def __init__(
        self,
        q_max=1.0,
        firing_mode="Default",
        max_axons: int | None = None,
        max_neurons: int | None = None,
        allow_axon_tiling: bool = False,
    ):
        self.cores = []
        self.output_sources = []

        self.q_max = q_max
        self.firing_mode = firing_mode

        self.max_axons = max_axons
        self.max_neurons = max_neurons
        self.allow_axon_tiling = bool(allow_axon_tiling)
        
        assert firing_mode in ["Default", "Novena", "TTFS"]
        
        self._psum_group_counter = 0
        self._output_source_spans = None

    def map(self, model_representation):
        self.output_sources = np.array(model_representation.map(self)).flatten().tolist()
        self._output_source_spans = None

    def map_fc(self, 
        input_tensor_sources,  # 
        output_shape, # 
        fc_weights,
        fc_biases = None,
        activation_scale = torch.tensor(1.0), 
        parameter_scale = torch.tensor(1.0),
        input_activation_scale = torch.tensor(1.0)): # 

        # Helper: map a single FC block without tiling.
        def _map_fc_block(
            input_sources_block,
            weights_block,
            biases_block,
            *,
            core_name_prefix: str | None = None,
            psum_group_id: int | None = None,
            psum_role: str | None = None,
        ):
            w_rows = weights_block.shape[-2]
            w_cols = weights_block.shape[-1]
            x_rows = input_sources_block.shape[-2]
            x_cols = input_sources_block.shape[-1]

            assert x_rows == w_cols, "x_rows: {}, w_cols: {}".format(x_rows, w_cols)

            new_cores_count = x_cols
            out_neurons_count = w_rows
            input_axons_count = x_rows

            if biases_block is None:
                core_matrix = np.zeros([input_axons_count, out_neurons_count])
            else:
                core_matrix = np.zeros([input_axons_count + 1, out_neurons_count])
                core_matrix[-1, :] = to_numpy(biases_block.flatten())

            core_matrix[:input_axons_count, :] = to_numpy(weights_block).T

            for i in range(new_cores_count):
                spike_sources = []
                for j in range(input_axons_count):
                    source_core = input_sources_block[j, i].core_
                    source_neuron = input_sources_block[j, i].neuron_
                    source_is_input = input_sources_block[j, i].is_input_
                    source_is_off = input_sources_block[j, i].is_off_
                    spike_sources.append(
                        SpikeSource(source_core, source_neuron, source_is_input, source_is_off)
                    )

                if biases_block is not None:
                    spike_sources.append(SpikeSource(-3, 0, False, False, True))

                assert len(spike_sources) == core_matrix.shape[0]

                core_id = len(self.cores)
                core_name = None
                if core_name_prefix is not None:
                    core_name = f"{core_name_prefix}:{i}"

                self.cores.append(
                    SoftCore(
                        core_matrix.copy(),
                        spike_sources.copy(),
                        core_id,
                        activation_scale,
                        parameter_scale,
                        input_activation_scale,
                        name=core_name,
                        psum_group_id=psum_group_id,
                        psum_role=psum_role,
                    )
                )

            layer_sources = []
            core_offset = len(self.cores) - new_cores_count
            for neuron_idx in range(out_neurons_count):
                layer_sources.append(
                    [SpikeSource(core_offset + core_idx, neuron_idx) for core_idx in range(new_cores_count)]
                )

            return np.array(layer_sources)

        w_rows = fc_weights.shape[-2]
        w_cols = fc_weights.shape[-1]
        x_rows = input_tensor_sources.shape[-2]
        x_cols = input_tensor_sources.shape[-1]
        o_rows = output_shape[-2]
        o_cols = output_shape[-1]

        assert o_rows == w_rows, "o_rows: {}, w_rows: {}".format(o_rows, w_rows)
        assert o_cols == x_cols, "o_cols: {}, x_cols: {}".format(o_cols, x_cols)
        assert x_rows == w_cols, "x_rows: {}, w_cols: {}".format(x_rows, w_cols)

        max_axons = self.max_axons
        max_neurons = self.max_neurons

        # Backward-compatible: if limits are not provided, behave as before (no tiling).
        if max_axons is None or max_neurons is None:
            return _map_fc_block(input_tensor_sources, fc_weights, fc_biases)

        max_axons = int(max_axons)
        max_neurons = int(max_neurons)

        required_axons = x_rows + (1 if fc_biases is not None else 0)
        if required_axons <= max_axons and o_rows <= max_neurons:
            return _map_fc_block(input_tensor_sources, fc_weights, fc_biases)

        if not self.allow_axon_tiling and required_axons > max_axons:
            raise ValueError(
                f"Axon tiling disabled: required axons={required_axons} exceeds max_axons={max_axons}."
            )

        if o_rows > max_neurons:
            # Output-channel tiling for safety (already commonly handled at model level).
            pass

        # Output splitting: always split outputs into blocks that fit max_neurons.
        out_blocks = []
        out_start = 0
        while out_start < o_rows:
            out_end = min(o_rows, out_start + max_neurons)
            out_blocks.append((out_start, out_end))
            out_start = out_end

        # If axon tiling not needed, just map each output block.
        if required_axons <= max_axons:
            mapped = []
            for (a, b) in out_blocks:
                mapped.append(
                    _map_fc_block(
                        input_tensor_sources,
                        fc_weights[a:b, :],
                        (fc_biases[a:b] if fc_biases is not None else None),
                    )
                )
            return np.concatenate(mapped, axis=0)

        # Axon tiling with explicit partial-sum accumulation:
        # - Split input features into tiles of size <= max_axons (no bias in partials)
        # - Split weights into positive and negative parts per tile so partial cores remain
        #   non-negative contributions; accumulator subtracts negative contributions.
        # - For each output block, create partial cores per tile (pos+neg) and then an
        #   accumulator core that sums (+) and subtracts (-) partial outputs per neuron.
        tile_size = max_axons  # partials do not include bias
        tile_slices = []
        in_start = 0
        while in_start < x_rows:
            in_end = min(x_rows, in_start + tile_size)
            tile_slices.append((in_start, in_end))
            in_start = in_end

        tile_count = len(tile_slices)
        psum_group_id = self._psum_group_counter
        self._psum_group_counter += 1

        # To wire an elementwise sum using a standard core (shared axon_sources),
        # we expand the accumulator input vector as:
        #   (tile0_pos_all_neurons, ..., tileN_pos_all_neurons,
        #    tile0_neg_all_neurons, ..., tileN_neg_all_neurons)
        # so accumulator axons = (2 * tile_count * out_block_size) (+ bias).
        # This implies an additional constraint on out_block_size for the accumulator.
        bias_axons = 1 if fc_biases is not None else 0
        max_out_by_accum_axons = (max_axons - bias_axons) // (2 * tile_count)
        if max_out_by_accum_axons <= 0:
            raise ValueError(
                f"Cannot build psum accumulator: tile_count={tile_count} requires at least {2 * tile_count + bias_axons} axons."
            )

        out_block_size = min(max_neurons, max_out_by_accum_axons)
        if out_block_size <= 0:
            raise ValueError(
                f"Cannot build psum mapping: out_block_size={out_block_size}."
            )

        mapped = []
        a = 0
        while a < o_rows:
            b = min(o_rows, a + out_block_size)
            out_block = b - a

            weights_block = fc_weights[a:b, :]
            biases_block = (fc_biases[a:b] if fc_biases is not None else None)

            partial_pos = []
            partial_neg = []
            for t_idx, (ta, tb) in enumerate(tile_slices):
                w_tile = weights_block[:, ta:tb]
                if not torch.is_tensor(w_tile):
                    w_tile = torch.as_tensor(w_tile, dtype=torch.float32)
                # Split signed weights into non-negative contributions.
                w_pos = torch.clamp(w_tile, min=0)
                w_neg = torch.clamp(-w_tile, min=0)

                pos_sources = _map_fc_block(
                    input_tensor_sources[ta:tb, :],
                    w_pos,
                    None,  # bias is applied once in the accumulator
                    core_name_prefix=f"psum_partial_pos[g{psum_group_id}][t{t_idx}][o{a}:{b}]",
                    psum_group_id=psum_group_id,
                    psum_role="partial_pos",
                )
                neg_sources = _map_fc_block(
                    input_tensor_sources[ta:tb, :],
                    w_neg,
                    None,  # bias is applied once in the accumulator
                    core_name_prefix=f"psum_partial_neg[g{psum_group_id}][t{t_idx}][o{a}:{b}]",
                    psum_group_id=psum_group_id,
                    psum_role="partial_neg",
                )

                partial_pos.append(pos_sources)
                partial_neg.append(neg_sources)

            # Build expanded accumulator input sources:
            # (all pos tiles, then all neg tiles) each flattened by neuron index.
            acc_in = np.empty((2 * tile_count * out_block, o_cols), dtype=object)
            row = 0
            for t_idx in range(tile_count):
                for n in range(out_block):
                    acc_in[row, :] = partial_pos[t_idx][n, :]
                    row += 1
            for t_idx in range(tile_count):
                for n in range(out_block):
                    acc_in[row, :] = partial_neg[t_idx][n, :]
                    row += 1

            # Accumulator weights: for each neuron n, sum pos and subtract neg across tiles.
            # IMPORTANT: weights in SoftCore are stored unscaled; chip_quantization multiplies by
            # core.parameter_scale to get integer weights. To represent a true +/-1 connection,
            # we must store +/- (1/parameter_scale) here.
            ps = parameter_scale.item() if hasattr(parameter_scale, "item") else float(parameter_scale)
            unit = 1.0 / float(ps)
            acc_w = np.zeros((out_block, 2 * tile_count * out_block), dtype=float)
            pos_off = 0
            neg_off = tile_count * out_block
            for t_idx in range(tile_count):
                for n in range(out_block):
                    acc_w[n, pos_off + t_idx * out_block + n] = unit
                    acc_w[n, neg_off + t_idx * out_block + n] = -unit

            acc_sources = _map_fc_block(
                acc_in,
                acc_w,
                biases_block,
                core_name_prefix=f"psum_accum[g{psum_group_id}][o{a}:{b}]",
                psum_group_id=psum_group_id,
                psum_role="accum",
            )

            mapped.append(acc_sources)
            a = b

        return np.concatenate(mapped, axis=0)

    def get_output_source_spans(self):
        """
        Range-compressed view of output_sources for fast simulation / compact inspection.
        """
        if self._output_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._output_source_spans = compress_spike_sources(self.output_sources)
        return self._output_source_spans

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
        self._ir_sources = None
        self._cached_ir_mapping_id = None

    def owned_perceptron_groups(self):
        """
        Introspection hook for Perceptron-first pipelines.

        Returns a list of perceptron groups owned by this mapper node.
        - Default: no perceptrons.
        - PerceptronMapper: [[perceptron]]
        - Conv-as-perceptrons mappers: [[p1, p2, ...]] (output-channel groups)
        """
        return []
    
    @property
    def source_mapper(self):
        return self._source_mapper_container[0]
    
    def clear_cache(self):
        self._cached_output = None
        self._cached_input_id = None
        self._ir_sources = None
        self._cached_ir_mapping_id = None
    
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

    def map_to_ir(self, ir_mapping):
        """
        Map this node to IR (unified IR that supports ComputeOp).
        
        This is the new mapping interface that supports both NeuralCore and ComputeOp.
        Default implementation falls back to _map_to_ir or raises NotImplementedError.
        """
        if self._ir_sources is not None and self._cached_ir_mapping_id == id(ir_mapping):
            return self._ir_sources
        
        self._ir_sources = self._map_to_ir(ir_mapping)
        self._cached_ir_mapping_id = id(ir_mapping)
        return self._ir_sources

    def _map_to_ir(self, ir_mapping):
        """
        Override in subclasses to produce IR nodes.
        
        Default: raises NotImplementedError. Subclasses should implement
        this to produce either NeuralCore or ComputeOp nodes.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _map_to_ir. "
            f"Override this method to support unified IR mapping."
        )

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

    def _map_to_ir(self, ir_mapping):
        """Create IR input sources."""
        input_length = 1
        for dim in self.input_shape:
            input_length *= dim
        
        input_sources = []
        for input_idx in range(input_length):
            input_sources.append(_create_ir_input_source(input_idx))
        
        return np.array(input_sources).reshape(self.input_shape)

    def _forward_impl(self, x):
        return x
    
class ReshapeMapper(Mapper):
    def __init__(self, source_mapper, output_shape):
        super(ReshapeMapper, self).__init__(source_mapper)
        self.output_shape = output_shape
    
    def _map(self, mapping):
        return self.source_mapper.map(mapping).reshape(self.output_shape)

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping).reshape(self.output_shape)

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

    def _map_to_ir(self, ir_mapping):
        layer_sources = self.source_mapper.map_to_ir(ir_mapping)
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

    def _map_to_ir(self, ir_mapping):
        layer_sources_list = []
        for mapper in self.source_mappers:
            sources = mapper.map_to_ir(ir_mapping)
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

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)[self.index]
        
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

    def _map_to_ir(self, ir_mapping):
        layer_weights = PerceptronTransformer().get_effective_weight(self.perceptron)
        layer_biases = PerceptronTransformer().get_effective_bias(self.perceptron)

        layer_sources = self.source_mapper.map_to_ir(ir_mapping)
        layer_sources = layer_sources.transpose()

        # Keep the same semantics as the legacy SoftCoreMapping path:
        # sources shape (in_features, core_count) -> map_fc -> (out_features, core_count)
        output_shape = np.array([layer_weights.shape[0], layer_sources.shape[-1]])
        layer_sources = ir_mapping.map_fc(
            layer_sources,
            output_shape,
            layer_weights,
            layer_biases,
            self.perceptron.activation_scale,
            self.perceptron.parameter_scale,
            self.perceptron.input_activation_scale,
            name=getattr(self.perceptron, "name", None),
        )

        return layer_sources.transpose()

    def _forward_impl(self, x):
        return self.perceptron(x)

    def owned_perceptron_groups(self):
        return [[self.perceptron]]
    
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

    def _map_to_ir(self, ir_mapping):
        # Same as _map: pass through sources
        return self.source_mapper.map_to_ir(ir_mapping)

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

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

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

    def _map_to_ir(self, ir_mapping):
        return self.source_mapper.map_to_ir(ir_mapping)

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

    def _map_to_ir(self, ir_mapping):
        sources = self.source_mapper.map_to_ir(ir_mapping)
        if len(sources.shape) == 1:
            return sources.reshape(1, -1)
        return sources

    def _forward_impl(self, x):
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x


def _chunk_sizes(total: int, chunk: int):
    assert chunk > 0
    sizes = []
    remaining = int(total)
    while remaining > 0:
        sizes.append(min(chunk, remaining))
        remaining -= sizes[-1]
    return sizes


# ---------------------------------------------------------------------------
# Non-neural (ComputeOp) Mappers
# ---------------------------------------------------------------------------

class MaxPool2DMapper(Mapper):
    """
    MaxPool2d operation.
    
    - Forward: nn.MaxPool2d
    - Old mapping (SoftCoreMapping): raises NotImplementedError
    - IR mapping (IRMapping): produces ComputeOp node
    """

    def __init__(
        self,
        source_mapper,
        kernel_size,
        stride=None,
        padding=0,
        input_spatial_shape=None,  # (H, W) of input (without batch/channel)
        input_channels=None,       # C of input
        name: str = "MaxPool2d",
    ):
        super().__init__(source_mapper)
        self.name = str(name)
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        
        self.input_spatial_shape = input_spatial_shape
        self.input_channels = input_channels
        
        self.pool = nn.MaxPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def _map(self, mapping):
        """Old mapping: not supported in SoftCoreMapping."""
        raise NotImplementedError(
            f"{self.name}: pooling is not supported in SoftCoreMapping. "
            f"Use IRMapping for unified IR that supports ComputeOps."
        )

    def _map_to_ir(self, ir_mapping):
        """IR mapping: produce a ComputeOp node."""
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        
        # Infer input shape from sources if not provided
        if self.input_spatial_shape is not None and self.input_channels is not None:
            c, h_in, w_in = self.input_channels, self.input_spatial_shape[0], self.input_spatial_shape[1]
        else:
            # Try to infer from source shape
            if len(input_sources.shape) == 3:
                c, h_in, w_in = input_sources.shape
            else:
                raise ValueError(
                    f"{self.name}: cannot infer input shape. "
                    f"Provide input_spatial_shape and input_channels, or ensure source has 3D shape."
                )
        
        # Compute output shape
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        output_sources = ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="max_pool2d",
            params={
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
            input_shape=(c, h_in, w_in),
            output_shape=(c, h_out, w_out),
            name=self.name,
        )
        
        return output_sources

    def _forward_impl(self, x):
        return self.pool(x)


class AvgPool2DMapper(Mapper):
    """
    AvgPool2d operation.
    
    - Forward: nn.AvgPool2d
    - Old mapping: raises NotImplementedError
    - IR mapping: produces ComputeOp node
    """

    def __init__(
        self,
        source_mapper,
        kernel_size,
        stride=None,
        padding=0,
        input_spatial_shape=None,
        input_channels=None,
        name: str = "AvgPool2d",
    ):
        super().__init__(source_mapper)
        self.name = str(name)
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        
        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)
        
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        
        self.input_spatial_shape = input_spatial_shape
        self.input_channels = input_channels
        
        self.pool = nn.AvgPool2d(
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: pooling is not supported in SoftCoreMapping. "
            f"Use IRMapping for unified IR that supports ComputeOps."
        )

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        
        if self.input_spatial_shape is not None and self.input_channels is not None:
            c, h_in, w_in = self.input_channels, self.input_spatial_shape[0], self.input_spatial_shape[1]
        else:
            if len(input_sources.shape) == 3:
                c, h_in, w_in = input_sources.shape
            else:
                raise ValueError(f"{self.name}: cannot infer input shape.")
        
        h_out = (h_in + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (w_in + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        output_sources = ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="avg_pool2d",
            params={
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
            },
            input_shape=(c, h_in, w_in),
            output_shape=(c, h_out, w_out),
            name=self.name,
        )
        
        return output_sources

    def _forward_impl(self, x):
        return self.pool(x)


class AdaptiveAvgPool2DMapper(Mapper):
    """
    AdaptiveAvgPool2d operation.
    
    - Forward: nn.AdaptiveAvgPool2d
    - Old mapping: raises NotImplementedError
    - IR mapping: produces ComputeOp node
    """

    def __init__(
        self,
        source_mapper,
        output_size,
        input_channels=None,
        name: str = "AdaptiveAvgPool2d",
    ):
        super().__init__(source_mapper)
        self.name = str(name)
        
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = tuple(output_size)
        
        self.input_channels = input_channels
        self.pool = nn.AdaptiveAvgPool2d(self.output_size)

    def _map(self, mapping):
        raise NotImplementedError(
            f"{self.name}: adaptive pooling is not supported in SoftCoreMapping. "
            f"Use IRMapping for unified IR that supports ComputeOps."
        )

    def _map_to_ir(self, ir_mapping):
        input_sources = self.source_mapper.map_to_ir(ir_mapping)
        
        if self.input_channels is not None:
            c = self.input_channels
        elif len(input_sources.shape) >= 1:
            c = input_sources.shape[0]
        else:
            raise ValueError(f"{self.name}: cannot infer input channels.")
        
        h_out, w_out = self.output_size
        
        output_sources = ir_mapping.add_compute_op(
            input_sources=input_sources,
            op_type="adaptive_avg_pool2d",
            params={"output_size": self.output_size},
            input_shape=tuple(input_sources.shape) if len(input_sources.shape) >= 2 else None,
            output_shape=(c, h_out, w_out),
            name=self.name,
        )
        
        return output_sources

    def _forward_impl(self, x):
        return self.pool(x)


class Conv2DPerceptronMapper(Mapper):
    """
    Convolution implemented as:
    - Forward: efficient nn.Conv2d
    - Mapping: shared-weight Perceptron (im2col + matmul), tiled as needed.
    """

    def __init__(
        self,
        source_mapper,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias: bool = True,
        max_neurons: int | None = None,
        max_axons: int | None = None,
        use_batchnorm: bool = True,
        name: str = "Conv2DPerceptron",
    ):
        super().__init__(source_mapper)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        if isinstance(kernel_size, tuple):
            self.kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
        else:
            k = int(kernel_size)
            self.kernel_size = (k, k)

        if isinstance(stride, tuple):
            self.stride = (int(stride[0]), int(stride[1]))
        else:
            s = int(stride)
            self.stride = (s, s)

        if isinstance(padding, tuple):
            self.padding = (int(padding[0]), int(padding[1]))
        else:
            p = int(padding)
            self.padding = (p, p)

        if isinstance(dilation, tuple):
            self.dilation = (int(dilation[0]), int(dilation[1]))
        else:
            d = int(dilation)
            self.dilation = (d, d)

        self.bias = bool(bias)
        self.name = str(name)
        self.use_batchnorm = bool(use_batchnorm)
        self.max_neurons = max_neurons # used only during map

        k_h, k_w = self.kernel_size
        patch_size = self.in_channels * k_h * k_w

        # Create a single Perceptron that wraps the conv weights for pipeline introspection/mapping
        self.perceptron = Perceptron(
            output_channels=self.out_channels,
            input_features=patch_size,
            normalization=(nn.LazyBatchNorm1d() if self.use_batchnorm else nn.Identity()),
            bias=self.bias,
            name=f"{self.name}_full",
        )

    def owned_perceptron_groups(self):
        # Expose the single shared-weight perceptron.
        # Pipeline transformations will apply to this perceptron.
        return [[self.perceptron]]

    def _forward_impl(self, x):
        if x.dim() != 4:
            raise ValueError(
                f"{self.name}: expected input (B,C,H,W), got shape {tuple(x.shape)}"
            )

        # IMPORTANT: pipeline steps (input activation analysis/scaling) decorate
        # `perceptron.input_activation` and expect it to be invoked during forward.
        x = self.perceptron.input_activation(x)

        # Use the Perceptron's weight/bias because the pipeline might have transformed them
        # (e.g. quantization, scaling, BN fusion).
        # We assume self.perceptron is the source of truth.
        w = self.perceptron.layer.weight.view(
            self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        )
        b = self.perceptron.layer.bias if self.bias else None

        y = F.conv2d(
            x, w, b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        # Apply Perceptron's normalization/activation/scaling
        # BN1d expects (N, C) or (N, C, L). We have (N, C, H, W).
        # We flatten spatial dims to L for normalization.
        if self.use_batchnorm or not isinstance(self.perceptron.normalization, nn.Identity):
            b_sz, c_sz, h_sz, w_sz = y.shape
            y = y.view(b_sz, c_sz, -1)
            y = self.perceptron.normalization(y)
            y = y.view(b_sz, c_sz, h_sz, w_sz)
        
        # Scaler and Activation usually work elementwise or broadcast correctly over (N, C, H, W).
        # MaxValueScaler might depend on stats? It usually tracks global max.
        y = self.perceptron.scaler(y)
        y = self.perceptron.activation(y)
        if self.training:
            y = self.perceptron.regularization(y)
        
        return y

    def _map(self, mapping):
        input_sources = self.source_mapper.map(mapping)

        if len(input_sources.shape) != 3:
            raise ValueError(
                f"{self.name}: expects 3D input sources (C,H,W), got {input_sources.shape}"
            )

        c_in, h_in, w_in = input_sources.shape
        if int(c_in) != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        d_h, d_w = self.dilation

        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        zero_source = SpikeSource(-1, 0, False, True)
        if p_h > 0 or p_w > 0:
            pad_width = ((0, 0), (p_h, p_h), (p_w, p_w))
            input_sources = np.pad(
                input_sources,
                pad_width,
                mode="constant",
                constant_values=zero_source,
            )

        unfolded_sources_list = []
        for oh in range(h_out):
            for ow in range(w_out):
                h_start = oh * s_h
                w_start = ow * s_w

                patch = []
                for c in range(self.in_channels):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            r = h_start + kh * d_h
                            c_idx = w_start + kw * d_w
                            patch.append(input_sources[c, r, c_idx])

                unfolded_sources_list.append(patch)

        unfolded_sources = np.array(unfolded_sources_list, dtype=object).transpose()

        # Get effective weights from the single perceptron
        full_w = PerceptronTransformer().get_effective_weight(self.perceptron)
        full_b = PerceptronTransformer().get_effective_bias(self.perceptron)

        # Slice weights if grouping is required by hardware constraints
        if self.max_neurons is None:
            group_sizes = [self.out_channels]
        else:
            group_sizes = _chunk_sizes(self.out_channels, int(self.max_neurons))

        mapped_groups = []
        start_idx = 0
        for g in group_sizes:
            end_idx = start_idx + g
            
            w_slice = full_w[start_idx:end_idx, :]
            b_slice = full_b[start_idx:end_idx] if full_b is not None else None

            mapped = map_mm(
                mapping,
                unfolded_sources,
                w_slice,
                b_slice,
                self.perceptron.activation_scale,
                self.perceptron.parameter_scale,
                self.perceptron.input_activation_scale,
            )
            mapped_groups.append(mapped)
            start_idx = end_idx

        mapped_sources = np.concatenate(mapped_groups, axis=0)
        return mapped_sources.reshape(self.out_channels, h_out, w_out)

    def _map_to_ir(self, ir_mapping):
        """Map to IR using NeuralCore nodes (im2col + matmul)."""
        input_sources = self.source_mapper.map_to_ir(ir_mapping)

        if len(input_sources.shape) != 3:
            raise ValueError(
                f"{self.name}: expects 3D input sources (C,H,W), got {input_sources.shape}"
            )

        c_in, h_in, w_in = input_sources.shape
        if int(c_in) != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )

        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        p_h, p_w = self.padding
        d_h, d_w = self.dilation

        h_out = (h_in + 2 * p_h - d_h * (k_h - 1) - 1) // s_h + 1
        w_out = (w_in + 2 * p_w - d_w * (k_w - 1) - 1) // s_w + 1

        # Pad input sources with "off" sources
        IRSource = _get_ir_types()
        off_source = IRSource(node_id=-1, index=0)
        if p_h > 0 or p_w > 0:
            pad_width = ((0, 0), (p_h, p_h), (p_w, p_w))
            input_sources = np.pad(
                input_sources,
                pad_width,
                mode="constant",
                constant_values=off_source,
            )

        # Get effective weights from the perceptron
        full_w = PerceptronTransformer().get_effective_weight(self.perceptron)
        full_b = PerceptronTransformer().get_effective_bias(self.perceptron)

        # For conv, we create one core per spatial position, all sharing weights.
        # Each core processes one patch (receptive field) and outputs out_channels values.
        # Total cores = h_out * w_out, each with patch_size inputs and out_channels outputs.
        
        patch_size = self.in_channels * k_h * k_w
        num_positions = h_out * w_out
        
        # Collect output sources for all positions
        all_output_sources = []  # List of (out_channels,) arrays
        
        for oh in range(h_out):
            for ow in range(w_out):
                h_start = oh * s_h
                w_start = ow * s_w

                # Build patch sources for this position
                patch_sources = []
                for c in range(self.in_channels):
                    for kh in range(k_h):
                        for kw in range(k_w):
                            r = h_start + kh * d_h
                            c_idx = w_start + kw * d_w
                            patch_sources.append(input_sources[c, r, c_idx])
                
                patch_sources = np.array(patch_sources)  # (patch_size,)

                # Slice weights if grouping is required by hardware constraints
                if self.max_neurons is None:
                    group_sizes = [self.out_channels]
                else:
                    group_sizes = _chunk_sizes(self.out_channels, int(self.max_neurons))

                position_outputs = []
                start_idx = 0
                for g_idx, g in enumerate(group_sizes):
                    end_idx = start_idx + g

                    w_slice = full_w[start_idx:end_idx, :]
                    b_slice = full_b[start_idx:end_idx] if full_b is not None else None

                    # Create neural core for this position and group
                    core_outputs = ir_mapping.add_neural_core(
                        input_sources=patch_sources,
                        weights=w_slice,
                        biases=b_slice,
                        activation_scale=self.perceptron.activation_scale,
                        parameter_scale=self.perceptron.parameter_scale,
                        input_activation_scale=self.perceptron.input_activation_scale,
                        name=f"{self.name}_pos{oh}_{ow}_g{g_idx}",
                    )
                    position_outputs.append(core_outputs)
                    start_idx = end_idx

                all_output_sources.append(np.concatenate(position_outputs))

        # Reshape: (num_positions, out_channels) -> (out_channels, h_out, w_out)
        # First stack to (num_positions, out_channels), then reshape
        output_array = np.stack(all_output_sources, axis=0)  # (h_out*w_out, out_channels)
        output_array = output_array.T  # (out_channels, h_out*w_out)
        return output_array.reshape(self.out_channels, h_out, w_out)


class Conv1DPerceptronMapper(Mapper):
    """
    1D Convolution implemented as:
    - Forward: efficient nn.Conv1d
    - Mapping: shared-weight Perceptron (unfold + matmul), tiled as needed.
    """

    def __init__(
        self,
        source_mapper,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        max_neurons: int | None = None,
        max_axons: int | None = None,
        use_batchnorm: bool = True,
        name: str = "Conv1DPerceptron",
    ):
        super().__init__(source_mapper)

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.bias = bool(bias)
        self.name = str(name)
        self.use_batchnorm = bool(use_batchnorm)
        self.max_neurons = max_neurons

        patch_size = self.in_channels * self.kernel_size

        self.perceptron = Perceptron(
            output_channels=self.out_channels,
            input_features=patch_size,
            normalization=(nn.LazyBatchNorm1d() if self.use_batchnorm else nn.Identity()),
            bias=self.bias,
            name=f"{self.name}_full",
        )

    def owned_perceptron_groups(self):
        return [[self.perceptron]]

    def _forward_impl(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"{self.name}: expected input (B,C,L), got shape {tuple(x.shape)}"
            )

        # IMPORTANT: pipeline steps decorate `perceptron.input_activation`
        # and expect it to be invoked during forward.
        x = self.perceptron.input_activation(x)

        w = self.perceptron.layer.weight.view(
            self.out_channels, self.in_channels, self.kernel_size
        )
        b = self.perceptron.layer.bias if self.bias else None

        y = F.conv1d(
            x, w, b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )

        if self.use_batchnorm or not isinstance(self.perceptron.normalization, nn.Identity):
            # BN1d on (N, C, L) works natively.
            y = self.perceptron.normalization(y)
        
        y = self.perceptron.scaler(y)
        y = self.perceptron.activation(y)
        if self.training:
            y = self.perceptron.regularization(y)
        return y

    def _map(self, mapping):
        input_sources = self.source_mapper.map(mapping)
        if len(input_sources.shape) != 2:
            raise ValueError(
                f"{self.name}: expects 2D input sources (C,L), got {input_sources.shape}"
            )

        c_in, l_in = input_sources.shape
        if int(c_in) != self.in_channels:
            raise ValueError(
                f"{self.name}: expected in_channels={self.in_channels}, got {c_in}"
            )

        k = self.kernel_size
        s = self.stride
        p = self.padding
        d = self.dilation

        if p > 0:
            zero_source = SpikeSource(-1, 0, False, True)
            zeros = np.full((self.in_channels, p), zero_source, dtype=object)
            input_sources = np.concatenate([zeros, input_sources, zeros], axis=-1)

        l_out = (l_in + 2 * p - d * (k - 1) - 1) // s + 1

        unfolded_list = []
        for i in range(l_out):
            start = i * s
            end = start + k * d
            window_indices = np.arange(start, end, d)
            window = input_sources[:, window_indices]
            unfolded_list.append(window.flatten())

        unfolded_sources = np.stack(unfolded_list, axis=1)  # (F, L_out)

        full_w = PerceptronTransformer().get_effective_weight(self.perceptron)
        full_b = PerceptronTransformer().get_effective_bias(self.perceptron)

        if self.max_neurons is None:
            group_sizes = [self.out_channels]
        else:
            group_sizes = _chunk_sizes(self.out_channels, int(self.max_neurons))

        mapped_groups = []
        start_idx = 0
        for g in group_sizes:
            end_idx = start_idx + g
            
            w_slice = full_w[start_idx:end_idx, :]
            b_slice = full_b[start_idx:end_idx] if full_b is not None else None

            mapped = map_mm(
                mapping,
                unfolded_sources,
                w_slice,
                b_slice,
                self.perceptron.activation_scale,
                self.perceptron.parameter_scale,
                self.perceptron.input_activation_scale,
            )
            mapped_groups.append(mapped)
            start_idx = end_idx

        mapped_sources = np.concatenate(mapped_groups, axis=0)  # (C_out, L_out)
        return mapped_sources

# ... Conv1DMapper and Conv2DMapper (old non-perceptron ones) removed or kept?
# The user's code uses Conv2DPerceptronMapper in VGG16.
# I will keep the old ones just in case but they are not used in new models.
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

    def map_to_ir(self, ir_mapping):
        """
        Map this model representation to a unified IR (IRGraph).
        
        This produces an IRGraph containing both NeuralCore and ComputeOp nodes.
        """
        return self.output_layer_mapper.map_to_ir(ir_mapping)
    
    def construct_pytorch_module(self, module, next):
        return self.output_layer_mapper.construct_pytorch_module(self.pytorch_module)

    def _ensure_exec_graph(self):
        """
        Build a reusable topological execution order once (postorder: deps first).
        Also reused for perceptron enumeration to guarantee consistent ordering.
        """
        def deps_of(node):
            if isinstance(node, StackMapper):
                return [m for m in node.source_mappers if m is not None]
            if isinstance(node, AddMapper):
                return [m for m in [node.source_mapper_a, node.source_mapper_b] if m is not None]
            if hasattr(node, "source_mapper") and node.source_mapper is not None:
                return [node.source_mapper]
            return []

        if self._exec_order is not None and self._deps is not None:
            return

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

    def get_perceptron_groups(self):
        """
        Return perceptron groups in forward-topological order.
        Groups are used by scale/activation analysis steps to propagate scales.
        """
        self._ensure_exec_graph()

        seen = set()
        groups = []

        for node in self._exec_order:
            if not hasattr(node, "owned_perceptron_groups"):
                continue
            for group in node.owned_perceptron_groups():
                # De-duplicate while preserving group structure
                unique = []
                for p in group:
                    if id(p) in seen:
                        continue
                    seen.add(id(p))
                    unique.append(p)
                if len(unique) > 0:
                    groups.append(unique)

        return groups

    def get_perceptrons(self):
        """
        Flattened list of perceptrons in forward-topological order.
        """
        perceptrons = []
        for group in self.get_perceptron_groups():
            perceptrons.extend(group)
        return perceptrons

    def __call__(self, x):
        """
        Execute the mapper graph as a single source of truth for forward.
        """
        self._ensure_exec_graph()

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
