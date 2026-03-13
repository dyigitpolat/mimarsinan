"""
SoftCoreMapping and map_mm: map a ModelRepresentation to a list of SoftCores.

This module is separate from mapping_utils to break the cycle: ir.py imports
SoftCoreMapping only lazily (inside ir_graph_to_soft_core_mapping). This module
uses lazy imports for compress_spike_sources so it does not pull in mapping.ir
at load time.
"""

from __future__ import annotations

import numpy as np
import torch

from mimarsinan.code_generation.cpp_chip_model import SpikeSource
from mimarsinan.mapping.softcore_mapping import SoftCore


def _to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    return tensor_or_array.detach().cpu().numpy()


class SoftCoreMapping:
    def __init__(
        self,
        q_max=1.0,
        firing_mode="Default",
        max_axons: int | None = None,
        max_neurons: int | None = None,
    ):
        self.cores = []
        self.output_sources = []

        self.q_max = q_max
        self.firing_mode = firing_mode

        self.max_axons = max_axons
        self.max_neurons = max_neurons

        assert firing_mode in ["Default", "Novena", "TTFS"]

        self._psum_group_counter = 0
        self._output_source_spans = None

    def map(self, model_representation):
        self.output_sources = np.array(model_representation.map(self)).flatten().tolist()
        self._output_source_spans = None

    def map_fc(
        self,
        input_tensor_sources,
        output_shape,
        fc_weights,
        fc_biases=None,
        activation_scale=torch.tensor(1.0),
        parameter_scale=torch.tensor(1.0),
        input_activation_scale=torch.tensor(1.0),
    ):
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
                core_matrix[-1, :] = _to_numpy(biases_block.flatten())

            core_matrix[:input_axons_count, :] = _to_numpy(weights_block).T

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

        if max_axons is None or max_neurons is None:
            return _map_fc_block(input_tensor_sources, fc_weights, fc_biases)

        max_axons = int(max_axons)
        max_neurons = int(max_neurons)

        required_axons = x_rows + (1 if fc_biases is not None else 0)
        if required_axons <= max_axons and o_rows <= max_neurons:
            return _map_fc_block(input_tensor_sources, fc_weights, fc_biases)

        if o_rows > max_neurons:
            pass

        out_blocks = []
        out_start = 0
        while out_start < o_rows:
            out_end = min(o_rows, out_start + max_neurons)
            out_blocks.append((out_start, out_end))
            out_start = out_end

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

        tile_size = max_axons
        tile_slices = []
        in_start = 0
        while in_start < x_rows:
            in_end = min(x_rows, in_start + tile_size)
            tile_slices.append((in_start, in_end))
            in_start = in_end

        tile_count = len(tile_slices)
        psum_group_id = self._psum_group_counter
        self._psum_group_counter += 1

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
                w_pos = torch.clamp(w_tile, min=0)
                w_neg = torch.clamp(-w_tile, min=0)

                pos_sources = _map_fc_block(
                    input_tensor_sources[ta:tb, :],
                    w_pos,
                    None,
                    core_name_prefix=f"psum_partial_pos[g{psum_group_id}][t{t_idx}][o{a}:{b}]",
                    psum_group_id=psum_group_id,
                    psum_role="partial_pos",
                )
                neg_sources = _map_fc_block(
                    input_tensor_sources[ta:tb, :],
                    w_neg,
                    None,
                    core_name_prefix=f"psum_partial_neg[g{psum_group_id}][t{t_idx}][o{a}:{b}]",
                    psum_group_id=psum_group_id,
                    psum_role="partial_neg",
                )

                partial_pos.append(pos_sources)
                partial_neg.append(neg_sources)

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
        """Range-compressed view of output_sources for fast simulation / compact inspection."""
        if self._output_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._output_source_spans = compress_spike_sources(self.output_sources)
        return self._output_source_spans


def map_mm(
    mapping,
    layer_sources,
    layer_weights,
    layer_biases=None,
    activation_scale=torch.tensor(1.0),
    parameter_scale=torch.tensor(1.0),
    input_activation_scale=torch.tensor(1.0),
):
    layer_output_shape = np.array([layer_weights.shape[-2], layer_sources.shape[-1]])
    return mapping.map_fc(
        layer_sources,
        layer_output_shape,
        layer_weights,
        layer_biases,
        activation_scale,
        parameter_scale,
        input_activation_scale,
    )
