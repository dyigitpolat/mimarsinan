"""Spiking simulator for unified IRGraph (NeuralCore + ComputeOp); LIF and TTFS modes."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from mimarsinan.chip_simulation import spike_modes
from mimarsinan.mapping.ir import ComputeOp, IRGraph, NeuralCore
from mimarsinan.mapping.support.ir_source_spans import IRSourceSpan, compress_ir_sources
from mimarsinan.models.spiking.spiking_config import (
    COMPUTE_DTYPE,
    validate_spiking_init,
)
from mimarsinan.models.spiking.ttfs_activation import ttfs_activation_from_type
from mimarsinan.models.spiking.unified.lif_step import UnifiedLifStepMixin
from mimarsinan.models.spiking.unified.stage_io import UnifiedStageIOMixin
from mimarsinan.models.spiking.unified.ttfs_step import UnifiedTtfsStepMixin

# Backward compatibility for tests and external callers.
_ttfs_activation_from_type = ttfs_activation_from_type


class SpikingUnifiedCoreFlow(
    UnifiedStageIOMixin,
    UnifiedLifStepMixin,
    UnifiedTtfsStepMixin,
    nn.Module,
):
    """Flat IRGraph spiking sim: LIF/TTFS cores, ComputeOp sync barriers, shared WeightBank params."""

    def __init__(
        self,
        input_shape,
        ir_graph: IRGraph,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
        firing_mode: str = "Default",
        spike_mode: str = "Uniform",
        thresholding_mode: str = "<=",
        spiking_mode: str = "lif",
        ttfs_cycle_schedule: str = "cascaded",
        compute_dtype: torch.dtype = COMPUTE_DTYPE,
    ):
        super().__init__()

        self.input_shape = input_shape
        self.ir_graph = ir_graph
        self.nodes = ir_graph.nodes
        self.output_sources = ir_graph.output_sources
        self.simulation_length = simulation_length
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()
        self.firing_mode = firing_mode
        self.spike_mode = spike_mode
        self.thresholding_mode = thresholding_mode
        self.spiking_mode = spiking_mode
        self.ttfs_cycle_schedule = ttfs_cycle_schedule
        self._compute_dtype = compute_dtype

        validate_spiking_init(
            firing_mode=firing_mode,
            spike_mode=spike_mode,
            thresholding_mode=thresholding_mode,
        )

        self._bank_params = nn.ParameterDict()
        for bank_id, bank in ir_graph.weight_banks.items():
            w = torch.tensor(bank.core_matrix.T, dtype=torch.float32)
            self._bank_params[str(bank_id)] = nn.Parameter(w, requires_grad=False)

        self.neural_core_ids: list[int] = []
        self._id_to_bank: Dict[int, tuple[str, tuple[int, int] | None]] = {}
        self._id_to_owned_param: Dict[int, int] = {}
        self.neural_core_params = nn.ParameterList()

        for node in self.nodes:
            if not isinstance(node, NeuralCore):
                continue
            self.neural_core_ids.append(node.id)
            if node.has_weight_bank():
                self._id_to_bank[node.id] = (
                    str(node.weight_bank_id),
                    node.weight_row_slice,
                )
                continue
            weight = torch.tensor(node.core_matrix.T, dtype=torch.float32)
            self.neural_core_params.append(nn.Parameter(weight, requires_grad=False))
            self._id_to_owned_param[node.id] = len(self.neural_core_params) - 1

        threshold_vals: list[float] = []
        self._threshold_idx_cache: Dict[int, int] = {}
        for node in self.nodes:
            if not isinstance(node, NeuralCore):
                continue
            t = node.threshold
            threshold_vals.append(float(t.item()) if hasattr(t, "item") else float(t))
            self._threshold_idx_cache[node.id] = len(threshold_vals) - 1
        self.register_buffer(
            "_thresholds_packed",
            torch.tensor(threshold_vals, dtype=COMPUTE_DTYPE)
            if threshold_vals
            else torch.empty(0, dtype=COMPUTE_DTYPE),
        )

        hw_chunks: list[torch.Tensor] = []
        self._hw_bias_spans: Dict[int, tuple[int, int]] = {}
        hw_offset = 0
        for node in self.nodes:
            if not isinstance(node, NeuralCore) or node.hardware_bias is None:
                continue
            hb = torch.tensor(node.hardware_bias, dtype=COMPUTE_DTYPE).reshape(-1)
            hw_chunks.append(hb)
            self._hw_bias_spans[node.id] = (hw_offset, hb.numel())
            hw_offset += hb.numel()
        self.register_buffer(
            "_hw_bias_packed",
            torch.cat(hw_chunks) if hw_chunks else torch.empty(0, dtype=COMPUTE_DTYPE),
        )

        self._id_to_out_dim: Dict[int, int] = {}
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                self._id_to_out_dim[node.id] = node.get_output_count() if node.core_matrix is not None else (
                    (node.weight_row_slice[1] - node.weight_row_slice[0]) if node.weight_row_slice else
                    ir_graph.weight_banks[node.weight_bank_id].core_matrix.shape[1]
                )

        self._sync_points = [i for i, n in enumerate(self.nodes) if isinstance(n, ComputeOp)]

        self._input_spans: Dict[int, list[IRSourceSpan]] = {}
        for node in self.nodes:
            flat = list(node.input_sources.flatten())
            self._input_spans[int(node.id)] = compress_ir_sources(flat)
        self._output_spans: list[IRSourceSpan] = compress_ir_sources(list(self.output_sources.flatten()))

        self._release_at_step: Dict[int, list[int]] = {}
        consumed_by_output: set[int] = set()
        for sp in self._output_spans:
            if sp.kind == "node":
                consumed_by_output.add(int(sp.src_node_id))
        last_reader: Dict[int, int] = {}
        for reader_idx, node in enumerate(self.nodes):
            for sp in self._input_spans[int(node.id)]:
                if sp.kind != "node":
                    continue
                last_reader[int(sp.src_node_id)] = reader_idx
        for src_id, idx in last_reader.items():
            if src_id in consumed_by_output:
                continue
            self._release_at_step.setdefault(idx, []).append(src_id)

        self._assert_mapping_contracts(ir_graph)

        from mimarsinan.mapping.support.activation_scales import (
            compute_node_input_scales,
            compute_node_output_scales,
        )

        self._ttfs_node_output_scale = compute_node_output_scales(ir_graph)
        self._ttfs_node_input_scale = compute_node_input_scales(ir_graph)

    def to_spikes(self, tensor: torch.Tensor, cycle: int) -> torch.Tensor:
        return spike_modes.to_spikes(
            tensor,
            cycle,
            simulation_length=self.simulation_length,
            spike_mode=self.spike_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LIF spike-train sim or analytical TTFS; ComputeOps are rate-space sync barriers.

        ``ttfs_cycle_based`` uses the analytical quantized reference here — by the
        ReLU↔TTFS equivalence its value equals the single-spike result; the genuine
        single-spike simulation lives in the nevresim / SANA-FE backends.
        """
        from mimarsinan.chip_simulation.spiking_semantics import (
            is_cascaded_ttfs,
            requires_ttfs_firing,
        )

        try:
            if is_cascaded_ttfs(self.spiking_mode, self.ttfs_cycle_schedule):
                return self._forward_lif(x)
            if requires_ttfs_firing(self.spiking_mode):
                return self._forward_ttfs(x)
            return self._forward_lif(x)
        finally:
            if isinstance(x, torch.Tensor) and x.is_cuda:
                torch.cuda.empty_cache()

    def get_core_spike_rates(self) -> list[float]:
        """Per-core mean spike rate after forward (graph order)."""
        if not hasattr(self, '_last_core_spike_counts'):
            raise RuntimeError("get_core_spike_rates called before forward pass")
        
        rates = []
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                rates.append(self._last_core_spike_counts.get(node.id, 0.0))
        return rates

    def get_cores(self) -> list[NeuralCore]:
        """Return list of neural cores in graph order."""
        return [n for n in self.nodes if isinstance(n, NeuralCore)]

    def refresh_thresholds(self) -> None:
        """Sync node.threshold edits into the packed threshold buffer."""
        for node in self.nodes:
            if isinstance(node, NeuralCore):
                t = node.threshold
                self._set_threshold(node, float(t.item()) if hasattr(t, "item") else float(t))
