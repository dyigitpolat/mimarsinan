from mimarsinan.code_generation.cpp_chip_model import *

import torch
import numpy as np

def is_off(idx): return idx == -1
def is_input(idx): return idx == -2
def is_always_on(idx): return idx == -3

from mimarsinan.mapping.core_packing import greedy_pack_softcores

def compact_soft_core_mapping(cores, output_sources):
    """Remove pruned rows and columns from each soft core using pruning maps and reindex spans.

    Uses pruned_row_mask and pruned_col_mask on each SoftCore (set from IR pruning);
    does not use parameter values to detect zeros. Modifies each core's core_matrix
    and axon_sources in place, and replaces output_sources so neuron indices match
    the compacted layout.

    Returns:
        reindex_maps: dict[int, dict[int, int]] — core_id → {old_neuron_idx: new_neuron_idx}
    """
    import os
    reindex_maps = {}  # core_id -> {old_neuron_idx: new_neuron_idx}
    n_compacted = 0
    n_skipped = 0

    for core in cores:
        mat = np.asarray(core.core_matrix, dtype=np.float64)
        n_axons, n_neurons = mat.shape
        pruned_row_mask = getattr(core, "pruned_row_mask", None)
        pruned_col_mask = getattr(core, "pruned_col_mask", None)

        if (
            pruned_row_mask is not None
            and pruned_col_mask is not None
            and len(pruned_row_mask) == n_axons
            and len(pruned_col_mask) == n_neurons
        ):
            keep_rows = [r for r in range(n_axons) if not pruned_row_mask[r]]
            keep_cols = [c for c in range(n_neurons) if not pruned_col_mask[c]]
            # Always preserve the always-on bias row (last axon, wired to is_always_on_ source).
            # It must never be pruned even if its weight values happen to be zero.
            if (
                core.axon_sources
                and getattr(core.axon_sources[-1], "is_always_on_", False)
                and (n_axons - 1) not in keep_rows
            ):
                keep_rows.append(n_axons - 1)
                keep_rows.sort()
        else:
            keep_rows = list(range(n_axons))
            keep_cols = list(range(n_neurons))

        if len(keep_rows) < n_axons or len(keep_cols) < n_neurons:
            n_compacted += 1
            if keep_rows and keep_cols:
                core.core_matrix = mat[np.ix_(keep_rows, keep_cols)].copy()
                core.axon_sources = [core.axon_sources[r] for r in keep_rows]
                # Compact hardware_bias alongside columns.
                if getattr(core, "hardware_bias", None) is not None:
                    core.hardware_bias = core.hardware_bias[keep_cols]
            else:
                core.core_matrix = np.zeros((1, 1), dtype=np.float64)
                core.axon_sources = [SpikeSource(-1, 0, False, True)]
                if getattr(core, "hardware_bias", None) is not None:
                    core.hardware_bias = np.zeros(1)
            remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)}
            reindex_maps[core.id] = remap
        else:
            n_skipped += 1
            reindex_maps[core.id] = {j: j for j in range(n_neurons)}

        core._axon_source_spans = None

    # Apply reindex to all axon_sources
    for core in cores:
        new_sources = []
        for s in core.axon_sources:
            if s.is_off_:
                new_sources.append(s)
            elif s.is_input_ or s.is_always_on_:
                new_sources.append(s)
            else:
                cid, nidx = int(s.core_), int(s.neuron_)
                if cid in reindex_maps and nidx in reindex_maps[cid]:
                    new_sources.append(SpikeSource(cid, reindex_maps[cid][nidx], False, False))
                else:
                    new_sources.append(SpikeSource(-1, 0, False, True))
        core.axon_sources = new_sources
        core._axon_source_spans = None

    # Rebuild output_sources: keep order, reindex or drop pruned neuron refs
    had_output_refs = any(
        not s.is_off_ for s in output_sources
    )
    new_out = []
    for s in output_sources:
        if s.is_off_:
            continue
        if s.is_input_ or s.is_always_on_:
            new_out.append(s)
        else:
            cid, nidx = int(s.core_), int(s.neuron_)
            if cid in reindex_maps and nidx in reindex_maps[cid]:
                new_out.append(SpikeSource(cid, reindex_maps[cid][nidx], False, False))
    if had_output_refs and len(new_out) == 0:
        raise ValueError(
            "compact_soft_core_mapping: all output_sources were dropped by compaction "
            "(every output neuron was pruned). At least one output must remain; check pruning masks "
            "and initial mask assignment (e.g. 1:1 vs tiled layer matching)."
        )
    output_sources.clear()
    output_sources.extend(new_out)

    return reindex_maps


class SoftCore:
    def __init__(
        self,
        core_matrix,
        axon_sources,
        id,
        activation_scale=torch.tensor(1.0),
        parameter_scale=torch.tensor(1.0),
        input_activation_scale=torch.tensor(1.0),
        *,
        name: str | None = None,
        psum_group_id: int | None = None,
        psum_role: str | None = None,
        coalescing_group_id: int | None = None,
        coalescing_role: str | None = None,
    ):
        self.core_matrix = core_matrix
        self.axon_sources = axon_sources

        self.id = id
        self.input_activation_scale = input_activation_scale
        self.activation_scale = activation_scale
        self.parameter_scale = parameter_scale
        self.threshold = 1.0

        # Optional debug/IR metadata
        self.name = name
        self.psum_group_id = psum_group_id
        self.psum_role = psum_role
        self.coalescing_group_id = coalescing_group_id
        self.coalescing_role = coalescing_role

        # Hardware-bias mode: when set, bias lives in a dedicated register
        # (not as an always-on axon row).  Shape: (neurons,).
        self.hardware_bias = None

        self.latency = None
        self._axon_source_spans = None

        # Neuron-splitting metadata: when a soft core is split across multiple
        # hardware cores, each fragment records its offset within the original
        # neuron range so that neuron_mapping references the correct indices.
        self.neuron_offset_in_original = 0
        self.split_group_id = None          # unique id linking fragments
        self.split_fragment_index = None    # 0, 1, 2, ... per fragment
        self.split_original_neurons = None  # total neurons before splitting

    def get_input_count(self):
        return len(self.axon_sources)
    
    def get_output_count(self):
        return self.core_matrix.shape[-1]

    def get_axon_source_spans(self):
        """
        Return a range-compressed representation of axon_sources suitable for fast simulation.

        Note: This is a cached *view*; it must be invalidated if axon_sources are mutated.
        SoftCore axon_sources are typically immutable after construction.
        """
        if self._axon_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans
    
class HardCore:
    def __init__(self, axons_per_core, neurons_per_core, has_bias_capability=True):
        self.axons_per_core = axons_per_core
        self.neurons_per_core = neurons_per_core
        self.has_bias_capability = has_bias_capability

        self.core_matrix = np.zeros((axons_per_core, neurons_per_core))
        self.axon_sources = []

        self.available_axons = axons_per_core
        self.available_neurons = neurons_per_core

        self.input_activation_scale = None
        self.activation_scale = None
        self.parameter_scale = None
        self.threshold = None
        self.latency = None

        # Hardware-bias: per-neuron bias stored in dedicated register.
        # None means legacy (always-on axon row or no bias).
        self.hardware_bias = None

        self.unusable_space = 0
        self._axon_source_spans = None

    def get_input_count(self):
        return self.axons_per_core
    
    def get_output_count(self):
        return self.neurons_per_core

    def add_softcore(self, softcore):
        assert self.available_axons >= softcore.get_input_count() 
        assert self.available_neurons >= softcore.get_output_count()            

        axon_offset = self.axons_per_core - self.available_axons
        neuron_offset = self.neurons_per_core - self.available_neurons
        
        self.core_matrix[
            axon_offset : axon_offset+softcore.get_input_count(), 
            neuron_offset : neuron_offset+softcore.get_output_count()] \
                = softcore.core_matrix

        # print(f"prev threshold: {self.threshold}")
        # print(f"softcore threshold: {softcore.threshold}")
        # self.threshold = (self.threshold * neuron_offset + softcore.threshold * softcore.get_output_count()) / (neuron_offset + softcore.get_output_count())
        # print(f"new threshold: {self.threshold}")
        
        self.axon_sources.extend(softcore.axon_sources)
        self._axon_source_spans = None  # invalidate cached spans

        self.available_axons -= softcore.get_input_count()
        self.available_neurons -= softcore.get_output_count()

        if self.threshold is None:
            self.threshold = softcore.threshold

        if self.input_activation_scale is None:
            self.input_activation_scale = softcore.input_activation_scale

        if self.activation_scale is None:
            self.activation_scale = softcore.activation_scale

        if self.parameter_scale is None:
            self.parameter_scale = softcore.parameter_scale
        
        if self.latency is None:
            self.latency = softcore.latency

        # Copy hardware_bias from SoftCore into the correct neuron slice.
        if getattr(softcore, "hardware_bias", None) is not None:
            if self.hardware_bias is None:
                self.hardware_bias = np.zeros(self.neurons_per_core)
            self.hardware_bias[neuron_offset:neuron_offset + softcore.get_output_count()] = softcore.hardware_bias

        self.unusable_space += \
            (neuron_offset * softcore.get_input_count()) + \
            (axon_offset * softcore.get_output_count())

    def get_axon_source_spans(self):
        """
        Range-compressed view of axon_sources for fast simulation.

        HardCore axon_sources are built incrementally during packing; this view is cached and
        automatically invalidated on add_softcore().
        """
        if self._axon_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans

class HardCoreMapping:
    def __init__(self, chip_cores):
        self.unused_cores = chip_cores

        self.cores = []
        self.output_sources = []
        self.neuron_mapping = {}
        # hard→soft traceability: list indexed by hard core index; each element is a list of
        # placement dicts: {"ir_node_id", "axon_offset", "neuron_offset", "axons", "neurons"}
        self.soft_core_placements_per_hard_core = []

        self.unusable_space = 0
        self._output_source_spans = None

    @property
    def axons_per_core(self) -> int:
        """Max axon dimension across all used cores (for nevresim uniform padding)."""
        if not self.cores:
            return 0
        return max(int(hc.axons_per_core) for hc in self.cores)

    @property
    def neurons_per_core(self) -> int:
        """Max neuron dimension across all used cores (for nevresim uniform padding)."""
        if not self.cores:
            return 0
        return max(int(hc.neurons_per_core) for hc in self.cores)

    def merge_softcore_into(self, target_core_idx: int, hardcore, softcore):
        axon_offset = hardcore.axons_per_core - hardcore.available_axons
        neuron_offset = hardcore.neurons_per_core - hardcore.available_neurons
        axons = softcore.get_input_count()
        neurons = softcore.get_output_count()

        prev_output_count = hardcore.neurons_per_core - hardcore.available_neurons
        hardcore.add_softcore(softcore)

        while len(self.soft_core_placements_per_hard_core) <= target_core_idx:
            self.soft_core_placements_per_hard_core.append([])
        placement = {
            "ir_node_id": softcore.id,
            "axon_offset": axon_offset,
            "neuron_offset": neuron_offset,
            "axons": axons,
            "neurons": neurons,
            "coalescing_group_id": getattr(softcore, "coalescing_group_id", None),
            "coalescing_role": getattr(softcore, "coalescing_role", None),
        }
        if getattr(softcore, "split_group_id", None) is not None:
            orig_offset = getattr(softcore, "neuron_offset_in_original", 0)
            placement["split_group_id"] = softcore.split_group_id
            placement["split_fragment_index"] = softcore.split_fragment_index
            placement["split_original_neurons"] = softcore.split_original_neurons
            placement["neuron_range_in_original"] = [orig_offset, orig_offset + neurons]
        self.soft_core_placements_per_hard_core[target_core_idx].append(placement)

        orig_neuron_offset = getattr(softcore, "neuron_offset_in_original", 0)
        for local_idx in range(softcore.get_output_count()):
            original_neuron_idx = local_idx + orig_neuron_offset
            target_neuron_idx = prev_output_count + local_idx
            self.neuron_mapping[(softcore.id, original_neuron_idx)] = \
                (target_core_idx, target_neuron_idx)
                
    def map(self, softcore_mapping, *, allow_neuron_splitting: bool = False):
        def is_mapping_possible(core, hardcore):
            tolerance = 0.1
            if hardcore.threshold is not None:
                threshold_diff = abs(core.threshold - hardcore.threshold)
                diff_rate = threshold_diff / (hardcore.threshold + 1)
            else:
                diff_rate = 0.0

            # Latency check: both must have same latency value
            # If hardcore is empty (latency=None), any core can start it
            # If hardcore has a latency, core must have the SAME latency (not None)
            if hardcore.latency is None:
                latency_ok = True  # Empty hardcore can accept any core
            else:
                # Hardcore has a latency - core must match exactly
                latency_ok = (core.latency is not None and core.latency == hardcore.latency)

            return \
                diff_rate <= tolerance and \
                core.get_input_count() <= hardcore.available_axons and \
                core.get_output_count() <= hardcore.available_neurons and \
                latency_ok

        unmapped_cores = [core for core in softcore_mapping.cores]

        def place(core_idx: int, hardcore, softcore):
            self.merge_softcore_into(core_idx, hardcore, softcore)

        def fuse_hardcores(hcs):
            if not hcs:
                raise ValueError("Cannot fuse empty list of hardcores")
            first = hcs[0]
            fused = HardCore(
                axons_per_core=sum(hc.axons_per_core for hc in hcs),
                neurons_per_core=first.neurons_per_core,
                has_bias_capability=first.has_bias_capability
            )
            fused.threshold = first.threshold
            fused.input_activation_scale = first.input_activation_scale
            fused.activation_scale = first.activation_scale
            fused.parameter_scale = first.parameter_scale
            fused.latency = first.latency
            # All fused components share the same neuron layout; copy hardware_bias.
            fused.hardware_bias = first.hardware_bias
            # Record fusion structure for GUI (boundaries, fused badge).
            fused.fused_component_axons = [hc.axons_per_core for hc in hcs]
            return fused

        # --- Neuron splitting callback ---
        _split_counter = [0]

        def split_softcore_fn(core, available_neurons):
            group_id = _split_counter[0]
            _split_counter[0] += 1
            original_neurons = core.get_output_count()
            orig_offset = getattr(core, "neuron_offset_in_original", 0)

            # Fragment 1: first `available_neurons` columns
            frag1 = SoftCore(
                core_matrix=core.core_matrix[:, :available_neurons].copy(),
                axon_sources=list(core.axon_sources),
                id=core.id,
                activation_scale=core.activation_scale,
                parameter_scale=core.parameter_scale,
                input_activation_scale=core.input_activation_scale,
                name=core.name,
                psum_group_id=core.psum_group_id,
                psum_role=core.psum_role,
            )
            frag1.threshold = core.threshold
            frag1.latency = core.latency
            frag1.neuron_offset_in_original = orig_offset
            frag1.split_group_id = group_id
            frag1.split_fragment_index = 0
            frag1.split_original_neurons = original_neurons
            if getattr(core, "hardware_bias", None) is not None:
                frag1.hardware_bias = core.hardware_bias[:available_neurons].copy()

            # Fragment 2: remaining columns
            frag2 = SoftCore(
                core_matrix=core.core_matrix[:, available_neurons:].copy(),
                axon_sources=list(core.axon_sources),
                id=core.id,
                activation_scale=core.activation_scale,
                parameter_scale=core.parameter_scale,
                input_activation_scale=core.input_activation_scale,
                name=core.name,
                psum_group_id=core.psum_group_id,
                psum_role=core.psum_role,
            )
            frag2.threshold = core.threshold
            frag2.latency = core.latency
            frag2.neuron_offset_in_original = orig_offset + available_neurons
            frag2.split_group_id = group_id
            frag2.split_fragment_index = 1
            frag2.split_original_neurons = original_neurons
            if getattr(core, "hardware_bias", None) is not None:
                frag2.hardware_bias = core.hardware_bias[available_neurons:].copy()

            return frag1, frag2

        from mimarsinan.mapping.core_packing import greedy_pack_softcores
        greedy_pack_softcores(
            softcores=unmapped_cores,
            used_hardcores=self.cores,
            unused_hardcores=self.unused_cores,
            is_mapping_possible=is_mapping_possible,
            place=place,
            fuse_hardcores=fuse_hardcores,
            split_softcore=split_softcore_fn if allow_neuron_splitting else None,
        )

        def remap_sources(sources):
            for source in sources:
                if source.is_off_ or source.is_input_ or source.is_always_on_: continue
                source.core_, source.neuron_ = \
                    self.neuron_mapping[(source.core_, source.neuron_)]
            
        for hardcore in self.cores:
            remap_sources(hardcore.axon_sources)
            hardcore._axon_source_spans = None
                
        self.output_sources = np.array(softcore_mapping.output_sources)
        remap_sources(self.output_sources)
        self._output_source_spans = None

        for hardcore in self.cores:
            axon_count = len(hardcore.axon_sources)
            for _ in range(hardcore.axons_per_core - axon_count):
                hardcore.axon_sources.append(
                    SpikeSource(-1, 0, is_input=False, is_off=True))
            hardcore._axon_source_spans = None

    def get_output_source_spans(self):
        """
        Range-compressed view of the final mapping outputs.
        """
        if self._output_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._output_source_spans = compress_spike_sources(self.output_sources.flatten().tolist())
        return self._output_source_spans
                    