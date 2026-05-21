from mimarsinan.code_generation.cpp_chip_model import *

import torch
import numpy as np

def is_off(idx): return idx == -1
def is_input(idx): return idx == -2
def is_always_on(idx): return idx == -3

from mimarsinan.mapping.core_packing import greedy_pack_softcores

def compact_soft_core_mapping(cores, output_sources):
    """Compact soft cores from IR pruning masks; return core_id → neuron reindex maps."""
    reindex_maps = {}
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
            # Never prune the always-on bias row (last axon).
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
            # Bank-backed: IR pruning must already match bank shape; else drop bank ref.
            if getattr(core, "weight_bank_id", None) is not None:
                core.weight_bank_id = None
                core.bank_axon_slice = None
                core.bank_neuron_slice = None
                core.bank_includes_bias_row = False
            from mimarsinan.mapping.pruning_apply import compact_hardware_bias_columns

            if keep_rows and keep_cols:
                core.core_matrix = mat[np.ix_(keep_rows, keep_cols)].copy()
                core.axon_sources = [core.axon_sources[r] for r in keep_rows]
                core.hardware_bias = compact_hardware_bias_columns(
                    core.hardware_bias, keep_cols
                )
            elif keep_cols:
                # BIAS_ONLY core: every axon is dead but some neurons spike
                # from ``hardware_bias`` alone. Collapse to a single
                # OFF-source axon while preserving the live bias-driven
                # columns; collapsing to (1, 1) would silently delete live
                # neurons whose consumers still reference them.
                core.core_matrix = np.zeros((1, len(keep_cols)), dtype=np.float64)
                core.axon_sources = [SpikeSource(-1, 0, False, True)]
                core.hardware_bias = compact_hardware_bias_columns(
                    core.hardware_bias, keep_cols
                )
            else:
                # No surviving columns -> the core has no live neuron and
                # would have been deleted by ``compute_liveness +
                # IRGraph.remove_nodes`` in ``prune_ir_graph``. Reaching
                # this point means an upstream stage produced a dead-math
                # core that was not classified DEAD; fail loudly so the
                # regression is visible.
                raise AssertionError(
                    f"compact_soft_core_mapping: SoftCore id={core.id} has "
                    f"no surviving neurons (keep_cols={keep_cols}). "
                    "Liveness analysis should have removed it via "
                    "IRGraph.remove_nodes before soft-core mapping. "
                    "This indicates a regression in prune_ir_graph or a "
                    "stale pickle that bypassed the liveness pass."
                )
            remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_cols)}
            reindex_maps[core.id] = remap
        else:
            n_skipped += 1
            reindex_maps[core.id] = {j: j for j in range(n_neurons)}

        core._axon_source_spans = None

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
        threshold_group_id: int | None = None,
        weight_bank_id: int | None = None,
        bank_axon_slice: tuple[int, int] | None = None,
        bank_neuron_slice: tuple[int, int] | None = None,
        bank_includes_bias_row: bool = False,
    ):
        self.core_matrix = core_matrix
        self.axon_sources = axon_sources

        self.id = id
        self.input_activation_scale = input_activation_scale
        self.activation_scale = activation_scale
        self.parameter_scale = parameter_scale
        self.threshold = 1.0

        # Pack-time: share hardcore only when threshold_group_id matches; None → unique group.
        self.threshold_group_id = threshold_group_id

        self.name = name
        self.psum_group_id = psum_group_id
        self.psum_role = psum_role
        self.coalescing_group_id = coalescing_group_id
        self.coalescing_role = coalescing_role

        self.weight_bank_id = weight_bank_id
        self.bank_axon_slice = bank_axon_slice
        self.bank_neuron_slice = bank_neuron_slice
        self.bank_includes_bias_row = bool(bank_includes_bias_row)

        self.hardware_bias = None

        self.latency = None
        self._axon_source_spans = None

        self.neuron_offset_in_original = 0
        self.split_group_id = None
        self.split_fragment_index = None
        self.split_original_neurons = None

    def get_input_count(self):
        return len(self.axon_sources)
    
    def get_output_count(self):
        return self.core_matrix.shape[-1]

    def get_axon_source_spans(self):
        """Cached range-compressed axon_sources; invalidate if axon_sources mutate."""
        if self._axon_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans
    
class HardCore:
    def __init__(self, axons_per_core, neurons_per_core, has_bias_capability=True):
        self.axons_per_core = axons_per_core
        self.neurons_per_core = neurons_per_core
        self.has_bias_capability = has_bias_capability

        # Lazy zero matrix on first add_softcore (avoids eager float64 per hw core).
        self.core_matrix = None
        self.axon_sources = []

        self.available_axons = axons_per_core
        self.available_neurons = neurons_per_core

        self.input_activation_scale = None
        self.activation_scale = None
        self.parameter_scale = None
        self.threshold = None
        self.threshold_group_id: int | None = None
        self.latency = None

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

        if self.core_matrix is None:
            sc_dtype = getattr(softcore.core_matrix, "dtype", None)
            dtype = sc_dtype if sc_dtype is not None else np.float64
            self.core_matrix = np.zeros(
                (self.axons_per_core, self.neurons_per_core), dtype=dtype,
            )

        self.core_matrix[
            axon_offset : axon_offset+softcore.get_input_count(),
            neuron_offset : neuron_offset+softcore.get_output_count()] \
                = softcore.core_matrix

        self.axon_sources.extend(softcore.axon_sources)
        self._axon_source_spans = None

        self.available_axons -= softcore.get_input_count()
        self.available_neurons -= softcore.get_output_count()

        if self.threshold is None:
            self.threshold = softcore.threshold

        if self.threshold_group_id is None:
            tg = getattr(softcore, "threshold_group_id", None)
            self.threshold_group_id = (
                int(tg) if tg is not None else -(int(softcore.id) + 1)
            )

        if self.input_activation_scale is None:
            self.input_activation_scale = softcore.input_activation_scale

        if self.activation_scale is None:
            self.activation_scale = softcore.activation_scale

        if self.parameter_scale is None:
            self.parameter_scale = softcore.parameter_scale
        
        if self.latency is None:
            self.latency = softcore.latency

        if getattr(softcore, "hardware_bias", None) is not None:
            if self.hardware_bias is None:
                self.hardware_bias = np.zeros(self.neurons_per_core)
            self.hardware_bias[neuron_offset:neuron_offset + softcore.get_output_count()] = softcore.hardware_bias

        self.unusable_space += \
            (neuron_offset * softcore.get_input_count()) + \
            (axon_offset * softcore.get_output_count())

    def get_axon_source_spans(self):
        """Cached range-compressed axon_sources; invalidated on add_softcore."""
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
        self.soft_core_placements_per_hard_core = []
        self.weight_banks: dict[int, "object"] = {}

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
        bank_id = getattr(softcore, "weight_bank_id", None)
        if bank_id is not None:
            placement["weight_bank_id"] = int(bank_id)
            bas = getattr(softcore, "bank_axon_slice", None)
            bns = getattr(softcore, "bank_neuron_slice", None)
            placement["bank_axon_range"] = (
                (int(bas[0]), int(bas[1])) if bas is not None else (0, int(axons))
            )
            placement["bank_neuron_range"] = (
                (int(bns[0]), int(bns[1])) if bns is not None else (0, int(neurons))
            )
            placement["bank_includes_bias_row"] = bool(
                getattr(softcore, "bank_includes_bias_row", False)
            )
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
        banks = getattr(softcore_mapping, "weight_banks", None)
        if banks:
            self.weight_banks = dict(banks)

        from mimarsinan.mapping.core_packing import canonical_is_mapping_possible
        is_mapping_possible = canonical_is_mapping_possible

        unmapped_cores = [core for core in softcore_mapping.cores]

        def place(core_idx: int, hardcore, softcore):
            self.merge_softcore_into(core_idx, hardcore, softcore)

        from mimarsinan.mapping.core_packing import canonical_fuse_hardcores

        def _real_fuse(hcs):
            def _mk(*, axons, neurons, template, components):
                fused = HardCore(
                    axons_per_core=int(axons),
                    neurons_per_core=int(neurons),
                    has_bias_capability=template.has_bias_capability,
                )
                fused.threshold = template.threshold
                fused.input_activation_scale = template.input_activation_scale
                fused.activation_scale = template.activation_scale
                fused.parameter_scale = template.parameter_scale
                fused.latency = template.latency
                fused.hardware_bias = template.hardware_bias
                fused.fused_component_axons = [
                    int(hc.axons_per_core) for hc in components
                ]
                return fused
            return canonical_fuse_hardcores(hcs, make_fused=_mk)

        fuse_hardcores = _real_fuse

        from mimarsinan.mapping.core_packing import canonical_split_softcore

        _split_counter = [0]

        def _build_real_fragment(
            parent,
            *,
            matrix_slice,
            hardware_bias_slice,
            bank_neuron_slice,
            offset_delta,
            fragment_index,
            group_id,
            original_neurons,
        ):
            frag = SoftCore(
                core_matrix=matrix_slice,
                axon_sources=[
                    SpikeSource(s.core_, s.neuron_, s.is_input_, s.is_off_, s.is_always_on_)
                    for s in parent.axon_sources
                ],
                id=parent.id,
                activation_scale=parent.activation_scale,
                parameter_scale=parent.parameter_scale,
                input_activation_scale=parent.input_activation_scale,
                name=parent.name,
                psum_group_id=parent.psum_group_id,
                psum_role=parent.psum_role,
            )
            frag.threshold = parent.threshold
            frag.latency = parent.latency
            frag.neuron_offset_in_original = (
                getattr(parent, "neuron_offset_in_original", 0) + offset_delta
            )
            frag.split_group_id = group_id
            frag.split_fragment_index = fragment_index
            frag.split_original_neurons = original_neurons
            if hardware_bias_slice is not None:
                frag.hardware_bias = hardware_bias_slice
            if bank_neuron_slice is not None:
                frag.weight_bank_id = parent.weight_bank_id
                frag.bank_axon_slice = parent.bank_axon_slice
                frag.bank_neuron_slice = bank_neuron_slice
                frag.bank_includes_bias_row = parent.bank_includes_bias_row
            return frag

        def _make_real_fragments(*, softcore, first_neurons, remaining_neurons):
            group_id = _split_counter[0]
            _split_counter[0] += 1
            total = first_neurons + remaining_neurons
            hb = getattr(softcore, "hardware_bias", None)
            bns = None
            if getattr(softcore, "weight_bank_id", None) is not None:
                bns_parent = getattr(softcore, "bank_neuron_slice", None) or (0, total)
                bns = (
                    (int(bns_parent[0]), int(bns_parent[0] + first_neurons)),
                    (int(bns_parent[0] + first_neurons), int(bns_parent[1])),
                )
            frag1 = _build_real_fragment(
                softcore,
                matrix_slice=softcore.core_matrix[:, :first_neurons].copy(),
                hardware_bias_slice=hb[:first_neurons].copy() if hb is not None else None,
                bank_neuron_slice=bns[0] if bns is not None else None,
                offset_delta=0,
                fragment_index=0,
                group_id=group_id,
                original_neurons=total,
            )
            frag2 = _build_real_fragment(
                softcore,
                matrix_slice=softcore.core_matrix[:, first_neurons:].copy(),
                hardware_bias_slice=hb[first_neurons:].copy() if hb is not None else None,
                bank_neuron_slice=bns[1] if bns is not None else None,
                offset_delta=first_neurons,
                fragment_index=1,
                group_id=group_id,
                original_neurons=total,
            )
            return frag1, frag2

        def split_softcore_fn(core, available_neurons):
            return canonical_split_softcore(
                core, available_neurons, make_fragments=_make_real_fragments,
            )

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
        """Cached range-compressed output_sources."""
        if self._output_source_spans is None:
            from mimarsinan.mapping.spike_source_spans import compress_spike_sources
            self._output_source_spans = compress_spike_sources(self.output_sources.flatten().tolist())
        return self._output_source_spans
                    