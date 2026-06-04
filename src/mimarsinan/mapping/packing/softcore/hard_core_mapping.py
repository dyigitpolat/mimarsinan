"""HardCoreMapping: pack SoftCoreMapping instances onto chip HardCores."""

from mimarsinan.code_generation.cpp_chip_model import SpikeSource

import numpy as np

from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.packing.softcore.soft_core import SoftCore


class RuntimeMaterializer:
    """Weight-bearing :class:`~mimarsinan.mapping.packing.placement_engine.Materializer`.

    Places real ``SoftCore``s onto ``HardCore``s (recording per-neuron
    ``neuron_mapping`` and placement provenance via
    ``HardCoreMapping.merge_softcore_into``), fuses physical cores into wider
    ``HardCore``s, and splits softcores along the neuron dimension by slicing
    weight / bias / bank state.
    """

    def __init__(self, hard_core_mapping: "HardCoreMapping") -> None:
        self._hcm = hard_core_mapping
        self._split_counter = 0

    @staticmethod
    def is_mapping_possible(hardcore, softcore) -> bool:
        from mimarsinan.mapping.packing.core_packing import canonical_is_mapping_possible

        return canonical_is_mapping_possible(hardcore, softcore)

    def place(self, core_idx: int, hardcore, softcore) -> None:
        self._hcm.merge_softcore_into(core_idx, hardcore, softcore)

    def fuse_hardcores(self, hcs):
        from mimarsinan.mapping.packing.core_packing import canonical_fuse_hardcores

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

    def _build_real_fragment(
        self,
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
        frag.perceptron_index = getattr(parent, "perceptron_index", None)
        frag.perceptron_output_slice = getattr(parent, "perceptron_output_slice", None)
        if hardware_bias_slice is not None:
            frag.hardware_bias = hardware_bias_slice
        if bank_neuron_slice is not None:
            frag.weight_bank_id = parent.weight_bank_id
            frag.bank_axon_slice = parent.bank_axon_slice
            frag.bank_neuron_slice = bank_neuron_slice
            frag.bank_includes_bias_row = parent.bank_includes_bias_row
        return frag

    def _make_real_fragments(self, *, softcore, first_neurons, remaining_neurons):
        group_id = self._split_counter
        self._split_counter += 1
        total = first_neurons + remaining_neurons
        hb = getattr(softcore, "hardware_bias", None)
        bns = None
        if getattr(softcore, "weight_bank_id", None) is not None:
            bns_parent = getattr(softcore, "bank_neuron_slice", None) or (0, total)
            bns = (
                (int(bns_parent[0]), int(bns_parent[0] + first_neurons)),
                (int(bns_parent[0] + first_neurons), int(bns_parent[1])),
            )
        frag1 = self._build_real_fragment(
            softcore,
            matrix_slice=softcore.core_matrix[:, :first_neurons].copy(),
            hardware_bias_slice=hb[:first_neurons].copy() if hb is not None else None,
            bank_neuron_slice=bns[0] if bns is not None else None,
            offset_delta=0,
            fragment_index=0,
            group_id=group_id,
            original_neurons=total,
        )
        frag2 = self._build_real_fragment(
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

    def split_softcore(self, softcore, available_neurons):
        from mimarsinan.mapping.packing.core_packing import canonical_split_softcore

        return canonical_split_softcore(
            softcore, available_neurons, make_fragments=self._make_real_fragments,
        )


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
            "perceptron_index": getattr(softcore, "perceptron_index", None),
            "perceptron_output_slice": getattr(softcore, "perceptron_output_slice", None),
            "psum_role": getattr(softcore, "psum_role", None),
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

        unmapped_cores = [core for core in softcore_mapping.cores]

        from mimarsinan.mapping.packing.placement_engine import run_placement

        run_placement(
            softcores=unmapped_cores,
            used_hardcores=self.cores,
            unused_hardcores=self.unused_cores,
            materializer=RuntimeMaterializer(self),
            allow_neuron_splitting=allow_neuron_splitting,
        )

        def remap_sources(sources):
            for source in sources:
                if source.is_off_ or source.is_input_ or source.is_always_on_:
                    continue
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
            from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
            self._output_source_spans = compress_spike_sources(self.output_sources.flatten().tolist())
        return self._output_source_spans
