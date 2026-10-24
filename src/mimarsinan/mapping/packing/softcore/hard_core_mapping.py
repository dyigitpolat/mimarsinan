"""HardCoreMapping: pack SoftCoreMapping instances onto chip HardCores."""

from mimarsinan.code_generation.cpp_chip_model import SpikeSource

import numpy as np

from mimarsinan.mapping.packing.softcore.fragments import make_real_fragments
from mimarsinan.mapping.packing.softcore.hard_core import HardCore
from mimarsinan.mapping.platform.platform_constraints import (
    PER_AXON,
    SIGN_GRANULARITY_FIELD,
    pool_sign_granularity,
)
from mimarsinan.mapping.support.spike_source_columns import PlacementNeuronMap, SpikeSourceColumns
from mimarsinan.mapping.support.spike_source_spans import (
    compress_spike_sources,
    pack_output_sources_state,
    unpack_output_sources_state,
)


class RuntimeMaterializer:
    """Places, fuses and neuron-splits real SoftCores onto HardCores."""

    def __init__(self, hard_core_mapping: "HardCoreMapping") -> None:
        self._hcm = hard_core_mapping
        self._split_counter = 0

    @staticmethod
    def is_mapping_possible(softcore, hardcore) -> bool:
        from mimarsinan.mapping.packing.core_packing import canonical_is_mapping_possible

        return canonical_is_mapping_possible(softcore, hardcore)

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
            if getattr(template, "readout_law", None) is not None:
                fused.readout_law = template.readout_law
            # A fused core is per-axon when any component is (the pool's fold): the per-axon row gate then checks
            # every component's rows, whichever core type the pool declared first.
            declared = [{SIGN_GRANULARITY_FIELD: g} for hc in components
                        if (g := getattr(hc, "weight_sign_granularity", None)) is not None]
            if declared and pool_sign_granularity(declared) == PER_AXON:
                fused.weight_sign_granularity = PER_AXON
            fused.fused_component_axons = [
                int(hc.axons_per_core) for hc in components
            ]
            return fused

        return canonical_fuse_hardcores(hcs, make_fused=_mk)

    def _make_real_fragments(self, *, softcore, first_neurons, remaining_neurons):
        group_id = self._split_counter
        self._split_counter += 1
        return make_real_fragments(
            softcore=softcore, first_neurons=first_neurons,
            remaining_neurons=remaining_neurons, group_id=group_id,
        )

    def split_softcore(self, softcore, available_neurons):
        from mimarsinan.mapping.packing.core_packing import canonical_split_softcore

        return canonical_split_softcore(
            softcore, available_neurons, make_fragments=self._make_real_fragments,
        )


class HardCoreMapping:
    def __init__(self, chip_cores):
        self.unused_cores = chip_cores

        self.cores = []
        self.output_sources: "np.ndarray | list | SpikeSourceColumns" = []
        # (soft core id, neuron) -> (hard core index, neuron), held per placement (coverage W9).
        self.neuron_mapping = PlacementNeuronMap()
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
        if isinstance(self.neuron_mapping, PlacementNeuronMap):
            self.neuron_mapping.place(softcore.id, orig_neuron_offset, softcore.get_output_count(),
                                      target_core_idx, prev_output_count)
            return
        for local_idx in range(softcore.get_output_count()):  # a mapping unpickled from a per-neuron dict
            original_neuron_idx = local_idx + orig_neuron_offset
            target_neuron_idx = prev_output_count + local_idx
            self.neuron_mapping[(softcore.id, original_neuron_idx)] = \
                (target_core_idx, target_neuron_idx)

    def map(self, softcore_mapping, *, allow_neuron_splitting: bool = False,
            allow_coalescing: bool = True):
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
            allow_coalescing=allow_coalescing,
        )

        self._finalize_sources(softcore_mapping)

    def map_identity(self, softcore_mapping):
        """1:1 SoftCore->HardCore placement; pure IR semantics (rung-2 gate)."""
        banks = getattr(softcore_mapping, "weight_banks", None)
        if banks:
            self.weight_banks = dict(banks)

        for core_idx, softcore in enumerate(softcore_mapping.cores):
            hardcore = HardCore(
                axons_per_core=softcore.get_input_count(),
                neurons_per_core=softcore.get_output_count(),
                has_bias_capability=getattr(softcore, "hardware_bias", None) is not None,
            )
            self.cores.append(hardcore)
            self.merge_softcore_into(core_idx, hardcore, softcore)

        self._finalize_sources(softcore_mapping)

    def _finalize_sources(self, softcore_mapping):
        """Remap placed sources to (core, neuron) and pad unused axons off."""

        def remap_sources(sources):
            for source in sources:
                if source.is_off_ or source.is_input_ or source.is_always_on_:
                    continue
                source.core_, source.neuron_ = \
                    self.neuron_mapping[(source.core_, source.neuron_)]

        for hardcore in self.cores:
            if isinstance(hardcore.axon_sources, SpikeSourceColumns):
                hardcore.axon_sources = hardcore.axon_sources.remapped(self._lookup)
            else:
                remap_sources(hardcore.axon_sources)
            hardcore._axon_source_spans = None

        if isinstance(softcore_mapping.output_sources, SpikeSourceColumns):
            self.output_sources = softcore_mapping.output_sources.remapped(self._lookup)
        else:
            self.output_sources = np.array(softcore_mapping.output_sources)
            remap_sources(self.output_sources)
        self._output_source_spans = None

        for hardcore in self.cores:
            if isinstance(hardcore.axon_sources, SpikeSourceColumns):
                hardcore.axon_sources = hardcore.axon_sources.padded(hardcore.axons_per_core)
                hardcore._axon_source_spans = None
                continue
            axon_count = len(hardcore.axon_sources)
            for _ in range(hardcore.axons_per_core - axon_count):
                hardcore.axon_sources.append(
                    SpikeSource(-1, 0, is_input=False, is_off=True))
            hardcore._axon_source_spans = None

    def _lookup(self, cores, neurons):
        """``remapped``'s lookup through ``neuron_mapping``: every entry mapped, else the first missing key's
        ``KeyError``, as the per-source lookup raises it."""
        if isinstance(self.neuron_mapping, PlacementNeuronMap):
            return self.neuron_mapping.lookup(cores, neurons)
        pairs = [self.neuron_mapping[key] for key in zip(cores.tolist(), neurons.tolist())]
        return (np.array([p[0] for p in pairs], dtype=np.int64), np.array([p[1] for p in pairs], dtype=np.int64),
                np.ones(len(pairs), dtype=bool))

    def __getstate__(self) -> dict:
        """Pickle output_sources range-compressed; drop the span cache."""
        return pack_output_sources_state(self.__dict__)

    def __setstate__(self, state: dict) -> None:
        """Decode packed output_sources; legacy states load unchanged."""
        self.__dict__.update(unpack_output_sources_state(state))

    def get_output_source_spans(self):
        """Cached range-compressed output_sources."""
        if self._output_source_spans is None:
            sources = self.output_sources
            self._output_source_spans = compress_spike_sources(
                sources if isinstance(sources, SpikeSourceColumns) else np.asarray(sources).flatten().tolist()
            )
        return self._output_source_spans
