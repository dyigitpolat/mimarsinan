"""HardCore container for chip-level neuron/axon packing."""

from typing import Any

import numpy as np

from mimarsinan.mapping.packing.placement_cost import note_resident_bank
from mimarsinan.mapping.packing.softcore.matrix_placement import (
    MatrixPlacement,
    core_matrix_content_key,
    core_matrix_dtype,
    core_matrix_payloads,
    resolve_core_matrix,
)
from mimarsinan.mapping.platform.core_residency import (
    ALL_SINGLETON_NAMES,
    adopt_or_check,
    ungrouped_fallback_id,
)


class HardCore:
    def __init__(self, axons_per_core, neurons_per_core, has_bias_capability=True):
        self.axons_per_core = axons_per_core
        self.neurons_per_core = neurons_per_core
        self.has_bias_capability = has_bias_capability

        # ``core_matrix`` is OWNED-DENSE only (external writes / legacy
        # pickles) and takes precedence; packed cores carry placements.
        self.core_matrix: "np.ndarray | None" = None
        self.matrix_placements: list[MatrixPlacement] = []
        self.axon_sources = []

        self.available_axons = axons_per_core
        self.available_neurons = neurons_per_core

        # Which of the singleton values THIS target constrains; the rest are stored per neuron.
        self.constrained_properties: frozenset[str] = ALL_SINGLETON_NAMES
        # Singleton values a hardware core stores once; see platform/core_residency.py.
        self.input_activation_scale: Any = None
        self.boundary_grid: Any = None
        self.activation_scale: Any = None
        self.parameter_scale: Any = None
        self.threshold: float | None = None
        self.residency_class_id: int | None = None
        self.latency = None

        self.hardware_bias = None

        # Axon widths of the original cores merged into this one by fusion; None when not fused.
        self.fused_component_axons: list[int] | None = None

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

        placement = self._softcore_placement(softcore, axon_offset, neuron_offset)
        if self.core_matrix is not None:
            # Legacy owned-dense core: paste eagerly (pre-descriptor writers).
            self.core_matrix[
                axon_offset : axon_offset+softcore.get_input_count(),
                neuron_offset : neuron_offset+softcore.get_output_count()] \
                    = placement.materialize()
        else:
            self.matrix_placements.append(placement)

        self.axon_sources.extend(softcore.axon_sources)
        self._axon_source_spans = None

        self.available_axons -= softcore.get_input_count()
        self.available_neurons -= softcore.get_output_count()

        adopt_or_check(self, softcore, constrained=self.constrained_properties)
        note_resident_bank(self, softcore)

        if self.residency_class_id is None:
            tg = getattr(softcore, "residency_class_id", None)
            self.residency_class_id = (
                int(tg) if tg is not None else ungrouped_fallback_id(softcore.id)
            )

        if self.latency is None:
            self.latency = softcore.latency

        if getattr(softcore, "hardware_bias", None) is not None:
            if self.hardware_bias is None:
                self.hardware_bias = np.zeros(self.neurons_per_core)
            self.hardware_bias[neuron_offset:neuron_offset + softcore.get_output_count()] = softcore.hardware_bias

        self.unusable_space += \
            (neuron_offset * softcore.get_input_count()) + \
            (axon_offset * softcore.get_output_count())

    def _softcore_placement(
        self, softcore, axon_offset: int, neuron_offset: int,
    ) -> MatrixPlacement:
        keep_rows = getattr(softcore, "compact_keep_rows", None)
        if softcore.core_matrix is not None:
            source, rows, cols = softcore.core_matrix, None, None
        elif keep_rows is not None:
            source = softcore.compact_source_matrix
            rows, cols = keep_rows, softcore.compact_keep_cols
        else:
            raise ValueError(
                f"HardCore.add_softcore: softcore id={softcore.id} carries "
                f"neither a core_matrix nor a compaction descriptor."
            )
        return MatrixPlacement(
            source=source, keep_rows=rows, keep_cols=cols,
            axon_offset=axon_offset, neuron_offset=neuron_offset,
            axons=softcore.get_input_count(),
            neurons=softcore.get_output_count(),
        )

    def has_core_matrix(self) -> bool:
        return self.core_matrix is not None or bool(self.matrix_placements)

    def get_core_matrix(self):
        """Full ``(axons_per_core, neurons_per_core)`` weight grid.

        An exact-fit single plain placement returns the SHARED payload object
        unchanged (stable identity: pickle memoization and upload dedup store
        each bank payload once); composites materialize TRANSIENTLY per call.
        """
        return resolve_core_matrix(
            self.core_matrix, self.matrix_placements,
            self.axons_per_core, self.neurons_per_core,
            owner="HardCore",
        )

    def core_matrix_key(self) -> tuple:
        """Content identity: equal keys imply byte-identical resolved grids."""
        return core_matrix_content_key(
            self.core_matrix, self.matrix_placements,
            self.axons_per_core, self.neurons_per_core,
        )

    def core_matrix_dtype(self) -> "np.dtype | None":
        return core_matrix_dtype(self.core_matrix, self.matrix_placements)

    def core_matrix_payloads(self) -> tuple:
        """Shared arrays behind :meth:`core_matrix_key` — a memo holding a key
        must retain these and re-check identity (ids can be re-used)."""
        return core_matrix_payloads(self.core_matrix, self.matrix_placements)

    def get_axon_source_spans(self):
        """Cached range-compressed axon_sources; invalidated on add_softcore."""
        if self._axon_source_spans is None:
            from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans

    def __getstate__(self) -> dict:
        """Pickle axon_sources range-compressed; drop the transient span cache."""
        from mimarsinan.mapping.support.spike_source_spans import encode_spike_sources_packed

        state = dict(self.__dict__)
        state["axon_sources"] = encode_spike_sources_packed(self.axon_sources)
        state["_axon_source_spans"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Decode packed axon_sources; legacy raw-list states load unchanged."""
        from mimarsinan.mapping.support.spike_source_spans import decode_spike_sources_packed

        encoded = state.get("axon_sources")
        if isinstance(encoded, tuple):
            state = dict(state)
            state["axon_sources"] = decode_spike_sources_packed(encoded)
        self.__dict__.update(state)
