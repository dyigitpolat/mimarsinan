"""HardCore container for chip-level neuron/axon packing."""

from typing import Any

import numpy as np

from mimarsinan.mapping.packing.placement_cost import note_resident_bank
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

        self.core_matrix = None
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

        if (
            self.core_matrix is None
            and softcore.get_input_count() == self.axons_per_core
            and softcore.get_output_count() == self.neurons_per_core
        ):
            # Exact-fit 1:1: alias the (possibly bank-shared) matrix so pickle
            # memoization stores each bank payload once across duplicate
            # cores. Safe: the core is now full, so no later add can write.
            self.core_matrix = softcore.core_matrix
        else:
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

    def get_axon_source_spans(self):
        """Cached range-compressed axon_sources; invalidated on add_softcore."""
        if self._axon_source_spans is None:
            from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans
