"""SoftCore container for a mapped neural IR node."""

import torch


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
            from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
            self._axon_source_spans = compress_spike_sources(self.axon_sources)
        return self._axon_source_spans
