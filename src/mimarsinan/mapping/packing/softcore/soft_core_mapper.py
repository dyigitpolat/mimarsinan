"""SoftCoreMapping container for packed soft cores (IR → simulation / hard-core packing)."""

from __future__ import annotations


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

        self.weight_banks: dict[int, "object"] = {}

        self._psum_group_counter = 0
        self._output_source_spans = None

    def get_output_source_spans(self):
        """Range-compressed view of output_sources for fast simulation / compact inspection."""
        if self._output_source_spans is None:
            from mimarsinan.mapping.support.spike_source_spans import compress_spike_sources
            self._output_source_spans = compress_spike_sources(self.output_sources)
        return self._output_source_spans
