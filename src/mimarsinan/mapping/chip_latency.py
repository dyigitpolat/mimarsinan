from collections import defaultdict


class ChipLatency:
    def __init__(self, mapping):
        self.mapping = mapping
        self.memo = {}

    def __get_non_zero_axon_sources(self, core, neuron_idx):
        non_zero_axon_sources = []
        for axon_idx, w in enumerate(core.core_matrix[:, neuron_idx]):
            if abs(w) == 0:
                continue
            src = core.axon_sources[axon_idx]
            # Skip off axons: IR may mark is_off_ without zeroing core_matrix; treating them as inputs gave delay 0 and under-scheduled sources (see mapping/ARCHITECTURE.md).
            if getattr(src, "is_off_", False):
                continue
            non_zero_axon_sources.append(src)

        return non_zero_axon_sources

    def __is_direct_signal(self, source):
        return source.core_ < 0

    def get_delay_for(self, source):
        key = (source.core_, source.neuron_)

        if key in self.memo: return self.memo[key]

        if self.__is_direct_signal(source):
            self.memo[key] = 0
            return 0

        current_core = self.mapping.cores[source.core_]
        non_zero_axon_sources = self.__get_non_zero_axon_sources(
            current_core, source.neuron_)

        if len(non_zero_axon_sources) == 0:
            self.memo[key] = 0
            return 0

        result = 1 + max([
            self.get_delay_for(source) for source in non_zero_axon_sources])

        self.memo[key] = result
        return result

    def calculate(self):
        self.memo = {}
        if len(self.mapping.output_sources) == 0:
            raise ValueError(
                "ChipLatency.calculate: mapping has no output_sources (empty list). "
                "This usually means all output neurons were pruned or compaction removed every output ref. "
                "Check IR pruning and soft-core compaction."
            )
        result = max([
            self.get_delay_for(source) for source in self.mapping.output_sources])

        latencies = defaultdict(int)
        for key in self.memo:
            core_idx, neuron_idx = key
            latencies[core_idx] = max(latencies[core_idx], self.memo[key] - 1)

        for core_idx in latencies:
            if core_idx >= 0:
                self.mapping.cores[core_idx].latency = latencies[core_idx]

        result = max(result, self._enforce_core_latency_invariant())
        result = self._align_shiftable_cores(result)
        return result

    @staticmethod
    def _live_cross_core_sources(core):
        """Yield (axon_idx, src) for cross-core refs (not off / always-on / segment input)."""
        for axon_idx, src in enumerate(core.axon_sources):
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_always_on_", False):
                continue
            if getattr(src, "is_input_", False):
                continue
            yield axon_idx, src

    def _classify_shiftable(self):
        """Cores with only shiftable or dead cross-core inputs (fixed-point over cores)."""
        cores = self.mapping.cores
        shiftable: set[int] = set()
        for core_idx, core in enumerate(cores):
            if not any(True for _ in self._live_cross_core_sources(core)):
                shiftable.add(core_idx)

        changed = True
        while changed:
            changed = False
            for core_idx, core in enumerate(cores):
                if core_idx in shiftable:
                    continue
                live = list(self._live_cross_core_sources(core))
                if not live:
                    continue  # caught above; defensive
                if all(int(src.core_) in shiftable for _, src in live):
                    shiftable.add(core_idx)
                    changed = True
        return shiftable

    def _build_consumer_map(self):
        """Return ``{src_core_idx: set(consumer_core_idx)}`` for the mapping."""
        consumers: dict[int, set[int]] = defaultdict(set)
        for core_idx, core in enumerate(self.mapping.cores):
            for _, src in self._live_cross_core_sources(core):
                consumers[int(src.core_)].add(core_idx)
        return consumers

    def _align_shiftable_cores(self, current_max_latency):
        """Raise shiftable-core latencies so consumers read a live firing window."""
        cores = self.mapping.cores
        if not cores:
            return current_max_latency

        shiftable = self._classify_shiftable()
        if not shiftable:
            return current_max_latency

        consumers = self._build_consumer_map()
        new_max = current_max_latency
        changed = True
        passes = 0
        max_passes = len(cores) + 1
        while changed and passes < max_passes:
            changed = False
            passes += 1
            for core_idx in shiftable:
                cons = consumers.get(core_idx, set())
                if not cons:
                    continue
                consumer_max = max(int(cores[c].latency or 0) for c in cons)
                required = consumer_max - 1
                current = int(cores[core_idx].latency or 0)
                if current < required:
                    cores[core_idx].latency = required
                    new_max = max(new_max, required + 1)
                    changed = True
        return new_max

    def _enforce_core_latency_invariant(self):
        """consumer.latency >= max(live source core latency) + 1; monotone sweep."""
        cores = self.mapping.cores
        if not cores:
            return 0
        changed = True
        passes = 0
        max_passes = len(cores) + 1
        new_max = max(int(c.latency or 0) for c in cores)
        while changed and passes < max_passes:
            changed = False
            passes += 1
            for core_idx, core in enumerate(cores):
                current_lat = int(core.latency or 0)
                max_src_lat = -1
                for axon in core.axon_sources:
                    if getattr(axon, "is_off_", False):
                        continue
                    if getattr(axon, "is_always_on_", False):
                        continue
                    if getattr(axon, "is_input_", False):
                        continue
                    src_core_idx = int(axon.core_)
                    if src_core_idx < 0:
                        continue
                    src_lat = int(cores[src_core_idx].latency or 0)
                    if src_lat > max_src_lat:
                        max_src_lat = src_lat
                if max_src_lat < 0:
                    continue
                required = max_src_lat + 1
                if current_lat < required:
                    core.latency = required
                    new_max = max(new_max, required)
                    changed = True
        return new_max