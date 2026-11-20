from collections import defaultdict

import numpy as np

from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.mapping.latency.upstream import CoreReads, core_reads, direct_signals, source_arrays

#: A pass's hard-core-mapping attribute: per core, the latency the fused segment runs it at (set only where pinned).
PINNED_LATENCIES = "pinned_latencies"


#: A hard core's axon sources as arrays (``upstream.source_arrays``; the name kept for its importers).
_source_arrays = source_arrays


class CyclicCoreGraphError(ToolchainRefusal, RuntimeError):
    """[W12] A hard-core graph with a cycle: no core order exists to walk it, so the program is refused by name
    (Lemma A; the pass projection's H1-latency refusal comes first on a shipped build)."""


class ChipLatency:
    def __init__(self, mapping):
        self.mapping = mapping
        self.memo = {}
        # [U-LIVE] each core's axons under the one liveness predicate (``upstream.core_reads``), read once per
        # instance; the matrix is read inside the predicate and never held here.
        self._reads_cache = {}

    def _reads(self, core_idx, *, entries: bool = False) -> CoreReads:
        cached = self._reads_cache.get(core_idx)
        if cached is None or (entries and cached.entries is None):
            cached = core_reads(self.mapping.cores[core_idx])
            self._reads_cache[core_idx] = cached
        return cached

    def _live_source_cores(self, core_idx):
        """[U-LIVE] Source core of every live axon (the predicate: not off, not a stage input or the always-on line,
        and its weight row nonzero): the sweep, the shift alignment and the consumer map read one source set with the
        per-neuron delays."""
        reads = self._reads(core_idx)
        return reads.core[reads.live]

    def _charged_sources(self, core_idx, neuron_idx):
        """The ``(core, neuron)`` of each axon whose entry for ``neuron_idx`` delivers charge (the predicate per
        neuron), a stage core's own where it is live and ``(-1, neuron)`` (a direct signal) where it reads none."""
        reads = self._reads(core_idx, entries=True)
        entries = reads.entries
        assert entries is not None  # read with its entries
        rows = min(entries.shape[0], len(reads.core))
        hit = np.flatnonzero(entries[:rows, neuron_idx])
        return [(int(reads.core[a]) if reads.live[a] else -1, int(reads.neuron[a])) for a in hit]

    def get_delay_for(self, source):
        """Per-neuron delay of an output source: 0 for a direct signal (off, a stage input or the always-on line, by
        the predicate's marks), else ``1 +`` the largest delay over the entries that deliver charge to it."""
        key = (int(source.core_), int(source.neuron_))
        if bool(direct_signals([source])[0]):
            return self.memo.setdefault(key, 0)
        return self._delay_of(key)

    def _delay_of(self, key):
        if key in self.memo:
            return self.memo[key]
        core, neuron = key
        if core < 0:
            self.memo[key] = 0
            return 0
        sources = self._charged_sources(core, neuron)
        result = 1 + max((self._delay_of(s) for s in sources), default=-1) if sources else 0
        self.memo[key] = result
        return result

    def _compute_all_delays(self):
        """Fill ``self.memo`` for every (core, neuron) — bit-equal to the
        recursive ``get_delay_for`` walk (locked by test), one vectorized
        pass per core in topological order instead of per-neuron recursion."""
        cores = self.mapping.cores
        n_cores = len(cores)
        live_masks, axon_meta, deps = [], [], [set() for _ in range(n_cores)]
        for ci in range(n_cores):
            # [U-LIVE] per (row, neuron) the entries that deliver charge (nonzero, not off; bias rows may extend past
            # axon_sources) and per axon the predicate: a producer's delay is read only through a live axon.
            reads = self._reads(ci, entries=True)
            live = reads.entries
            assert live is not None  # read with its entries
            n_rows = live.shape[0]
            n_src = len(reads.core)
            if n_rows > n_src and bool(live[n_src:].any()):
                raise IndexError(
                    f"core {ci}: nonzero weight row {n_src}+ has no axon source"
                )
            src_core = np.full(n_rows, -1, dtype=np.int64)
            src_neuron = np.zeros(n_rows, dtype=np.int64)
            rows = min(n_rows, n_src)
            source = reads.live[:rows]
            src_core[:rows] = np.where(source, reads.core[:rows], -1)
            src_neuron[:rows] = np.where(source, reads.neuron[:rows], 0)
            deps[ci].update(int(c) for c in np.unique(src_core[:rows][source]))
            live_masks.append(live)
            axon_meta.append((src_core, src_neuron))

        indegree = {ci: len(deps[ci]) for ci in range(n_cores)}
        consumers = defaultdict(set)
        for ci in range(n_cores):
            for p in deps[ci]:
                consumers[p].add(ci)
        ready = [ci for ci in range(n_cores) if indegree[ci] == 0]
        order = []
        while ready:
            ci = ready.pop()
            order.append(ci)
            for c in consumers[ci]:
                indegree[c] -= 1
                if indegree[c] == 0:
                    ready.append(c)
        if len(order) != n_cores:
            raise CyclicCoreGraphError(
                "ChipLatency: cyclic core dependency; the recursive walk "
                "would not terminate either"
            )

        delays: list = [None] * n_cores
        for ci in order:
            live = live_masks[ci]
            n_out = live.shape[1]
            src_core, src_neuron = axon_meta[ci]
            axon_delay = np.zeros(len(src_core), dtype=np.int64)
            # Dead axons (all-zero weights) are masked out per neuron and may
            # reference cores outside the dependency order — never read them.
            live_any = (
                live.any(axis=1) if live.shape[0] else np.zeros(0, dtype=bool)
            )
            for p in np.unique(src_core[(src_core >= 0) & live_any]):
                sel = (src_core == p) & live_any
                axon_delay[sel] = delays[int(p)][src_neuron[sel]]
            if live.shape[0] == 0:
                neuron_delays = np.zeros(n_out, dtype=np.int64)
            else:
                masked = np.where(live, axon_delay[:, None], -1)
                neuron_delays = masked.max(axis=0) + 1
            delays[ci] = neuron_delays
            for j in range(n_out):
                self.memo[(ci, j)] = int(neuron_delays[j])

    def calculate(self):
        pins = getattr(self.mapping, PINNED_LATENCIES, None)
        if pins is not None:
            for core, pin in zip(self.mapping.cores, pins):
                core.latency = int(pin)
            return max((int(p) for p in pins), default=-1) + 1
        self.memo = {}
        self._reads_cache = {}
        if len(self.mapping.output_sources) == 0:
            raise ValueError(
                "ChipLatency.calculate: mapping has no output_sources (empty list). "
                "This usually means all output neurons were pruned or compaction removed every output ref. "
                "Check IR pruning and soft-core compaction."
            )
        self._compute_all_delays()
        # every (core, neuron) is in the memo: the sweep reads the live sources only, so the entries are let go
        self._reads_cache = {ci: r._replace(entries=None) for ci, r in self._reads_cache.items()}
        outputs = list(self.mapping.output_sources)
        direct = direct_signals(outputs)  # the output sources' direct signals by the predicate's marks, at once
        keys = [(int(s.core_), int(s.neuron_)) for s in outputs]
        result = max(self.memo.setdefault(key, 0) if d else self._delay_of(key) for key, d in zip(keys, direct.tolist()))

        latencies = defaultdict(int)
        for (core_idx, _neuron_idx), delay in self.memo.items():
            latencies[core_idx] = max(latencies[core_idx], delay - 1)

        # a direct signal's pseudo-core (a negative index) is no core of the mapping
        for core_idx, core in enumerate(self.mapping.cores):
            if core_idx in latencies:
                core.latency = latencies[core_idx]

        result = max(result, self._enforce_core_latency_invariant())
        result = self._align_shiftable_cores(result)
        self._reads_cache = {}
        return result

    def _classify_shiftable(self):
        """Cores with only shiftable or dead cross-core inputs (fixed-point over cores)."""
        cores = self.mapping.cores
        live = [np.unique(self._live_source_cores(i)) for i in range(len(cores))]
        shiftable: set[int] = {i for i in range(len(cores)) if not live[i].size}

        changed = True
        while changed:
            changed = False
            for core_idx in range(len(cores)):
                if core_idx in shiftable or not live[core_idx].size:
                    continue
                if all(int(c) in shiftable for c in live[core_idx]):
                    shiftable.add(core_idx)
                    changed = True
        return shiftable

    def _build_consumer_map(self):
        """Return ``{src_core_idx: set(consumer_core_idx)}`` for the mapping."""
        consumers: dict[int, set[int]] = defaultdict(set)
        for core_idx in range(len(self.mapping.cores)):
            for c in np.unique(self._live_source_cores(core_idx)):
                consumers[int(c)].add(core_idx)
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
        sources = []
        for core_idx in range(len(cores)):
            ids = self._live_source_cores(core_idx)
            sources.append(np.unique(ids[ids >= 0]).tolist())
        while changed and passes < max_passes:
            changed = False
            passes += 1
            for core_idx, core in enumerate(cores):
                current_lat = int(core.latency or 0)
                if not sources[core_idx]:
                    continue
                max_src_lat = max(int(cores[s].latency or 0) for s in sources[core_idx])
                required = max_src_lat + 1
                if current_lat < required:
                    core.latency = required
                    new_max = max(new_max, required)
                    changed = True
        return new_max
