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
            # ``off`` axons deliver no signal regardless of any residual
            # weight value: IR pruning can zero an axon's source (mark it
            # ``is_off_``) without zeroing the corresponding column in
            # ``core_matrix``. Including such axons here previously made
            # ``get_delay_for`` see ``src.core_ == -1`` as a *direct input*
            # (delay 0) and pulled the dependent core's latency down to
            # whatever depth the bias path implied. Concrete failure mode:
            # a deep core whose every consumer-facing neuron has its only
            # non-zero weights on off-axons ended up at ``latency=1`` while
            # its downstream consumers (correctly at lat=1) read its
            # buffer cycles before the source had fired — wedging NF→SCM
            # by O(3 pp) on compilagent 8-deep architectures.
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

        # Post-pass: enforce the per-core latency invariant. The chip
        # schedules each core as a whole during ``[core.latency, core.latency+T)``;
        # a consumer core can only safely fire once *every* upstream core
        # it depends on has fired, regardless of which specific source
        # neurons the consumer's weights happen to reference. The
        # per-neuron walk above can under-estimate latency when a
        # consumer's non-zero weights land on source-core neurons that
        # themselves have shallow per-neuron delays (e.g. bias-only,
        # all-off-axon non-zero weights left over from IR pruning).
        # Without this enforcement, consumers get scheduled to fire
        # before their source-core's active window starts and integrate
        # zeros — producing dead edges that wedge NF→SCM accuracy
        # proportionally to the volume of such edges.
        result = max(result, self._enforce_core_latency_invariant())

        # Post-pass: align cores whose *only* live inputs are off / always-on
        # / segment-input with the depth their consumers expect.
        #
        # IR pruning can rewire every axon source of a core to ``off`` when
        # all of that core's upstream output columns get pruned (Phase 2 of
        # ``prune_ir_graph``).  Such cores are then *time-shiftable*: their
        # output depends only on ``hardware_bias`` (and any always-on
        # axons), so they fire the same pattern at any cycle.  The backward
        # walk above assigns them ``latency=0`` because their non-zero axon
        # sources all return delay 0 — but a consumer at ``latency=L`` then
        # reads their buffer at cycles ``[L, L+T)`` while they only update
        # the buffer during ``[0, T)``, leaving the consumer to integrate
        # the same stale cycle ``T-1`` firing ``T`` times.  HCM's
        # ``record_in_t`` therefore disagrees with the SANA-FE spike trace
        # (which counts what the source actually emitted), and the segment
        # output drifts away from NF.
        #
        # We fix this in-place: for every shiftable chain, bump latencies
        # forward so the source's active window ends exactly when the
        # deepest consumer reads its last cycle.  This preserves the
        # per-cycle cascade end-to-end without touching the simulation
        # loop, and the shiftable cores produce the same total spikes (the
        # window is the same length, just translated in time).
        result = self._align_shiftable_cores(result)
        return result

    # ------------------------------------------------------------------
    # Shiftable-chain alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _live_cross_core_sources(core):
        """Yield ``(axon_idx, src)`` pairs whose axon source is a real
        cross-core reference (not off / always-on / segment input)."""
        for axon_idx, src in enumerate(core.axon_sources):
            if getattr(src, "is_off_", False):
                continue
            if getattr(src, "is_always_on_", False):
                continue
            if getattr(src, "is_input_", False):
                continue
            yield axon_idx, src

    def _classify_shiftable(self):
        """Return the set of core indices that may be freely time-shifted.

        A core is shiftable iff every one of its *live* (non-off,
        non-always-on, non-input) cross-core axon sources points to a
        shiftable core.  Dead-input cores (no live cross-core sources)
        are the base case — their firing depends on ``hardware_bias``
        alone, so any active window of length ``T`` produces the same
        spike train.  Shiftable-ness propagates forward: a core whose
        only cross-core inputs come from shiftable cores can be shifted
        in lockstep with them.

        Walks to a fixed point — propagation is monotone (cores only
        enter the set), so termination is bounded by ``len(cores)``.
        """
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
        """Bump shiftable cores forward so their window ends at the consumer's.

        For every shiftable core ``C`` we want
        ``C.latency = max(consumer.latency for consumer in consumers(C)) - 1``
        so the consumer's first-cycle read sees ``C``'s first-cycle firing
        (the chip's single-cycle output buffer holds one cycle's spike).
        Cores with no consumers fall back to whatever the backward walk
        assigned (typically 0).  We sweep until stable so a long
        shiftable chain ``dead → mid → mid → real-consumer`` propagates
        the deepest consumer's depth all the way to the chain's root.
        """
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
        # Bounded sweep: each iteration can only raise latencies, and a
        # core's latency is bounded by the deepest non-shiftable core's
        # latency, so ``len(cores)`` iterations is a generous ceiling.
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

    # ------------------------------------------------------------------
    # Per-core latency invariant enforcement
    # ------------------------------------------------------------------

    def _enforce_core_latency_invariant(self):
        """Ensure each core's latency >= max(source_core_lat) + 1.

        Sweeps until fixed point — propagation is monotone (latencies
        only rise), so termination is bounded by ``len(cores)``. Only
        considers ``kind == "core"`` axon source spans; off / always-on
        / input axons don't introduce a chip-side scheduling
        dependency on another core's active window.
        """
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
                # Walk ``axon_sources`` directly (rather than the
                # span-compressed view from ``get_axon_source_spans``) so
                # this pass works with the SimpleNamespace stubs the unit
                # tests use as well as the real ``HardCore``. Off, always-on
                # and segment-input axons don't introduce a chip-side
                # scheduling dependency on another core's active window.
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
                    # No live cross-core source — leave whatever the
                    # backward walk or shiftable alignment assigned.
                    continue
                required = max_src_lat + 1
                if current_lat < required:
                    core.latency = required
                    new_max = max(new_max, required)
                    changed = True
        return new_max