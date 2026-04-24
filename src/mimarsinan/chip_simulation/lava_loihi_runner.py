"""Lava process-graph Loihi simulation of a ``HybridHardCoreMapping``.

Per neural segment, per hard core the runner builds a literal Lava process
graph::

    RingBuffer(input_spikes) → Dense(weights) → SubtractiveLIFReset → RingBuffer(output)

``SubtractiveLIFReset`` is a one-off subclass of ``lava.proc.lif.process.LIF``
with a float model that (a) has no current or voltage decay (du = dv = 0
with u treated as direct synaptic input via du=1 semantics) and (b) applies
subtractive reset on spike (``v -= vth``) to match nevresim's
``firing_mode='Default'`` semantics exactly.  The model also applies the
periodic state reset behaviour of ``LIFReset`` so the same graph can
process many input samples in sequence — samples are packed into the time
dimension and separated by a per-window warmup + reset phase.

Host-side responsibilities (CPU, not Loihi):

* Rate-encoding the original input into spike trains for the first neural
  segment (matches ``SimulationRunner``'s uniform rate encoding).
* Running :class:`HybridStage.compute` stages on the host (the ComputeOps
  wrap arbitrary PyTorch modules — rearrange, concatenate, host-side
  Perceptrons).  This follows the same gather / execute / scatter pattern
  as ``SimulationRunner._execute_compute_op_np`` and is reused directly
  from that class.
* Sequencing stages in topological order and threading rates through a
  shared ``state_buffer`` keyed by IR node id.

The runner emits per-stage timing and per-sample agreement profiling to
make end-to-end debugging tractable (see ``run`` and
``_RunProfile``).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.simulation_runner import SimulationRunner
from mimarsinan.data_handling.data_loader_factory import DataLoaderFactory, shutdown_data_loader
from mimarsinan.mapping.hybrid_hardcore_mapping import (
    HybridHardCoreMapping,
    HybridStage,
)
from mimarsinan.mapping.softcore_mapping import HardCore, HardCoreMapping


# ---------------------------------------------------------------------------
# SubtractiveLIFReset process.  Lazy-imported so the runner module remains
# importable even when Lava itself is broken on the host interpreter.
# The class must live in its own module (not a closure) so Lava's process
# model discovery (`ProcGroupDiGraphs._find_proc_models`) can re-import it.
# ---------------------------------------------------------------------------

_SUBTRACTIVE_LIF_CLS = None


def _probe_lava() -> None:
    """Import Lava and cache the SubtractiveLIFReset class on first use."""
    global _SUBTRACTIVE_LIF_CLS
    if _SUBTRACTIVE_LIF_CLS is not None:
        return
    from mimarsinan.chip_simulation.subtractive_lif import SubtractiveLIFReset
    _SUBTRACTIVE_LIF_CLS = SubtractiveLIFReset


def _subtractive_lif_cls():
    _probe_lava()
    return _SUBTRACTIVE_LIF_CLS


# ---------------------------------------------------------------------------
# Rate-encoded spike generators (match SpikingUnifiedCoreFlow's uniform mode).
# ---------------------------------------------------------------------------


def _uniform_rate_encode(rates: np.ndarray, T: int) -> np.ndarray:
    """Uniform-rate spike encoding matching SCM ``spike_mode='Uniform'``.

    Parameters
    ----------
    rates : (N, D) array of non-negative values in [0, 1].
    T     : number of cycles.

    Returns
    -------
    (N, D, T) binary spike train — for each (n, d), ``N_d = round(rate * T)``
    spikes are placed at uniformly-spaced cycle indices.
    """
    rates = np.clip(rates, 0.0, 1.0)
    N_samples, D = rates.shape
    spikes = np.zeros((N_samples, D, T), dtype=np.float32)
    for cycle in range(T):
        n = np.round(rates * T).astype(np.int64)  # (N, D)
        mask_full = (n == T)
        mask_active = (n != 0) & (n != T) & (cycle < T)
        n_safe = np.maximum(n, 1)
        spacing = T / n_safe.astype(np.float64)
        fire = mask_active & (np.floor(cycle / spacing) < n_safe) & (np.floor(cycle % spacing) == 0)
        spikes[:, :, cycle] = fire.astype(np.float32)
        spikes[:, :, cycle][mask_full] = 1.0
    return spikes


# ---------------------------------------------------------------------------
# Profiling trace.
# ---------------------------------------------------------------------------


@dataclass
class _StageTrace:
    name: str
    kind: str
    seconds: float
    cores: int = 0
    samples: int = 0


@dataclass
class _RunProfile:
    stages: List[_StageTrace] = field(default_factory=list)
    total_seconds: float = 0.0

    def log(self) -> None:
        print("=== LavaLoihiRunner profile ===")
        by_kind: Dict[str, float] = defaultdict(float)
        for st in self.stages:
            per = f", {st.cores} cores" if st.kind == "neural" else ""
            print(f"  [{st.kind:>7}] {st.seconds:7.2f}s  {st.name}{per}")
            by_kind[st.kind] += st.seconds
        print(f"  ---")
        for k, s in by_kind.items():
            print(f"  Σ {k:>7}: {s:7.2f}s")
        print(f"  Σ total : {self.total_seconds:7.2f}s")


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------


class LavaLoihiRunner:
    """Evaluate a ``HybridHardCoreMapping`` under a real Lava process graph.

    The runner splits responsibility between the Loihi-compatible chip side
    (Dense + SubtractiveLIFReset per core) and the host (ComputeOp stages,
    routing, input encoding).  Its outputs should fall within the
    ``degradation_tolerance`` of the nevresim path; substantial drift is a
    bug in either the graph wiring or the state-buffer routing and should
    fail the Loihi step.
    """

    def __init__(
        self,
        pipeline,
        mapping: HybridHardCoreMapping,
        simulation_length: int,
        preprocessor: nn.Module | None = None,
    ):
        self.pipeline = pipeline
        self.mapping = mapping
        self.T = int(simulation_length)
        self.preprocessor = preprocessor if preprocessor is not None else nn.Identity()

        self.device = pipeline.config["device"]
        # Cap the Loihi pass separately from nevresim so a slow Lava runtime
        # doesn't explode the overall pipeline budget.  Default: 50 samples.
        self.max_samples = int(
            pipeline.config.get(
                "max_loihi_samples",
                min(50, int(pipeline.config.get("max_simulation_samples", 500))),
            )
        )

        spiking = pipeline.config.get("spiking_mode", "lif")
        if spiking != "lif":
            raise ValueError(
                f"LavaLoihiRunner only supports spiking_mode='lif'; got {spiking!r}. "
                "TTFS modes do not map onto Loihi LIF dynamics."
            )

        _probe_lava()  # surface any Lava import failures up front

        self._data_loader_factory = DataLoaderFactory(pipeline.data_provider_factory)
        self._profile = _RunProfile()
        self._accuracy: float | None = None

    # --------------------------------------------------------------------- load

    def _load_test_samples(self) -> Tuple[np.ndarray, np.ndarray]:
        provider = self._data_loader_factory.create_data_provider()
        loader = self._data_loader_factory.create_test_loader(
            provider.get_test_batch_size(), provider
        )
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        total = 0
        for x, y in loader:
            if total >= self.max_samples:
                break
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            if total + len(x_np) > self.max_samples:
                take = self.max_samples - total
                x_np = x_np[:take]
                y_np = y_np[:take]
            xs.append(x_np)
            ys.append(y_np)
            total += len(x_np)
        shutdown_data_loader(loader)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    # ------------------------------------------------------------------ prepare

    def _preprocess(self, x_np: np.ndarray) -> np.ndarray:
        """Apply the shared preprocessor then flatten to (N, input_size)."""
        x = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            x = self.preprocessor(x)
        return x.detach().cpu().numpy().reshape(x.shape[0], -1)

    # ------------------------------------------------------------------ lava op

    def _run_core_lava(
        self,
        *,
        weights: np.ndarray,
        threshold: float,
        hardware_bias: np.ndarray | None,
        input_spikes: np.ndarray,
    ) -> np.ndarray:
        """Run ONE hard core's Dense + SubtractiveLIFReset on Lava.

        Parameters
        ----------
        weights        : (n_out, n_in) float array.
        threshold      : scalar vth.
        hardware_bias  : optional (n_out,) bias vector.
        input_spikes   : (n_in, N * T) packed spike train across samples.

        Returns
        -------
        output_spikes  : (n_out, N * T) packed output spike train.

        Timing model
        ------------
        Lava's RingBuffer source → Dense → SubtractiveLIF → RingBuffer sink
        pipeline has a per-cycle send/recv delay, so the input bit at
        ``data[k]`` is processed by the LIF at timestep ``t = k + 3``, and
        its output spike is stored at sink index ``t`` (= ``k + 3``).

        Layout::

            [pad_head: 2 zero cycles] [warmup: T cycles] [N*T sample cycles] [tail: 3 zeros]

        Reset boundaries are aligned with sample-window starts via
        ``reset_offset = (pad_head + T + 3) % T``: this ensures the LIF's
        state is wiped at exactly the cycle where it begins processing each
        sample's first bit.  Total runtime is
        ``pad_head + T + N*T + tail_pad`` so sink.buffer is large enough to
        capture the very last sample's last spike.
        """
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava.proc.dense.process import Dense
        from lava.proc.io.sink import RingBuffer as Sink
        from lava.proc.io.source import RingBuffer as Source

        n_out, n_in = weights.shape
        total_cycles = input_spikes.shape[1]
        assert total_cycles % self.T == 0
        N = total_cycles // self.T
        T = self.T

        PAD_HEAD = 2  # absorbs the first 2 cycles when LIF has no useful input
        TAIL = 3      # covers Source→Sink delivery delay for the last sample
        PIPELINE_DELAY = 1  # empirically: data[k] → sink index k + 1

        pad_head_block = np.zeros((n_in, PAD_HEAD), dtype=np.float32)
        warmup = input_spikes[:, :T]
        tail_block = np.zeros((n_in, TAIL), dtype=np.float32)
        data = np.concatenate([pad_head_block, warmup, input_spikes, tail_block], axis=1).astype(np.float32)
        total_steps = data.shape[1]

        reset_offset = (PAD_HEAD + T + PIPELINE_DELAY) % T

        if hardware_bias is None:
            bias_mant = np.zeros((n_out,), dtype=np.float32)
        else:
            bias_mant = np.asarray(hardware_bias, dtype=np.float32).reshape(-1)

        SubLIF = _subtractive_lif_cls()
        src = Source(data=data)
        dense = Dense(weights=weights.astype(np.float32))
        lif = SubLIF(
            shape=(n_out,),
            du=1,
            dv=0,
            vth=float(threshold),
            bias_mant=bias_mant,
            reset_interval=T,
            reset_offset=reset_offset,
        )
        sink = Sink(shape=(n_out,), buffer=total_steps)

        src.s_out.connect(dense.s_in)
        dense.a_out.connect(lif.a_in)
        lif.s_out.connect(sink.a_in)

        try:
            lif.run(
                condition=RunSteps(num_steps=total_steps),
                run_cfg=Loihi2SimCfg(select_tag="floating_pt"),
            )
            raw = sink.data.get()  # (n_out, total_steps)
        finally:
            lif.stop()

        start = PAD_HEAD + T + PIPELINE_DELAY
        return np.asarray(raw[:, start : start + N * T], dtype=np.float32)

    # -------------------------------------------------------- segment execution

    def _run_neural_segment(
        self,
        seg: HardCoreMapping,
        seg_input_rates: np.ndarray,
    ) -> np.ndarray:
        """Execute one neural segment as a SINGLE Lava process graph.

        Build one Lava graph where:
        * a :class:`Source` supplies segment input spikes,
        * an "always-on" :class:`Source` (constant 1s) supplies bias axons,
        * every hard core becomes a :class:`SubtractiveLIFReset` with one
          :class:`Dense` edge per distinct upstream source.  Edges summed
          into each LIF's ``a_in`` reproduce the core's weighted input.
        * each LIF is probed by a :class:`Sink` so the host can read spike
          trains back after the run.

        This matches how a Loihi deployment of the same mapping is
        structured: one compiled chip program per segment, with all cores
        executing in parallel inside a single run.  Running as a single
        graph also amortises Lava's Python-runtime compile + teardown
        costs, which per-core graph construction turned into an
        O(N_cores × seconds) bottleneck.

        Parameters
        ----------
        seg              : the segment's HardCoreMapping.
        seg_input_rates  : (N, seg_in_size) input rates in [0, 1].

        Returns
        -------
        seg_output_rates : (N, output_count) output rates in [0, 1],
                           ordered per ``seg.output_sources``.
        """
        from collections import defaultdict
        from lava.magma.core.run_conditions import RunSteps
        from lava.magma.core.run_configs import Loihi2SimCfg
        from lava.proc.dense.process import Dense
        from lava.proc.io.sink import RingBuffer as Sink
        from lava.proc.io.source import RingBuffer as Source

        from mimarsinan.mapping.spike_source_spans import compress_spike_sources

        SubLIF = _subtractive_lif_cls()

        T = self.T
        PAD_HEAD = 2
        TAIL = 3
        PIPELINE_DELAY = 1  # one Lava send/recv cycle per process

        N = seg_input_rates.shape[0]
        seg_in_size = seg_input_rates.shape[1]

        # Encode segment input spikes: (seg_in, N, T) → (seg_in, N*T).
        seg_input_spikes = _uniform_rate_encode(seg_input_rates, T)
        packed_input = seg_input_spikes.transpose(1, 0, 2).reshape(seg_in_size, N * T)

        # Pad + warmup so reset boundaries align with sample windows.
        pad_head_block = np.zeros((seg_in_size, PAD_HEAD), dtype=np.float32)
        warmup = packed_input[:, :T]
        tail_block = np.zeros((seg_in_size, TAIL), dtype=np.float32)
        seg_data = np.concatenate(
            [pad_head_block, warmup, packed_input, tail_block], axis=1
        ).astype(np.float32)
        total_steps = seg_data.shape[1]
        reset_offset = (PAD_HEAD + T + PIPELINE_DELAY) % T

        # ------------------------------------------------------------------
        # Build the Lava graph.
        # ------------------------------------------------------------------

        seg_input_src = Source(data=seg_data)
        always_on_src = Source(data=np.ones((1, total_steps), dtype=np.float32))

        lifs: Dict[int, object] = {}
        sinks: Dict[int, object] = {}
        core_out_sizes: Dict[int, int] = {}

        # Cores in allocation order; wire latency-independent since Lava
        # resolves timing via the graph connections.
        for core_idx, core in enumerate(seg.cores):
            n_out = int(core.neurons_per_core)
            core_out_sizes[core_idx] = n_out

            bias_vec = (
                np.asarray(core.hardware_bias, dtype=np.float32).reshape(-1)
                if core.hardware_bias is not None
                else np.zeros((n_out,), dtype=np.float32)
            )

            lif = SubLIF(
                shape=(n_out,),
                du=1,
                dv=0,
                vth=float(core.threshold),
                bias_mant=bias_vec,
                reset_interval=T,
                reset_offset=reset_offset,
            )
            lifs[core_idx] = lif

            sink = Sink(shape=(n_out,), buffer=total_steps)
            lif.s_out.connect(sink.a_in)
            sinks[core_idx] = sink

        # Keep a handle on one process so we can drive lif.run()/stop() later.
        anchor = next(iter(lifs.values()))

        def _edge_weights(core, group_spans, src_n: int) -> np.ndarray:
            """Build (n_out, src_n) Dense weights for a (src → core) edge."""
            n_out = int(core.neurons_per_core)
            W = np.zeros((n_out, src_n), dtype=np.float32)
            # core.core_matrix has shape (axons, neurons); row a gives the
            # weights-per-neuron for axon a.
            core_mat = np.asarray(core.core_matrix, dtype=np.float32)
            for sp in group_spans:
                d0 = int(sp.dst_start)
                length = int(sp.length)
                s0 = int(sp.src_start)
                # axon d0..d0+length gets src neurons s0..s0+length
                W[:, s0 : s0 + length] = core_mat[d0 : d0 + length, :].T
            return W

        def _bias_weights(core, group_spans) -> np.ndarray:
            """Collapse all always-on axons into a single column vector.

            Always-on axons fire 1 every cycle; their contribution is just
            the row-sum of ``core_matrix`` at those axon indices.
            """
            n_out = int(core.neurons_per_core)
            W = np.zeros((n_out, 1), dtype=np.float32)
            core_mat = np.asarray(core.core_matrix, dtype=np.float32)
            for sp in group_spans:
                d0 = int(sp.dst_start)
                length = int(sp.length)
                W[:, 0] += core_mat[d0 : d0 + length, :].sum(axis=0)
            return W

        # Keep references so Lava doesn't GC edge processes before run.
        all_edges = []

        for core_idx, core in enumerate(seg.cores):
            lif_k = lifs[core_idx]
            spans = core.get_axon_source_spans()

            # Group spans by (kind, src_core).  ``on``/``off`` are handled
            # specially below.
            groups: Dict[tuple, list] = defaultdict(list)
            for sp in spans:
                if sp.kind == "off":
                    continue
                groups[(sp.kind, int(sp.src_core))].append(sp)

            for (kind, src_core_id), span_list in groups.items():
                if kind == "on":
                    W = _bias_weights(core, span_list)
                    dense = Dense(weights=W)
                    always_on_src.s_out.connect(dense.s_in)
                    dense.a_out.connect(lif_k.a_in)
                    all_edges.append(dense)
                elif kind == "input":
                    W = _edge_weights(core, span_list, seg_in_size)
                    dense = Dense(weights=W)
                    seg_input_src.s_out.connect(dense.s_in)
                    dense.a_out.connect(lif_k.a_in)
                    all_edges.append(dense)
                elif kind == "core":
                    if src_core_id not in lifs:
                        raise RuntimeError(
                            f"Segment references unknown source core id {src_core_id}"
                        )
                    src_n = core_out_sizes[src_core_id]
                    W = _edge_weights(core, span_list, src_n)
                    dense = Dense(weights=W)
                    lifs[src_core_id].s_out.connect(dense.s_in)
                    dense.a_out.connect(lif_k.a_in)
                    all_edges.append(dense)
                else:
                    raise ValueError(f"Unknown span kind: {kind}")

        # ------------------------------------------------------------------
        # Execute.
        # ------------------------------------------------------------------

        import time as _time
        t0 = _time.time()
        try:
            anchor.run(
                condition=RunSteps(num_steps=total_steps),
                run_cfg=Loihi2SimCfg(select_tag="floating_pt"),
            )
            # Pull each LIF's spike train off the sink.
            core_output_spikes: Dict[int, np.ndarray] = {}
            start = PAD_HEAD + T + PIPELINE_DELAY
            for core_idx, sink in sinks.items():
                raw = np.asarray(sink.data.get(), dtype=np.float32)
                core_output_spikes[core_idx] = raw[:, start : start + N * T]
        finally:
            anchor.stop()
        print(
            f"  [LavaLoihiRunner] segment run: {len(lifs)} cores, "
            f"{len(all_edges)} dense edges, {total_steps} cycles, "
            f"{_time.time() - t0:.1f}s"
        )

        # ------------------------------------------------------------------
        # Gather the segment's output rates per ``seg.output_sources``.
        # ------------------------------------------------------------------

        out_size = len(seg.output_sources)
        seg_out_spikes = np.zeros((out_size, N * T), dtype=np.float32)
        out_spans = compress_spike_sources(seg.output_sources)
        for sp in out_spans:
            d0 = int(sp.dst_start)
            d1 = int(sp.dst_end)
            if sp.kind == "off":
                continue
            if sp.kind == "on":
                seg_out_spikes[d0:d1, :] = 1.0
                continue
            if sp.kind == "input":
                seg_out_spikes[d0:d1, :] = packed_input[
                    int(sp.src_start) : int(sp.src_end), :
                ]
                continue
            src_core_id = int(sp.src_core)
            seg_out_spikes[d0:d1, :] = core_output_spikes[src_core_id][
                int(sp.src_start) : int(sp.src_end), :
            ]

        seg_out_rates = seg_out_spikes.reshape(out_size, N, T).mean(axis=2).T
        return seg_out_rates.astype(np.float32)

    # -------------------------------------------------------- top-level run

    def run(self) -> float:
        t_total = time.time()

        x_np, y_np = self._load_test_samples()
        N = int(x_np.shape[0])
        print(f"[LavaLoihiRunner] Loaded {N} test samples; T={self.T}")

        x_flat = self._preprocess(x_np)
        state_buffer: Dict[int, np.ndarray] = {-2: x_flat}

        is_ttfs = False  # runner is LIF-only
        out_scales = getattr(self.mapping, "node_activation_scales", {})
        in_scales = getattr(self.mapping, "node_input_activation_scales", out_scales)

        for stage in self.mapping.stages:
            t0 = time.time()
            if stage.kind == "neural":
                seg = stage.hard_core_mapping
                assert seg is not None
                seg_input = SimulationRunner._assemble_segment_input_np(
                    stage.input_map, state_buffer, N
                )
                seg_output = self._run_neural_segment(seg, seg_input)
                SimulationRunner._store_segment_output_np(
                    stage.output_map, state_buffer, seg_output
                )
                self._profile.stages.append(
                    _StageTrace(
                        name=stage.name, kind="neural",
                        seconds=time.time() - t0, cores=len(seg.cores), samples=N,
                    )
                )
            elif stage.kind == "compute":
                assert stage.compute_op is not None
                op_id = stage.compute_op.id
                ttfs_in_scale = in_scales.get(op_id, 1.0) if is_ttfs else 1.0
                ttfs_out_scale = out_scales.get(op_id, 1.0) if is_ttfs else 1.0
                result = SimulationRunner._execute_compute_op_np(
                    stage.compute_op, x_flat, state_buffer,
                    ttfs_in_scale=ttfs_in_scale,
                    ttfs_out_scale=ttfs_out_scale,
                )
                state_buffer[op_id] = result
                self._profile.stages.append(
                    _StageTrace(
                        name=stage.name, kind="compute",
                        seconds=time.time() - t0, samples=N,
                    )
                )
            else:
                raise ValueError(f"Unknown HybridStage kind: {stage.kind}")

        final = SimulationRunner._gather_final_output_np(
            self.mapping.output_sources, state_buffer, x_flat, N
        )
        preds = np.argmax(final, axis=1)
        correct = int((preds == y_np).sum())
        self._accuracy = correct / max(1, N)

        self._profile.total_seconds = time.time() - t_total
        self._profile.log()
        self.pipeline.reporter.report("Loihi Simulation", self._accuracy)
        return self._accuracy

    @property
    def accuracy(self) -> float | None:
        return self._accuracy
