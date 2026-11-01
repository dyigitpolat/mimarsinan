"""[U-SC3 A-1] The LIF segment walk streams its trains: bounded memory, outputs bit-equal to the eager walk.

S1's T-LIF session stopped at its 28G cap in Activation Quantization (U-SC2 F-5): per hop the D-hat walk held the whole
``(T, B, C_in, H, W)`` boundary re-encode twice (the 0/1 comb and its scaled copy) beside the hop's output train, so its
largest hop held 29.53 GiB at the 1,024-image validation batch (``phase_c_r/g1_tlif/s1mem/out/s1_accounting.json``).
Every walk of the deployed composition (the AQ D-hat, the LIF finalize twin, LIF Adaptation's metric, the parity
measurement plane, the certificates' walks) is this walk. Where autograd records nothing, the walk now encodes and
consumes each hop's input trains one cycle at a time (``spiking/cycle_trains.py``), and frees every train no later
reader of its segment reads; where autograd records, it keeps the whole trains (autograd saves every cycle's inputs for
the backward pass anyway), so the eager walk is the same walk under ``torch.enable_grad()``.

Peak memory is read from the torch allocator's CPU allocation events (``torch.profiler``, ``profile_memory=True``): the
largest sum of live bytes the walk allocated above its start. The equality tests are guards: the change's contract is
that no output moves, so each passes on the eager tree by construction and pins the streamed walk to it at sha256.

Not bounded here: a LIF activation's rate-mode forward (an encoding layer's value, a synchronized hop's value)
integrates its T steps inside spikingjelly's fused eval kernel on a ``(T, B, ...)`` expansion of its own input,
unchanged by this unit; the forms keep that expansion small beside the trains they bound.
"""

from __future__ import annotations

import contextlib
import hashlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from mimarsinan.models.nn.lif_kernels import measurement_plane
from mimarsinan.spiking.segment_forward import SegmentForwardDriver
from mimarsinan.spiking.segment_partition import is_encoding_perceptron, perceptron_of

from unit.spiking.host_gather_forms import MIXED_FORMS, SEMANTICS, SHAKEDOWN_FORMS, T, build, samples

BATCH = 64
FLOAT32 = 4


def _walk(program, x, semantics: str, *, plane: bool, grad: bool, record: bool = False):
    """``(output, per-perceptron accumulated activation outputs, recorded values)`` of one LIF segment walk: the
    parity gate's capture (``_capture_nf_streamed``'s per-cycle accumulation) and the DFQ recorder."""
    counts: dict = {}
    handles = []
    for index, perceptron in enumerate(program.perceptrons):
        def hook(_module, _inputs, out, index=index):
            counts[index] = out.detach() if index not in counts else counts[index] + out.detach()
        handles.append(perceptron.activation.register_forward_hook(hook))
    recorded: dict | None = {} if record else None
    mode = torch.enable_grad() if grad else torch.no_grad()
    try:
        with mode, (measurement_plane() if plane else contextlib.nullcontext()):
            out = SegmentForwardDriver(program.repr_, T, SEMANTICS[semantics][2]())(x, node_value_recorder=recorded)
    finally:
        for handle in handles:
            handle.remove()
    return out.detach(), counts, {k: v.detach() for k, v in (recorded or {}).items()}


def _sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def _peak_allocated_bytes(fn) -> int:
    """The largest sum of live CPU bytes the torch allocator holds above the call's start, from its allocation and
    free events."""
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        fn()
    events = sorted((e.start_ns(), e.nbytes()) for e in prof.profiler.kineto_results.events() if e.name() == "[memory]")
    live = peak = 0
    for _, nbytes in events:
        live += nbytes
        peak = max(peak, live)
    return peak


def _values(program, x) -> dict:
    """node -> its value on one plain walk of the mapper graph (the shapes every train is T times)."""
    order, deps, _ = program.repr_._ensure_exec_graph()
    values: dict = {}
    with torch.no_grad():
        for node in order:
            d = deps.get(node, [])
            values[node] = node.forward(x) if not d else node.forward(values[d[0]] if len(d) == 1 else
                                                                       tuple(values[k] for k in d))
    return values


def _largest_reencode_bytes(program, x) -> int:
    """The bytes of the largest whole boundary re-encode the walk builds: T x a core's widest input from outside its
    hop chain, or T x an encoding layer's output, in float32."""
    order, deps, _ = program.repr_._ensure_exec_graph()
    values = _values(program, x)
    largest = 0
    for node in order:
        if perceptron_of(node) is None:
            continue
        if is_encoding_perceptron(node):
            largest = max(largest, values[node].numel())
        else:
            largest = max([largest] + [values[d].numel() for d in deps.get(node, [])])
    return T * largest * FLOAT32


def _hop_train_bytes(program, x) -> int:
    """The bytes of the largest hop output train, T x a core's output, in float32."""
    order, _, _ = program.repr_._ensure_exec_graph()
    values = _values(program, x)
    return T * max(values[n].numel() for n in order if perceptron_of(n) is not None) * FLOAT32


# ---------------------------------------------------------------------------
# Bounded memory (red on the eager walk)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plane", (False, True), ids=("tuning_plane", "measurement_plane"))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
def test_the_walk_holds_less_than_one_whole_boundary_re_encode(semantics, plane):
    """S1's dense-unit read: a narrow core reads a wide host value (``relu(bn(cat))``). The eager walk held the comb
    and its scaled copy of that boundary re-encode at once (at least twice its bytes); the streamed walk holds one
    cycle of it."""
    program = build("wide_concat_host_read", 0, 1.6, semantics, False)
    x = samples(program, BATCH, 0)
    bound = _largest_reencode_bytes(program, x)
    peak = _peak_allocated_bytes(lambda: _walk(program, x, semantics, plane=plane, grad=False))
    assert peak < bound, (semantics, plane, peak, bound)


@pytest.mark.parametrize("plane", (False, True), ids=("tuning_plane", "measurement_plane"))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
def test_the_walk_frees_each_hop_train_after_its_last_reader(semantics, plane):
    """Chains of two and six cores in one segment: the eager walk kept every hop's train (and, in the measurement
    plane, every hop's event train) to the segment's end, so four more hops held at least four more trains; the
    streamed walk drops each after its last reader, so its peak does not grow with the chain by one hop train."""
    peaks = {}
    for form in ("chain_2", "chain_6"):
        program = build(form, 0, 1.6, semantics, False)
        x = samples(program, BATCH, 0)
        peaks[form] = _peak_allocated_bytes(lambda: _walk(program, x, semantics, plane=plane, grad=False))
        hop = _hop_train_bytes(program, x)
    assert peaks["chain_6"] - peaks["chain_2"] < hop, (semantics, plane, peaks, hop)


@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
def test_the_measurement_plane_keeps_no_event_train_no_reader_reads(semantics):
    """U-SC2's verify observation: inside the measurement plane a float-mode hop's fold emits its 0/1 events, kept for
    the next hop's fold. The eager walk kept them (the emission list and its stack) even where no later node of the
    segment reads them, so the plane walk held at least one more hop train than the tuning-plane walk; now a hop with
    no reader keeps none, and the plane adds less than half a hop train (the fold's per-cycle tables)."""
    program = build("wide_core_output", 0, 1.6, semantics, False)
    x = samples(program, BATCH, 0)
    hop = _hop_train_bytes(program, x)
    peaks = {plane: _peak_allocated_bytes(lambda plane=plane: _walk(program, x, semantics, plane=plane, grad=False))
             for plane in (False, True)}
    assert peaks[True] - peaks[False] < hop / 2, (semantics, peaks, hop)


# ---------------------------------------------------------------------------
# Bit-equal to the eager walk (guards: equal on the eager tree by construction)
# ---------------------------------------------------------------------------

_EQUALITY_FORMS = SHAKEDOWN_FORMS + tuple(MIXED_FORMS) + ("densenet_bc", "wide_encoding_read",
                                                           "wide_concat_host_read", "wide_core_output", "chain_6")


@pytest.mark.parametrize("plane", (False, True), ids=("tuning_plane", "measurement_plane"))
@pytest.mark.parametrize("wq", (False, True), ids=("float", "wq4"))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
@pytest.mark.parametrize("form", _EQUALITY_FORMS)
def test_the_streamed_walk_is_sha256_equal_to_the_eager_walk(form, semantics, wq, plane):
    """The output (the D-hat logits), every perceptron's per-cycle accumulated outputs (the parity gate's window
    counts) and every recorded value, sha256-equal with and without autograd recording."""
    program = build(form, 0, 1.6, semantics, wq)
    x = samples(program, 16, 0)
    out_s, counts_s, rec_s = _walk(program, x, semantics, plane=plane, grad=False, record=True)
    out_e, counts_e, rec_e = _walk(program, x, semantics, plane=plane, grad=True, record=True)
    assert _sha(out_s) == _sha(out_e)
    assert sorted(counts_s) == sorted(counts_e) and counts_s
    assert {k: _sha(v) for k, v in counts_s.items()} == {k: _sha(v) for k, v in counts_e.items()}
    assert sorted(rec_s) == sorted(rec_e)
    assert {k: _sha(v) for k, v in rec_s.items()} == {k: _sha(v) for k, v in rec_e.items()}


@pytest.mark.parametrize("plane", (False, True), ids=("tuning_plane", "measurement_plane"))
@pytest.mark.parametrize("semantics", ("streamed_lif", "windowed_lif"))
def test_the_streamed_walk_is_sha256_equal_on_more_seeds_and_a_larger_batch(semantics, plane):
    """The same equality on the forms whose re-encodes the streamed walk builds per cycle in row blocks, at a batch
    large enough to take several blocks, on four seeds."""
    for form in ("wide_encoding_read", "wide_concat_host_read", "densenet_bc"):
        for seed in range(4):
            program = build(form, seed, 1.6, semantics, False)
            x = samples(program, BATCH, seed)
            out_s, counts_s, _ = _walk(program, x, semantics, plane=plane, grad=False)
            out_e, counts_e, _ = _walk(program, x, semantics, plane=plane, grad=True)
            assert _sha(out_s) == _sha(out_e), (form, seed)
            assert {k: _sha(v) for k, v in counts_s.items()} == {k: _sha(v) for k, v in counts_e.items()}, (form, seed)
