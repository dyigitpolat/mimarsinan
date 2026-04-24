"""SubtractiveLIFReset process-model correctness.

Pin that a Lava SubtractiveLIFReset neuron fed through a Dense matches the
T-level saturated staircase rate that ``LIFActivation`` / ``nevresim``
``firing_mode='Default'`` produce for the same inputs.  This is the
critical correctness check for the Loihi simulation path: if this fails,
the Lava runtime is not matching our trained model's forward and no
accuracy target is reachable.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reference_rate(w_dot_x: float, vth: float, T: int) -> float:
    """Reference: subtractive-reset IF rate = min(floor(T*V/vth), T) / T."""
    if w_dot_x <= 0.0:
        return 0.0
    return min(np.floor(T * w_dot_x / vth), float(T)) / float(T)


def _run_single_core_lava(
    weights: np.ndarray,
    threshold: float,
    input_spikes: np.ndarray,
    T: int,
) -> np.ndarray:
    """Build Source→Dense→SubtractiveLIFReset→Sink and return output spikes.

    Uses the same 2-cycle latency pad + 1-window warmup layout as
    ``LavaLoihiRunner._run_core_lava``.
    """
    from lava.magma.core.run_conditions import RunSteps
    from lava.magma.core.run_configs import Loihi2SimCfg
    from lava.proc.dense.process import Dense
    from lava.proc.io.sink import RingBuffer as Sink
    from lava.proc.io.source import RingBuffer as Source

    from mimarsinan.chip_simulation.lava_loihi_runner import _subtractive_lif_cls

    SubLIF = _subtractive_lif_cls()
    total = input_spikes.shape[1]
    assert total % T == 0
    N = total // T

    PAD_HEAD = 2
    TAIL = 3
    PIPELINE_DELAY = 1
    pad_head_block = np.zeros((input_spikes.shape[0], PAD_HEAD), dtype=np.float32)
    warmup = input_spikes[:, :T]
    tail_block = np.zeros((input_spikes.shape[0], TAIL), dtype=np.float32)
    data = np.concatenate(
        [pad_head_block, warmup, input_spikes, tail_block], axis=1
    ).astype(np.float32)
    total_steps = data.shape[1]
    reset_offset = (PAD_HEAD + T + PIPELINE_DELAY) % T

    n_out = weights.shape[0]
    src = Source(data=data)
    dense = Dense(weights=weights.astype(np.float32))
    lif = SubLIF(
        shape=(n_out,),
        du=1, dv=0,
        vth=float(threshold),
        bias_mant=np.zeros((n_out,), dtype=np.float32),
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
        raw = sink.data.get()
    finally:
        lif.stop()

    start = PAD_HEAD + T + PIPELINE_DELAY
    return np.asarray(raw[:, start : start + N * T], dtype=np.float32)


@pytest.mark.parametrize(
    "rates",
    [
        [1.0],
        [0.5],
        [0.25],
        [0.0],
        [1.0, 0.5, 0.25, 0.0],
    ],
)
def test_single_core_matches_reference_staircase(rates):
    """One neuron, one input, constant per-sample rate → matches reference."""
    from mimarsinan.chip_simulation.lava_loihi_runner import _uniform_rate_encode

    T = 4
    vth = 1.0
    weights = np.array([[1.0]], dtype=np.float32)  # one input → one output
    N = len(rates)

    rate_array = np.array(rates, dtype=np.float32).reshape(N, 1)
    spikes = _uniform_rate_encode(rate_array, T)  # (N, 1, T)
    packed = spikes.transpose(1, 0, 2).reshape(1, N * T)

    out = _run_single_core_lava(weights, vth, packed, T)  # (1, N*T)

    for i, r in enumerate(rates):
        actual_rate = float(out[0, i * T : (i + 1) * T].mean())
        expected = _reference_rate(r, vth, T)
        assert actual_rate == pytest.approx(expected, abs=1e-5), (
            f"sample {i}: input_rate={r}, expected={expected}, got={actual_rate}"
        )


def test_multi_input_multi_output_matches_reference():
    """3 inputs → 2 outputs, mixed weights."""
    from mimarsinan.chip_simulation.lava_loihi_runner import _uniform_rate_encode

    T = 8
    vth = 2.0
    weights = np.array(
        [[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]], dtype=np.float32
    )  # out0 sums all, out1 weighted
    rates = np.array([[1.0, 0.5, 0.25], [0.5, 0.5, 0.5]], dtype=np.float32)
    N = rates.shape[0]

    spikes = _uniform_rate_encode(rates, T)
    packed = spikes.transpose(1, 0, 2).reshape(3, N * T)

    out = _run_single_core_lava(weights, vth, packed, T)

    for i in range(N):
        for j in range(2):
            actual = float(out[j, i * T : (i + 1) * T].mean())
            # The Lava pipeline integrates over spikes that arrived each cycle,
            # so the true expected is min(floor(T*sum_spikes/(vth)), T)/T where
            # sum_spikes = Σ_cycles (W @ spike_vector_at_cycle).  For uniform
            # encoding the time-averaged contribution equals W @ rates.
            expected_current = float(weights[j] @ rates[i])
            expected = _reference_rate(expected_current, vth, T)
            # Tolerance: uniform-encoding quantization can shift one spike
            # either way across T cycles when inputs aren't commensurate with
            # T, so allow ±1/T drift.
            assert actual == pytest.approx(expected, abs=1.0 / T + 1e-5), (
                f"(sample {i}, out {j}): expected={expected}, got={actual}, "
                f"current={expected_current}, rates={rates[i]}"
            )
