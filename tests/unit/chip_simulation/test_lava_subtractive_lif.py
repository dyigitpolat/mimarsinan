"""SubtractiveLIFReset process-model exactness against the reference executor.

The toolchain's Lava executor (``LavaCoreMixin._run_core_lava``: Source -> Dense ->
SubtractiveLIFReset -> Sink, with its 2-cycle latency pad and one warm-up window) must
reproduce, spike for spike at atol 0, the reference executor's single-core cycle
(``lif_core_contribute_and_fire``, the hardware-core model's step: integrate ``W x + b``,
compare by the declared ``thresholding_mode``, reset by the declared ``firing_mode``),
with the membrane cleared at each sample. This is the critical correctness check for the
Loihi simulation path: if it fails, the Lava runtime does not run the trained model's
forward and no accuracy target is reachable.

The comparator and the reset come from ONE ``NeuralBehaviorConfig`` that the executor and
the reference both receive, so no expectation can encode a comparator the executor did not
run. The seven original cases read the deployment default (``thresholding_mode`` "<=",
inclusive); their expectations had been computed with a strict ``v > vth`` reference,
written at a48cf940 when the hardware-core model's and Lava's default comparator was "<",
and left behind when 80f644fd moved both defaults to "<=" together.

Every slow test here builds a fresh Lava process graph through the executor (the start-method
guard precedes every lava import there; see ``test_lava_start_method_isolation.py``), so
they are marked ``pytest.mark.slow``, and each holds the ``lava_graph_slot`` (conftest): the
graphs run one at a time across xdist workers, since four at once reached a 150-task
process-tree cap.
"""

from __future__ import annotations

import numpy as np
import pytest
from spiking_points import declaring_fire_while_integrating
import torch

from mimarsinan.chip_simulation.behavior_config import NeuralBehaviorConfig
from mimarsinan.chip_simulation.lava_loihi import _uniform_rate_encode
from mimarsinan.chip_simulation.lava_loihi.core_worker import _WaveCoreHost
from mimarsinan.models.spiking.lif_core_step import lif_core_contribute_and_fire

COMPARATORS = ("<", "<=")


def _behavior(*, firing_mode: str = "Default", thresholding_mode: str | None = None) -> NeuralBehaviorConfig:
    """The deployment's behavior; ``thresholding_mode`` None keeps the deployment default."""
    cfg = {"spiking_mode": "lif", "firing_mode": firing_mode, "spike_generation_mode": "Uniform"}
    if thresholding_mode is not None:
        cfg["thresholding_mode"] = thresholding_mode
    return NeuralBehaviorConfig.for_lava(declaring_fire_while_integrating(cfg))  # [U-LIF2 W16] Lava fires per cycle


def _reference_spikes(
    weights: np.ndarray,
    vth: float,
    spikes: np.ndarray,
    behavior: NeuralBehaviorConfig,
    *,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """``(N, n_in, T)`` spikes -> ``(n_out, N*T)`` by the reference executor's cycle."""
    n_samples, _, T = spikes.shape
    n_out = weights.shape[0]
    w = torch.tensor(weights, dtype=torch.float64)
    theta = torch.tensor(float(vth), dtype=torch.float64)
    hw_bias = None if bias is None else torch.tensor(bias, dtype=torch.float64).reshape(1, -1)
    out = np.zeros((n_out, n_samples * T), dtype=np.float64)
    for i in range(n_samples):
        memb = torch.zeros((1, n_out), dtype=torch.float64)
        for cycle in range(T):
            inp = torch.tensor(spikes[i, :, cycle], dtype=torch.float64).reshape(1, -1)
            fired = lif_core_contribute_and_fire(
                memb, w, inp, theta,
                hw_bias=hw_bias,
                thresholding_mode=behavior.thresholding_mode,
                firing_mode=behavior.firing_mode,
                output_dtype=torch.float64,
            )
            out[:, i * T + cycle] = fired.reshape(-1).numpy()
    return out


def _assert_executor_exact(
    weights,
    vth: float,
    rates,
    T: int,
    behavior: NeuralBehaviorConfig,
    *,
    bias=None,
) -> None:
    """Run the executor on uniform-rate inputs and compare every spike with the reference."""
    weights = np.asarray(weights, dtype=np.float32)
    rate_array = np.asarray(rates, dtype=np.float32)
    bias_array = None if bias is None else np.asarray(bias, dtype=np.float32)
    spikes = _uniform_rate_encode(rate_array, T)  # (N, n_in, T)
    packed = spikes.transpose(1, 0, 2).reshape(weights.shape[1], -1)

    actual = np.asarray(
        _WaveCoreHost(T, behavior)._run_core_lava(
            weights=weights,
            threshold=vth,
            hardware_bias=bias_array,
            input_spikes=packed,
        ),
        dtype=np.float64,
    )
    expected = _reference_spikes(
        weights.astype(np.float64), vth, spikes.astype(np.float64), behavior,
        bias=None if bias_array is None else bias_array.astype(np.float64),
    )
    n_samples = rate_array.shape[0]
    np.testing.assert_array_equal(
        actual, expected,
        err_msg=(
            f"{behavior.firing_mode} reset, thresholding_mode={behavior.thresholding_mode!r}: "
            f"rates per (out, sample) expected={expected.reshape(-1, n_samples, T).mean(axis=2).tolist()}, "
            f"got={actual.reshape(-1, n_samples, T).mean(axis=2).tolist()}, input rates={rate_array.tolist()}"
        ),
    )


_SINGLE_RATES = [
    [1.0],
    [0.5],
    [0.25],
    [0.0],
    [1.0, 0.5, 0.25, 0.0],
]

# (name, weights, vth, per-sample input rates, T, firing_mode, hardware bias)
_CASES = [
    *[
        (f"single{i}", [[1.0]], 1.0, [[r] for r in rates], 4, "Default", None)
        for i, rates in enumerate(_SINGLE_RATES)
    ],
    ("multi", [[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]], 2.0,
     [[1.0, 0.5, 0.25], [0.5, 0.5, 0.5]], 8, "Default", None),
    ("novena", [[1.0]], 1.0, [[1.0]], 4, "Novena", None),
    ("bias", [[1.0, -1.0], [0.5, 1.0]], 2.0,
     [[1.0, 0.25], [0.5, 0.75]], 8, "Default", [0.5, -0.25]),
]


@pytest.mark.slow
@pytest.mark.parametrize("rates", _SINGLE_RATES)
def test_single_core_matches_reference_staircase(rates, lava_graph_slot):
    """One neuron, one input, constant per-sample rate -> the reference's spikes, at the deployment default."""
    _assert_executor_exact([[1.0]], 1.0, [[r] for r in rates], 4, _behavior())


@pytest.mark.slow
def test_multi_input_multi_output_matches_reference(lava_graph_slot):
    """3 inputs -> 2 outputs, mixed weights, at the deployment default."""
    _assert_executor_exact(
        [[1.0, 1.0, 1.0], [2.0, 0.0, 1.0]], 2.0,
        [[1.0, 0.5, 0.25], [0.5, 0.5, 0.5]], 8, _behavior(),
    )


@pytest.mark.slow
def test_novena_zero_reset_matches_reference(lava_graph_slot):
    """Novena zero-reset semantics (``firing_mode='Novena'``) at the deployment default comparator."""
    _assert_executor_exact([[1.0]], 1.0, [[1.0]], 4, _behavior(firing_mode="Novena"))


@pytest.mark.slow
@pytest.mark.parametrize("thresholding_mode", COMPARATORS)
@pytest.mark.parametrize(
    "name,weights,vth,rates,T,firing_mode,bias", _CASES, ids=[c[0] for c in _CASES],
)
def test_each_comparator_is_exact_to_the_reference(
    name, weights, vth, rates, T, firing_mode, bias, thresholding_mode, lava_graph_slot,
):
    """Every case under each declared comparator (strict and inclusive), hardware bias included."""
    _assert_executor_exact(
        weights, vth, rates, T,
        _behavior(firing_mode=firing_mode, thresholding_mode=thresholding_mode),
        bias=bias,
    )


def test_the_original_cases_read_the_deployment_default_comparator():
    """The seven original cases run the comparator a deployment runs when it declares none."""
    default = NeuralBehaviorConfig.from_deployment_config(declaring_fire_while_integrating({}))  # Lava fires per cycle
    assert _behavior().thresholding_mode == default.thresholding_mode == "<="
