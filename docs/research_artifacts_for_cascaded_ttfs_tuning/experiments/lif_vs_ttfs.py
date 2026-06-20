"""LOCALIZE the cratering: LIF-rate vs TTFS-staircase vs TTFS-genuine on the
IDENTICAL converted flow (same weights, same calibration), across depth x S.

User's claim: the TTFS-staircase cratering at low-S/deep is NOT fundamental
quantization, because LIF (same T+1 per-neuron levels) reaches ~0.97 on the real
pipeline. This isolates the question on one model:

  - TTFS-staircase : analytical round-staircase decode (cycle_accurate OFF), the
                     optimal LINEAR single-spike timing decode.  relu(x)/theta ->
                     clamp[0,1] -> round to T+1 levels -> *theta.
  - LIF-rate       : analytical multi-step rate (T-step signed IF integrate),
                     count/T -> *theta.  T+1 levels, SAME granularity.
  - TTFS-genuine   : the deployed single-spike ramp-integrate cascade.

If LIF-rate >> TTFS-staircase cold, the loss is a TTFS-quantizer-specific cold
defect (round/clamp/relu interaction). If LIF-rate ~= TTFS-staircase cold (both
crater), the cold quantizers are equivalent and LIF's 0.97 comes from TUNING --
which then must be ported to the genuine TTFS forward.

    source env/bin/activate
    python docs/research_artifacts_for_cascaded_ttfs_tuning/experiments/lif_vs_ttfs.py
"""

from __future__ import annotations

import copy
import os
import sys

import torch

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))
from ft_budget import DEV, build  # noqa: E402
from cascade_lab import _accuracy  # noqa: E402
from mimarsinan.models.nn.activations.lif import LIFActivation  # noqa: E402
from mimarsinan.models.nn.activations.ttfs_spiking import TTFSActivation  # noqa: E402
from mimarsinan.models.spiking.training.ttfs_segment_forward import TTFSSegmentForward  # noqa: E402


def install_lif_nodes(flow, S):
    """Replace each perceptron's activation with a rate-coded LIFActivation at
    resolution S (reusing the already-calibrated activation_scale)."""
    for p in flow.get_perceptrons():
        p.set_activation(LIFActivation(
            T=S,
            activation_scale=p.activation_scale,
            thresholding_mode="<=",
        ))
    return flow.double()


def lif_rate_acc(flow, x, y, S):
    """Analytical LIF rate forward (value-domain, T+1-level quantizer per layer)."""
    f = copy.deepcopy(flow)
    install_lif_nodes(f, S)
    for m in f.modules():
        if isinstance(m, LIFActivation):
            m.set_cycle_accurate(False)
    f.double().eval()
    with torch.no_grad():
        return _accuracy(f(x.double()), y)


def ttfs_staircase_acc(flow, x, y, S):
    """Analytical TTFS round-staircase forward (the optimal linear timing decode)."""
    f = copy.deepcopy(flow)
    for m in f.modules():
        if isinstance(m, TTFSActivation):
            m.set_cycle_accurate(False)
    f.double().eval()
    with torch.no_grad():
        return _accuracy(f(x.double()), y)


def ttfs_genuine_acc(flow, x, y, S):
    with torch.no_grad():
        return _accuracy(TTFSSegmentForward(flow.get_mapper_repr(), S)(x.double()), y)


if __name__ == "__main__":
    print(f"=== LIF-rate vs TTFS-staircase vs TTFS-genuine (device={DEV}) ===")
    print(f"{'d':>2} {'S':>3} | {'cont':>6} {'LIF_rate':>9} {'TTFS_stair':>11} "
          f"{'TTFS_gen':>9} | {'stair-LIF':>10}")
    for depth in (3, 6, 9):
        for S in (8, 16):
            flow, _xtr, _ytr, xte, yte, cont, _teacher, _base = build(depth, S)
            lif = lif_rate_acc(flow, xte, yte, S)
            stair = ttfs_staircase_acc(flow, xte, yte, S)
            gen = ttfs_genuine_acc(flow, xte, yte, S)
            print(f"{depth:>2} {S:>3} | {cont:>6.3f} {lif:>9.3f} {stair:>11.3f} "
                  f"{gen:>9.3f} | {stair - lif:>+10.3f}")
