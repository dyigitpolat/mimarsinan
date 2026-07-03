"""DFQ per-neuron bias correction for the deployed LIF cascade."""

from __future__ import annotations

import torch

from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.dfq_bias_correction import (
    dfq_correct_biases,
    teacher_channel_means,
)


def _lif_cascade_channel_means(model, cal_x, T) -> dict:
    """Per-perceptron deployed-cascade decoded value, keyed by perceptron index.

    Reads the values the LIF segment policy records through the forward's
    ``node_value_recorder`` side-channel — the exact deployed dynamics.
    """
    recorder: dict = {}
    with torch.no_grad():
        chip_aligned_segment_forward(model, cal_x, int(T), node_value_recorder=recorder)
    out: dict = {}
    for k, perceptron in enumerate(model.get_perceptrons()):
        value = recorder.get(id(perceptron))
        if value is not None:
            out[k] = value
    return out


def match_lif_activation_distributions(
    model, teacher, cal_x, T, *, bias_iters: int = 10, eta: float = 0.5,
) -> dict:
    """Match the deployed LIF cascade's per-neuron mean to the teacher ANN's.

    Runs ``bias_iters`` rounds of DFQ bias correction over the deployed
    cycle-accurate cascade. Mutates the model in place; returns a stats dict.
    """
    T = int(T)
    ann_mean = teacher_channel_means(teacher, cal_x)
    stats = dfq_correct_biases(
        model,
        ann_mean,
        lambda: _lif_cascade_channel_means(model, cal_x, T),
        bias_iters=bias_iters,
        eta=eta,
    )
    stats.update({
        "bias_iters": int(bias_iters),
        "eta": float(eta),
        "num_perceptrons": len(list(model.get_perceptrons())),
    })
    return stats
