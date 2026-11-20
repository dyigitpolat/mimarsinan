"""[U-LIF2, the review's p-11(b)] The synchronized count executor computes each window's count by formula, which is
the integrate-then-emit count of a subtractive reset only: an input-free emission window of a zero-reset neuron emits
at most one spike. So the executor refuses a zero reset by name as a guard, whatever builds its flow (the randomized
certification builds flows past the resolution). Red on the set export: the executor runs a zero-reset flow."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW
from mimarsinan.models.spiking.hybrid.executors import sync_counts


def _flow(firing_mode):
    return SimpleNamespace(firing_mode=firing_mode, lif_membrane_init=0.0, thresholding_mode="<=",
                           spiking_mode="lif", ttfs_cycle_schedule="cascaded", soma_law=DEFAULT_SOMA_LAW)


def test_a_zero_reset_flow_is_refused_by_name():
    refusal = getattr(importlib.import_module("mimarsinan.models.spiking.serial.refusals"), "ZeroResetWindowCountError")
    with pytest.raises(refusal, match="firing_mode='Novena'"):
        sync_counts.run_neural_segment_counts(_flow("Novena"), torch.zeros(4, 1, 1), seg={}, T=4, batch_size=1,
                                              device="cpu", soma_law=DEFAULT_SOMA_LAW)


def test_the_subtractive_reset_passes_the_guard():
    with pytest.raises(KeyError):  # past the guard, into the (empty) segment
        sync_counts.run_neural_segment_counts(_flow("Default"), torch.zeros(4, 1, 1), seg={}, T=4, batch_size=1,
                                              device="cpu", soma_law=DEFAULT_SOMA_LAW)


def test_the_docstring_states_the_reset_the_closed_form_needs():
    doc = " ".join((sync_counts.run_neural_segment_counts.__doc__ or "").split())
    assert "subtractive-or-zero" not in doc and "zero-reset emission window emits at most one spike" in doc
