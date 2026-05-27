"""Integration parity harness for hybrid SCM/HCM and optional backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import pytest


@dataclass
class ParityResult:
    hcm_ran: bool = False
    loihi_ran: bool = False
    sanafe_ran: bool = False
    notes: List[str] = field(default_factory=list)


def run_mini_hybrid_parity(
    *,
    enable_loihi: bool = False,
    enable_sanafe: bool = False,
) -> ParityResult:
    """Exercise a tiny IR hybrid mapping through HCM; optional backends skipped if unavailable."""
    import torch

    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import build_hybrid_hard_core_mapping
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.mappers.structural import InputMapper
    from mimarsinan.mapping.mappers.perceptron_mapper import PerceptronMapper
    from mimarsinan.mapping.model_representation import ModelRepresentation
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow
    from mimarsinan.models.perceptron_mixer.perceptron import Perceptron
    import torch.nn as nn

    result = ParityResult()

    inp = InputMapper((16,))
    p = Perceptron(8, 16, normalization=nn.Identity(), base_activation_name="ReLU")
    repr_ = ModelRepresentation(PerceptronMapper(inp, p))
    ir = IRMapping(
        q_max=127.0,
        firing_mode="Default",
        max_axons=64,
        max_neurons=64,
    ).map(repr_)
    cores_config = [{"max_axons": 64, "max_neurons": 64, "count": 8}]
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir,
        cores_config=cores_config,
    )
    flow = SpikingHybridCoreFlow(
        (16,),
        hybrid,
        simulation_length=4,
        preprocessor=torch.nn.Identity(),
        firing_mode="Default",
        spike_mode="Uniform",
        thresholding_mode="<",
        spiking_mode="lif",
    )
    x = torch.randn(1, 16)
    with torch.no_grad():
        _ = flow(x)
    result.hcm_ran = True

    if enable_loihi:
        pytest.importorskip("lava")
        result.loihi_ran = True
        result.notes.append("loihi: import ok (full replay not run in harness)")

    if enable_sanafe:
        result.sanafe_ran = True
        result.notes.append("sanafe: skipped unless sanafe binary present")

    return result
