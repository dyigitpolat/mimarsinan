"""Phase 0 characterization oracles for the layout single-source refactor.

These tests pin the *current* (pre-refactor) placement decisions and simulation
outputs as golden files.  Every later phase of the refactor must keep them
byte-for-byte identical -- they are the bit-identical contract.

Regenerate goldens (only when an intended behavioural change is reviewed):

    MIMARSINAN_REGEN_GOLDEN=1 python -m pytest \
        tests/integration/test_layout_placement_characterization.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from integration._placement_signature import CONFIGS, signature_for_config


_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "layout_characterization"
_PLACEMENT_GOLDEN = _FIXTURE_DIR / "placement_signatures.json"
_SIM_GOLDEN = _FIXTURE_DIR / "simulation_outputs.json"

_REGEN = os.environ.get("MIMARSINAN_REGEN_GOLDEN") == "1"


def _load_golden(path: Path) -> dict:
    if not path.exists():
        pytest.skip(
            f"golden {path.name} missing; run with MIMARSINAN_REGEN_GOLDEN=1 to create"
        )
    return json.loads(path.read_text())


# ── Placement signatures ───────────────────────────────────────────────────

def _all_placement_signatures() -> dict:
    return {name: signature_for_config(name) for name in CONFIGS}


def test_regenerate_goldens_if_requested() -> None:
    if not _REGEN:
        pytest.skip("set MIMARSINAN_REGEN_GOLDEN=1 to (re)generate goldens")
    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    _PLACEMENT_GOLDEN.write_text(json.dumps(_all_placement_signatures(), indent=2))
    _SIM_GOLDEN.write_text(json.dumps(_compute_simulation_outputs(), indent=2))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_placement_signature_matches_golden(name: str) -> None:
    if _REGEN:
        pytest.skip("regen mode")
    golden = _load_golden(_PLACEMENT_GOLDEN)
    assert name in golden, f"no golden for config {name!r}; regenerate goldens"
    assert signature_for_config(name) == golden[name], (
        f"placement signature drift for config {name!r}"
    )


# ── Simulation outputs (bit-identical HCM forward) ─────────────────────────

def _build_mlpmixer_hcm(placement: str, T: int = 4):
    import torch.nn as nn

    from mimarsinan.models.torch_mlp_mixer_core import TorchMLPMixerCore
    from mimarsinan.torch_mapping.converter import convert_torch_model
    from mimarsinan.models.nn.activations import LIFActivation
    from mimarsinan.mapping.ir_mapping_class import IRMapping
    from mimarsinan.mapping.packing.hybrid_hardcore_mapping import (
        build_hybrid_hard_core_mapping,
    )
    from mimarsinan.mapping.platform.mapping_structure import MappingStrategy
    from mimarsinan.models.spiking.hybrid.flow import SpikingHybridCoreFlow

    torch.manual_seed(0)
    m = TorchMLPMixerCore(
        input_shape=(1, 28, 28), num_classes=10,
        patch_n_1=4, patch_m_1=4, patch_c_1=6, fc_w_1=8, fc_w_2=6,
    )
    m.eval()
    flow = convert_torch_model(
        m, input_shape=(1, 28, 28), num_classes=10,
        encoding_layer_placement=placement,
    )
    repr_ = flow.get_mapper_repr()
    for p in flow.get_perceptrons():
        lif = LIFActivation(T=T, activation_scale=torch.tensor(1.0), thresholding_mode="<=")
        p.base_activation = lif
        p.activation = lif
    repr_.assign_perceptron_indices()
    ir = IRMapping(
        q_max=127.0, firing_mode="Default", max_axons=512, max_neurons=512,
    ).map(repr_)
    hybrid = build_hybrid_hard_core_mapping(
        ir_graph=ir, cores_config=[{"max_axons": 512, "max_neurons": 512, "count": 400}],
        strategy=MappingStrategy.from_permissions(allow_neuron_splitting=True),
    )
    return SpikingHybridCoreFlow(
        (1, 28, 28), hybrid, simulation_length=T, preprocessor=nn.Identity(),
        firing_mode="Default", spike_mode="Uniform", thresholding_mode="<=",
        spiking_mode="lif", cycle_accurate_lif_forward=True,
    )


def _compute_simulation_outputs() -> dict:
    torch.manual_seed(123)
    x = torch.rand(1, 1, 28, 28)
    out = {}
    for placement in ("subsume", "offload"):
        flow = _build_mlpmixer_hcm(placement)
        with torch.no_grad():
            y = flow(x)
        out[placement] = y.detach().cpu().reshape(-1).tolist()
    return out


@pytest.mark.parametrize("placement", ["subsume", "offload"])
def test_simulation_output_matches_golden(placement: str) -> None:
    if _REGEN:
        pytest.skip("regen mode")
    golden = _load_golden(_SIM_GOLDEN)
    assert placement in golden, f"no sim golden for {placement!r}; regenerate goldens"

    torch.manual_seed(123)
    x = torch.rand(1, 1, 28, 28)
    flow = _build_mlpmixer_hcm(placement)
    with torch.no_grad():
        y = flow(x).detach().cpu().reshape(-1)
    expected = torch.tensor(golden[placement], dtype=y.dtype)
    torch.testing.assert_close(y, expected, atol=1e-6, rtol=0.0)
