"""[t0_54 regression] the NF twin's law comes from the DEPLOYMENT, never a default.

The tier cell ``t0_54`` (lifse / simple_mlp / wq / S=4, the ``per_event`` point)
died RED at Soft Core Mapping with 1004/3152 per-cycle emission mismatches —
worst nf=1 vs scm=7 — because the chip-aligned NF forward the LIF adaptation
step INSTALLS on the model was constructed without the resolved ``SomaLaw``.
It silently ran the default per-cycle law (one spike per cycle, by
construction) while the deployment folded the event-serial law. These tests pin
the two halves of the fix: the point is default-free everywhere the twin is
built, and the fixture that reproduces the mismatch stays discriminating.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mimarsinan.chip_simulation.soma_law import DEFAULT_SOMA_LAW, SomaLaw
from mimarsinan.models.nn.activations.lif_serial import (
    _DECOMPOSITION_ATOL,
    SerialFoldSlot,
)
from mimarsinan.models.spiking.serial import (
    CycleAtomicRefusalError,
    SerialDecompositionMismatchError,
)
from mimarsinan.pipelining.core import nf_scm_parity
from mimarsinan.spiking.chip_aligned_nf import chip_aligned_segment_forward
from mimarsinan.spiking.segment_forward import LifSegmentPolicy
from mimarsinan.tuning.forward_install import (
    ChipAlignedNFForward,
    SomaLawMissingError,
)

from .test_serial_executors import (
    PER_EVENT_LAW,
    PER_EVENT_UNBOUNDED,
    T,
    build_chain,
)
from .test_serial_nf_twin import _nf_model

_PER_EVENT_CONFIG = {
    "spiking_family": "lif", "spiking_variant": "streamed",
    "firing_granularity": "per_event", "thresholding_mode": "<=",
    "membrane_bits": 8, "simulation_steps": T,
}


# ── the install carries the point ────────────────────────────────────────────

def _lif_tuner_stub(config: dict):
    import types

    class _Model(nn.Module):
        def get_perceptrons(self):
            return []

    return types.SimpleNamespace(
        model=_Model(), _T=T, _cycle_accurate=True, _per_hop_retiming=False,
        _phase_dither=False, _synchronized=False,
        pipeline=types.SimpleNamespace(config=dict(config)),
    )


def test_the_finalize_install_carries_the_deployments_resolved_point():
    """The t0_54 defect, pinned at its origin: the installed forward IS the
    twin every later stage runs, so its law must be the deployment's."""
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    stub = _lif_tuner_stub(_PER_EVENT_CONFIG)
    forward = LIFAdaptationTuner._finalize_forward_for(stub, stub.model)
    assert forward.soma_law == SomaLaw.resolve(_PER_EVENT_CONFIG)
    assert forward.soma_law.is_per_event


def test_a_default_point_deployment_still_installs_the_default_law():
    from mimarsinan.tuning.tuners.lif_adaptation_tuner import LIFAdaptationTuner

    stub = _lif_tuner_stub({"simulation_steps": T})
    forward = LIFAdaptationTuner._finalize_forward_for(stub, stub.model)
    assert forward.soma_law == DEFAULT_SOMA_LAW


def test_the_nf_forward_cannot_be_built_without_a_point():
    with pytest.raises(TypeError, match="soma_law"):
        ChipAlignedNFForward(nn.Identity(), T)  # type: ignore[call-arg]


def test_the_walk_cannot_be_run_without_a_point():
    with pytest.raises(TypeError, match="soma_law"):
        chip_aligned_segment_forward(  # type: ignore[call-arg]
            nn.Identity(), torch.zeros(1, 8), T)


def test_a_pre_axes_cache_artifact_refuses_instead_of_defaulting():
    """An unpickled forward with no law must NOT quietly become the default
    point — that is the same silent wrong-physics the cell died of."""
    forward = ChipAlignedNFForward(nn.Identity(), T, soma_law=PER_EVENT_LAW)
    del forward.__dict__["soma_law"]
    with pytest.raises(SomaLawMissingError, match="no soma_law"):
        forward(torch.zeros(1, 8))


# ── the fixture that reproduces nf=1 vs scm=7 ────────────────────────────────

def _nf_rasters(repr_, x, *, soma_law):
    model = _nf_model(repr_)
    model.forward = ChipAlignedNFForward(model, T, soma_law=soma_law)
    return nf_scm_parity._capture_nf_streamed_rasters(model, x)


def _sample():
    torch.manual_seed(3)
    return torch.rand(4, 8) * 0.9


def test_the_default_law_twin_emits_at_most_one_spike_per_cycle():
    """The t0_54 signature at fixture scale: every NF raster entry is 0 or 1,
    whatever the deployment's law is."""
    repr_, _ = build_chain()
    rasters = _nf_rasters(repr_, _sample(), soma_law=DEFAULT_SOMA_LAW)
    assert rasters, "the streamed capture must see the hops"
    assert max(float(r.max()) for r in rasters.values()) == 1.0


def test_the_point_twin_emits_the_serial_multiplicity():
    """Teeth for the test above: on the SAME fixture the per-event law fires
    more than once in a cycle, so the two laws are distinguishable here."""
    repr_, _ = build_chain()
    rasters = _nf_rasters(repr_, _sample(), soma_law=PER_EVENT_UNBOUNDED)
    assert max(float(r.max()) for r in rasters.values()) >= 2.0


# ── the wide-fan-in decomposition residual ───────────────────────────────────

def _wide_slot(theta: float, n_slots: int, law: SomaLaw = PER_EVENT_LAW):
    """t0_54's real shape: a WQ hop's effective weights are integers over a
    scale, so ``weight * theta`` is not exactly representable and the two
    reductions round apart."""
    torch.manual_seed(5)
    weight = torch.randint(-4, 5, (8, n_slots)).float() / 7.0
    bias = torch.randint(-2, 3, (8,)).float() / 7.0
    events = torch.randint(0, 4, (2, n_slots)).float()
    slot = SerialFoldSlot(
        soma_law=law, weight=weight, bias=bias, theta=theta, membrane_init=0.0,
    )
    fused = torch.nn.functional.linear(events, weight, bias)
    return slot, events, fused


def test_a_wide_hop_survives_the_float32_reduction_residual():
    """t0_54's real hops are 128-256 slots wide with theta up to 37: the two
    float32 reductions differ by ~1.5e-7 RELATIVE, which an absolute-only
    tolerance reads as a slot-order defect."""
    slot, events, fused = _wide_slot(37.0, 256)
    slot.feed(events)
    slot.run_cycle(fused, 1.0)


def test_the_wide_hop_residual_really_exceeds_the_absolute_tolerance():
    """Teeth: without the relative term the call above raises."""
    slot, events, fused = _wide_slot(37.0, 256)
    decomposed = torch.nn.functional.linear(events, slot.weight, slot.bias)
    residual = float((fused * 37.0 - decomposed).abs().max())
    assert residual > _DECOMPOSITION_ATOL * 37.0


def test_a_permuted_slot_order_still_fails_loud_on_a_wide_hop():
    """The check's whole job survives the relative term: a wrong order moves
    the sum by O(magnitude), far above any float32 residual."""
    slot, events, fused = _wide_slot(37.0, 256)
    slot.feed(events[:, torch.randperm(256)])
    with pytest.raises(SerialDecompositionMismatchError):
        slot.run_cycle(fused, 1.0)


# ── the cycle-atomic NF walks refuse under the point ─────────────────────────

@pytest.mark.parametrize("discipline", ["synchronized", "retime"])
def test_a_cycle_atomic_nf_walk_refuses_under_the_point(discipline):
    with pytest.raises(CycleAtomicRefusalError, match="per_event"):
        LifSegmentPolicy(**{discipline: True}, soma_law=PER_EVENT_LAW)


@pytest.mark.parametrize("discipline", ["synchronized", "retime"])
def test_the_same_walks_are_untouched_at_the_default_point(discipline):
    policy = LifSegmentPolicy(**{discipline: True}, soma_law=DEFAULT_SOMA_LAW)
    assert getattr(policy, discipline) is True


# ── the point denies the membrane-readout decode by DERIVATION ───────────────

def test_the_membrane_readout_derives_off_under_a_saturating_point():
    """Not a late crash inside the executor that refuses it: the recipe fold
    turns the knob off, exactly like an unsupported backend enable."""
    from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe

    dp = dict(_PER_EVENT_CONFIG, spiking_mode="lif", firing_mode="Novena")
    fold_conversion_recipe(dp, "lif", explicit_keys=set(_PER_EVENT_CONFIG))
    assert dp["lif_membrane_readout"] is False


def test_an_explicit_membrane_readout_contradicts_the_point_loudly():
    from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe

    dp = dict(
        _PER_EVENT_CONFIG, spiking_mode="lif", firing_mode="Novena",
        lif_membrane_readout=True,
    )
    with pytest.raises(ValueError, match="lif_membrane_readout=true"):
        fold_conversion_recipe(
            dp, "lif",
            explicit_keys=set(_PER_EVENT_CONFIG) | {"lif_membrane_readout"},
        )


def test_the_default_point_keeps_the_membrane_readout_recipe_default():
    from mimarsinan.config_schema.recipe_fold import fold_conversion_recipe

    dp = {"spiking_mode": "lif", "simulation_steps": T}
    fold_conversion_recipe(dp, "lif", explicit_keys={"simulation_steps"})
    assert dp["lif_membrane_readout"] is True
