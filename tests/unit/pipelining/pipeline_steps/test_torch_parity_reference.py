"""The torch↔deployed-sim parity gate's corrected reference (T6 Part-A follow-up).

For floor-convention-trained models, SoftCoreMappingStep bakes the mapping-time
+0.5/Tq bias compensation BEFORE the parity gate runs, while the trained
activation stack still carries the training-time half-step ShiftDecorator — the
raw torch reference computes half a level high (double shift) and self-degrades
the measurement. ``torch_parity_reference`` neutralizes the trained shift on
comp-baked perceptrons of a DEEPCOPY (encoders keep the trained convention:
they deploy as host ops running the trained module); the threshold is never
weakened. Models without the bake (sync-exact-trained, cascaded, continuous
ttfs) pass through unchanged.
"""

from __future__ import annotations

import numpy as np
import torch

from conftest import MockPipeline, default_config, make_tiny_supermodel
from mimarsinan.mapping.support.bias_compensation import (
    TTFS_COMP_BAKED_FLAG,
    apply_ttfs_quantization_bias_compensation,
)
from mimarsinan.pipelining.core.nf_scm_parity import torch_parity_reference
from mimarsinan.tuning.orchestration.adaptation_manager import AdaptationManager

_TQ = 4
_DELTA = 1.0 / _TQ


def _floor_model(tq=_TQ):
    """A tiny model trained on the floor + half-step proxy (ttfs_quantized style)."""
    cfg = default_config()
    cfg["spiking_mode"] = "ttfs_quantized"
    cfg["activation_quantization"] = True
    cfg["target_tq"] = tq
    cfg["simulation_steps"] = tq
    model = make_tiny_supermodel()
    manager = AdaptationManager()
    manager.quantization_rate = 1.0
    for p in model.get_perceptrons():
        manager.update_activation(cfg, p)
    return model


def _bake(model, tq=_TQ):
    apply_ttfs_quantization_bias_compensation(model, tq)
    return model


# Non-negative interiors (the ReLU base is identity there), off bin edges.
_SWEEP = torch.tensor([0.05, 0.1, 0.3, 0.55, 0.6, 0.8, 0.95])


def _trained_floor_halfstep(z):
    return torch.floor((z + _DELTA / 2) / _DELTA) * _DELTA


def _pure_floor(z):
    return torch.floor(z / _DELTA) * _DELTA


class TestReferenceCorrection:
    def test_unbaked_model_passes_through_as_itself(self):
        model = _floor_model()
        assert torch_parity_reference(model) is model

    def test_baked_model_yields_a_corrected_deepcopy(self):
        model = _bake(_floor_model())
        ref = torch_parity_reference(model)
        assert ref is not model
        for p_ref, p_live in zip(ref.get_perceptrons(), model.get_perceptrons()):
            if getattr(p_live, "is_encoding_layer", False):
                continue
            assert getattr(p_live, TTFS_COMP_BAKED_FLAG, False)
            with torch.no_grad():
                y_ref = p_ref.activation(_SWEEP)
                y_live = p_live.activation(_SWEEP)
            # the corrected reference computes the pure floor staircase — with
            # the comp'd bias upstream that IS the trained (and deployed)
            # function; the live stack still carries the trained half-step.
            np.testing.assert_allclose(
                y_ref.numpy(), _pure_floor(_SWEEP).numpy(), atol=1e-6,
            )
            np.testing.assert_allclose(
                y_live.numpy(), _trained_floor_halfstep(_SWEEP).numpy(), atol=1e-6,
            )

    def test_double_shift_removed_end_to_end_per_neuron(self):
        # trained function on the raw preact u: floor((u + D/2)/D) * D.
        # comp'd bias shifts the preact by +D/2; the corrected reference then
        # computes floor((u + D/2)/D) * D == trained, while the uncorrected
        # stack reads floor((u + D)/D) * D — one half-level high.
        model = _bake(_floor_model())
        ref = torch_parity_reference(model)
        u = _SWEEP
        comp = _DELTA / 2
        p_ref = [p for p in ref.get_perceptrons()
                 if not getattr(p, "is_encoding_layer", False)][0]
        p_live = [p for p in model.get_perceptrons()
                  if not getattr(p, "is_encoding_layer", False)][0]
        with torch.no_grad():
            corrected = p_ref.activation(u + comp)
            uncorrected = p_live.activation(u + comp)
        np.testing.assert_allclose(
            corrected.numpy(), _trained_floor_halfstep(u).numpy(), atol=1e-6,
        )
        assert (uncorrected - corrected).max().item() >= _DELTA - 1e-6, (
            "the uncorrected reference must exhibit the half-level-high defect "
            "this fix removes"
        )

    def test_encoder_keeps_the_trained_convention(self):
        # The subsumed encoder deploys as a HOST op running the trained torch
        # module, so the exact reference keeps its trained convention.
        model = _bake(_floor_model())
        ref = torch_parity_reference(model)
        enc_ref = [p for p in ref.get_perceptrons()
                   if getattr(p, "is_encoding_layer", False)][0]
        enc_live = [p for p in model.get_perceptrons()
                    if getattr(p, "is_encoding_layer", False)][0]
        assert not getattr(enc_live, TTFS_COMP_BAKED_FLAG, False)
        with torch.no_grad():
            assert torch.equal(enc_ref.activation(_SWEEP), enc_live.activation(_SWEEP))

    def test_live_model_is_never_mutated(self):
        model = _bake(_floor_model())
        pre_sd = {k: v.clone() for k, v in model.state_dict().items()}
        torch_parity_reference(model)
        post_sd = model.state_dict()
        for key in pre_sd:
            assert torch.equal(pre_sd[key], post_sd[key]), key
        with torch.no_grad():
            y = model.get_perceptrons()[-1].activation(_SWEEP)
        np.testing.assert_allclose(
            y.numpy(), _trained_floor_halfstep(_SWEEP).numpy(), atol=1e-6,
        )

    def test_sync_exact_trained_model_passes_through(self):
        # sync-exact models skip the comp bake entirely (marker-asserted), so
        # the reference is the model itself — no shift to neutralize.
        cfg = default_config()
        cfg["spiking_mode"] = "ttfs_cycle_based"
        cfg["ttfs_cycle_schedule"] = "synchronized"
        cfg["activation_quantization"] = True
        cfg["sync_exact_qat"] = True
        cfg["target_tq"] = _TQ
        cfg["simulation_steps"] = _TQ
        model = make_tiny_supermodel()
        manager = AdaptationManager()
        manager.quantization_rate = 1.0
        for p in model.get_perceptrons():
            manager.update_activation(cfg, p)
        assert torch_parity_reference(model) is model


class TestStepWiring:
    """SoftCoreMappingStep passes the corrected reference to the parity assert."""

    class _StubTrainer:
        def iter_validation_batches(self, n):
            yield torch.rand(4, 8), None

    class _SyncContract:
        def is_synchronized(self):
            return True

        def is_cascaded(self):
            return False

        def uses_ttfs_floor_ceil_convention(self):
            return True

        def training_forward_kind(self):
            return "analytical_staircase"

    def test_gate_receives_the_corrected_reference(self, monkeypatch):
        import mimarsinan.pipelining.core.nf_scm_parity as parity_mod
        import mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step as step_mod
        from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
            SoftCoreMappingStep,
        )

        model = _bake(_floor_model())
        captured = {}

        monkeypatch.setattr(
            step_mod, "build_deployment_contract",
            lambda pipeline: self._SyncContract(),
        )
        monkeypatch.setattr(
            step_mod, "build_identity_mapping_for_pipeline",
            lambda ir_graph, pipeline_config: object(),
        )
        monkeypatch.setattr(
            step_mod, "build_spiking_hybrid_flow",
            lambda pipeline, mapping, model: object(),
        )

        def _capture(reference, flow, samples, *, min_agreement):
            captured["reference"] = reference
            captured["min_agreement"] = min_agreement
            return 1.0

        monkeypatch.setattr(
            parity_mod, "assert_torch_vs_deployed_sim_parity_or_raise", _capture,
        )

        pipeline = MockPipeline(config=default_config())
        step = SoftCoreMappingStep(pipeline)
        step.trainer = self._StubTrainer()
        step._run_torch_sim_parity_check(model, ir_graph=object())

        assert captured["reference"] is not model, (
            "a comp-baked model must be checked against the corrected reference"
        )
        # never weaken the threshold
        assert captured["min_agreement"] == 0.98
