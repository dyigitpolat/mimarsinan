"""[U-ADC A-1] Readout-aware training: ``readout_aware_training`` puts the deployed readout law in the forward of
Weight Quantization's endpoint recovery.

The law in the loop is the deployed law: with the key on, the recovery's training forward of a fixture core equals the
deployed program's readout bit for bit, and so do a five-core network's and a residual convolution net's, with their
ReLUs, add and pool as host ops or as executed lanes, and so does every read of the tuner; the law is solved again
whenever the live weights move. Its gradient rule is the stamped readout quantizer's straight-through estimator. A
short recovery under a coarse law (adc_bits 4) moves the deployed count in the direction the law-aware loss gives,
against the key off with the same budget, on the fixture's validation inputs. The program Soft Core Mapping deploys
afterwards carries the laws the loop last trained through (no second law). With the key absent or false, nothing in
Weight Quantization changes. Where the key cannot apply, it is refused by name before any budget is spent: a row with
no readout law, a spiking row, a driver that runs no endpoint recovery, an operand core type, structured pre-mapping
pruning and an architecture search (the scope: the per-core and lane laws of a fixed value-domain program)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import mimarsinan.pipelining.pipeline_steps.quantization.readout_aware_training as rat
import mimarsinan.tuning.tuners.normalization_aware_perceptron_quantization_tuner as napq
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry import REGISTRY
from mimarsinan.mapping.lanes.install import lanes_report
from mimarsinan.models.nn.activations.readout_quantizer import ReadoutQuantizer, straight_through
from mimarsinan.pipelining.core.pipelines.deployment_specs import get_pipeline_step_specs
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import SoftCoreMappingStep
from mimarsinan.pipelining.pipeline_steps.quantization.normalization_fusion_step import NormalizationFusionStep
from mimarsinan.pipelining.pipeline_steps.quantization.quantization_verification_step import (
    QuantizationVerificationStep,
)
from mimarsinan.pipelining.pipeline_steps.quantization.readout_aware_training import (
    READOUT_AWARE_TRAINING,
    ReadoutAwareParityError,
    ReadoutAwareTrainingError,
    ReadoutLawInLoop,
    readout_aware_endpoint,
)
from mimarsinan.pipelining.pipeline_steps.quantization.weight_quantization_step import WeightQuantizationStep
from unit.pipelining import readout_aware_programs as F

KEY = READOUT_AWARE_TRAINING
#: The direction fixture's endpoint budget (training steps of the recovery, key on and off alike).
DIRECTION_STEPS = 100


def _hooks(model: nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def _capture_training_forward(monkeypatch, x):
    """Replace the recovery by one training step's forward inside the open context: the trainer's projection, then the
    live model in training mode on ``x`` (``PerceptronTransformTrainer._backward_pass_on_loss``'s forward)."""
    seen = {}

    def recovery(tuner, *, base_steps, target_floor=None):
        tuner.trainer._update_and_transform_model()
        tuner.model.train()
        seen["out"] = tuner.model(x).detach().clone()
        seen["hooks_open"] = _hooks(tuner.model)
        tuner.model.eval()
        with torch.no_grad():
            seen["read"] = tuner.model(x).detach().clone()   # the forward every read of the tuner takes

    monkeypatch.setattr(napq, "run_endpoint_recovery", recovery)
    return seen


class TestTheLawInTheLoopIsTheDeployedLaw:
    @pytest.mark.parametrize("net,lanes", (("core", None), ("mlp", None), ("mlp", ("L",)), ("resnet", None),
                                           ("resnet", ("L", "E", "P"))))
    def test_the_training_forward_equals_the_deployed_readout_bit_for_bit(self, tmp_path, monkeypatch, net, lanes):
        """``core``: one fixture core (Linear 64 -> 4 with batch norm); ``mlp``: five cores, ReLUs between; ``resnet``:
        convolution cores, a residual add and an average pool. ReLUs, adds and pools are host ops, or with ``lanes``
        executed lanes (the clamp, fused-add and pool lane laws). The weights do not move (the replaced recovery takes
        no step), so Soft Core Mapping deploys the weights the captured forward read."""
        fx = F.prepared(tmp_path, adc_bits=4, net=net, lanes=lanes, **{KEY: True})
        seen = _capture_training_forward(monkeypatch, fx.x)
        F.weight_quantization(fx)
        assert seen["hooks_open"] > 0
        ir = F.soft_core_mapping(fx)
        assert (lanes_report(ir) is not None) == bool(lanes)   # the lanes run where declared
        want = F.deployed(ir, fx.x)
        assert torch.equal(seen["out"], want.to(seen["out"].dtype))
        assert torch.equal(seen["read"], want.to(seen["read"].dtype))
        assert F.count(want, fx.y) > 0

    def test_the_gradient_rule_is_the_stamped_quantizer_s_straight_through_estimator(self, tmp_path, monkeypatch):
        fx = F.prepared(tmp_path, adc_bits=4, net="core", **{KEY: True})
        seen = {}

        def recovery(tuner, *, base_steps, target_floor=None):
            tuner.model.eval()
            with torch.no_grad():
                tuner.model(fx.x)   # the law solved for these weights before the taps go on
            perceptron = tuner.model.get_perceptrons()[-1]
            taps = {}
            h1 = perceptron.scaler.register_forward_pre_hook(lambda m, a: taps.__setitem__("y", a[0]))
            h2 = perceptron.activation.register_forward_pre_hook(lambda m, a: taps.__setitem__("z", a[0]))
            try:
                tuner.model.train()
                tuner.model(fx.x)
            finally:
                h1.remove()
                h2.remove()
                tuner.model.eval()
            y, z = taps["y"], taps["z"]
            y.retain_grad()
            z.retain_grad()
            (z * torch.arange(z.numel(), dtype=z.dtype).reshape(z.shape)).sum().backward()
            seen.update(y=y.detach(), z=z.detach(), dy=y.grad.detach(), dz=z.grad.detach(),
                        loop=rat.current_loop(tuner))

        monkeypatch.setattr(napq, "run_endpoint_recovery", recovery)
        F.weight_quantization(fx)
        assert isinstance(seen["loop"], ReadoutLawInLoop)
        assert torch.equal(seen["dy"], seen["dz"])        # backward: the identity onto the float output
        assert not torch.equal(seen["z"], seen["y"])      # forward: the law's values, not the float output
        deployed = F.deployed(F.soft_core_mapping(fx), fx.x)   # the one core's readout is the network's output
        assert torch.equal(seen["z"], deployed.to(seen["z"].dtype))
        values = seen["z"].double()
        y = seen["y"].requires_grad_(True)
        out = straight_through(values, y)
        assert torch.equal(out.detach(), values.to(y.dtype))

    def test_the_law_is_solved_again_when_the_live_weights_move(self, tmp_path, monkeypatch):
        fx = F.prepared(tmp_path, adc_bits=4, net="mlp", **{KEY: True})
        seen = {}

        def recovery(tuner, *, base_steps, target_floor=None):
            context = rat.current_loop(tuner)
            tuner.model.eval()
            with torch.no_grad():
                before = tuner.model(fx.x).detach().clone()
                first = context.solves
                tuner.model(fx.x)
                seen["unchanged"] = context.solves - first
                # One training step's way of moving the live weights: the float aux weight, then its projection.
                weight = tuner.trainer.aux_model.get_perceptrons()[0].layer.weight
                weight[0, 0] += 0.5 * float(weight.abs().max())
                tuner.trainer._update_and_transform_model()
                out = tuner.model(fx.x).detach().clone()
            seen.update(moved=context.solves - first, out=out, before=before, ir=context.last_ir)

        monkeypatch.setattr(napq, "run_endpoint_recovery", recovery)
        F.weight_quantization(fx)
        assert seen["unchanged"] == 0 and seen["moved"] == 1
        assert torch.equal(seen["out"], F.deployed(seen["ir"], fx.x).to(seen["out"].dtype))
        assert not torch.equal(seen["out"], seen["before"])

    def test_a_forward_the_law_carrying_copy_does_not_share_is_refused_by_name(self, tmp_path, monkeypatch):
        fx = F.prepared(tmp_path, adc_bits=4, net="mlp", **{KEY: True})
        derive = SoftCoreMappingStep.derive_law_ir

        def faulty(step, model, platform):
            ir = derive(step, model, platform)
            model.get_perceptrons()[0].activation = nn.Tanh()  # a forward the live model does not have
            return ir

        def recovery(tuner, *, base_steps, target_floor=None):
            tuner.model.eval()
            with torch.no_grad():
                tuner.model(fx.x)

        monkeypatch.setattr(SoftCoreMappingStep, "derive_law_ir", faulty)
        monkeypatch.setattr(napq, "run_endpoint_recovery", recovery)
        with pytest.raises(ReadoutAwareParityError, match="law-carrying copy"):
            F.weight_quantization(fx)
        assert issubclass(ReadoutAwareParityError, ToolchainRefusal)


class TestTheRecoveryTrainsThroughTheLaw:
    def _run(self, tmp_path, key: bool):
        extra = {KEY: True} if key else {}
        fx = F.prepared(tmp_path, adc_bits=4, net="mlp", wq_endpoint_recovery_steps=DIRECTION_STEPS, **extra)
        reports, loops = [], []
        recover = napq.run_endpoint_recovery

        def spy(tuner, *, base_steps, target_floor=None):
            loops.append(rat.current_loop(tuner))
            report = recover(tuner, base_steps=base_steps, target_floor=target_floor)
            reports.append(report)
            return report

        napq.run_endpoint_recovery = spy
        try:
            F.weight_quantization(fx)
        finally:
            napq.run_endpoint_recovery = recover
        ir = F.soft_core_mapping(fx)
        out = F.deployed(ir, fx.x)
        return fx, reports[0], loops[0], ir, out

    def test_a_short_recovery_under_a_coarse_law_moves_the_deployed_count_the_way_the_law_aware_loss_gives(
            self, tmp_path):
        _fx_off, off, loop_off, _ir_off, out_off = self._run(tmp_path / "off", key=False)
        fx_on, on, loop_on, _ir_on, out_on = self._run(tmp_path / "on", key=True)
        assert loop_off is None and isinstance(loop_on, ReadoutLawInLoop)
        assert on.engaged and on.steps_used > 0 and on.exit > on.entry      # the law-aware read rose
        assert F.deployed_loss(out_on, fx_on.y) < F.deployed_loss(out_off, fx_on.y)
        assert F.count(out_on, fx_on.y) > F.count(out_off, fx_on.y)
        # The tuner's law-aware reads (every validation input) are the deployed program's counts: at entry the
        # untrained weights', at exit the trained ones'. Without the key the law-free read sits at its target and the
        # recovery does not engage (the converter-blind arm deploys the untrained weights).
        assert on.entry == F.count(out_off, fx_on.y) / F.SAMPLES and on.exit == F.count(out_on, fx_on.y) / F.SAMPLES
        assert not off.engaged

    def test_the_deployed_program_carries_the_laws_the_loop_last_trained_through(self, tmp_path):
        _fx, _report, loop, ir, _out = self._run(tmp_path, key=True)
        assert loop.last_ir is not None and loop.solves > 1
        assert F.law_digests(loop.last_ir) == F.law_digests(ir)

    def test_the_context_leaves_the_model_as_it_found_it(self, tmp_path):
        """No hook stays on the model, and its modules and state keys are those of the run without the key."""
        fx_on, _report, _loop, _ir, _out = self._run(tmp_path / "on", key=True)
        fx_off, _report, _loop, _ir, _out = self._run(tmp_path / "off", key=False)
        on, off = F.model_of(fx_on), F.model_of(fx_off)
        assert _hooks(on) == 0 == _hooks(off)
        assert [type(m) for m in on.modules()] == [type(m) for m in off.modules()]
        assert list(on.state_dict()) == list(off.state_dict())


class TestAbsentKey:
    @pytest.mark.parametrize("value", (None, False))
    def test_an_absent_or_false_key_builds_no_endpoint_context(self, value):
        config = F.pipeline_config(4, **({} if value is None else {KEY: value}))
        assert readout_aware_endpoint(SimpleNamespace(config=config)) is None

    def test_weight_quantization_hands_its_tuner_no_endpoint_context(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(WeightQuantizationStep, "run_tuner", lambda self, cls, *a, **kw: captured.update(kw))
        fx = F.prepared(tmp_path, adc_bits=4, net="mlp")
        F.weight_quantization_process(fx)
        assert captured["endpoint_forward"] is None

    def test_the_tuner_without_a_context_calls_the_recovery_directly(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(napq, "run_endpoint_recovery",
                            lambda tuner, *, base_steps, target_floor=None: calls.append(rat.current_loop(tuner)))
        fx = F.prepared(tmp_path, adc_bits=4, net="mlp")
        F.weight_quantization(fx)
        assert calls == [None]

    def test_a_false_key_runs_weight_quantization_byte_for_byte_as_an_absent_one(self, tmp_path):
        runs = {}
        for name, extra in (("absent", {}), ("false", {KEY: False})):
            fx = F.prepared(tmp_path / name, adc_bits=4, net="mlp", wq_endpoint_recovery_steps=DIRECTION_STEPS,
                            **extra)
            F.weight_quantization(fx)
            runs[name] = (F.model_of(fx).state_dict(), F.law_digests(F.soft_core_mapping(fx)))
        (state_a, laws_a), (state_b, laws_b) = runs["absent"], runs["false"]
        assert state_a.keys() == state_b.keys() and all(torch.equal(state_a[k], state_b[k]) for k in state_a)
        assert laws_a == laws_b


class TestScope:
    """Refused by name where the key cannot apply, at Weight Quantization's entry, before any budget is spent."""

    def _refusal(self, config):
        with pytest.raises(ReadoutAwareTrainingError) as info:
            readout_aware_endpoint(SimpleNamespace(config=config))
        assert issubclass(ReadoutAwareTrainingError, ToolchainRefusal)
        return str(info.value)

    def test_a_row_with_no_readout_law(self):
        config = F.pipeline_config(4, **{KEY: True})
        config.pop("readout_law")
        assert "readout_law" in self._refusal(config)

    def test_a_spiking_row(self):
        config = F.pipeline_config(4, **{KEY: True})
        config["core_semantics"] = "spiking"
        assert "spiking" in self._refusal(config)

    def test_an_operand_core_type(self):
        config = F.pipeline_config(4, **{KEY: True})
        config["cores"] = [dict(config["cores"][0]), dict(config["cores"][0], operand={"payload_bits": 4})]
        assert "operand" in self._refusal(config)

    def test_structured_pre_mapping_pruning(self):
        config = F.pipeline_config(4, **{KEY: True})
        config["prune_sparsity"] = 0.25
        assert "prune_sparsity" in self._refusal(config)

    def test_a_driver_that_runs_no_endpoint_recovery(self):
        config = F.pipeline_config(4, **{KEY: True})
        config["optimization_driver"] = "controller"
        assert "optimization driver" in self._refusal(config)

    def test_an_architecture_search(self):
        config = F.pipeline_config(4, **{KEY: True})
        config["hw_config_mode"] = "search"
        assert "search" in self._refusal(config)

    def test_weight_quantization_refuses_before_its_tuner_runs(self, tmp_path, monkeypatch):
        fx = F.prepared(tmp_path, adc_bits=4, net="mlp", **{KEY: True})
        fx.pipeline.config["prune_sparsity"] = 0.25
        monkeypatch.setattr(WeightQuantizationStep, "run_tuner",
                            lambda *a, **k: pytest.fail("the tuner ran under an out-of-scope declaration"))
        with pytest.raises(ReadoutAwareTrainingError, match="prune_sparsity"):
            F.weight_quantization_process(fx)

    def test_the_replica_s_preparation_is_every_step_between_weight_quantization_and_soft_core_mapping(self):
        """The loop prepares its copy as the pipeline prepares the model it maps: Normalization Fusion's transform,
        then Soft Core Mapping's own preparation and IR derivation. Between the two steps the plan holds Quantization
        Verification (which reads, never writes, the model) and Normalization Fusion only."""
        specs = [cls for _name, cls in get_pipeline_step_specs(F.pipeline_config(4, **{KEY: True}))]
        between = specs[specs.index(WeightQuantizationStep) + 1:specs.index(SoftCoreMappingStep)]
        assert between == [QuantizationVerificationStep, NormalizationFusionStep]


class TestEstimator:
    def test_the_stamped_quantizer_forward_is_the_named_estimator(self):
        """``straight_through``: the values forward, the identity onto ``y`` backward (the stamped quantizer's)."""
        y = torch.tensor([0.4, 2.5, -9.0], requires_grad=True)
        values = torch.tensor([0.5, 2.0, -7.0], dtype=torch.float64)
        out = straight_through(values, y)
        assert out.dtype == y.dtype and out.tolist() == [0.5, 2.0, -7.0]
        out.sum().backward()
        assert y.grad.tolist() == [1.0, 1.0, 1.0]
        assert hasattr(ReadoutQuantizer, "adopt_law") and hasattr(ReadoutQuantizer, "release_law")


def test_the_key_is_the_registry_s():
    assert READOUT_AWARE_TRAINING == "readout_aware_training" and READOUT_AWARE_TRAINING in REGISTRY
