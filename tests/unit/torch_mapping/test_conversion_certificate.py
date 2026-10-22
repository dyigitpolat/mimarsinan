"""A converted flow carries its probe verdict from birth, and a cached flow without a current one never resumes.

``convert_torch_model`` stamps a ``ConversionCertificate`` (converter version, every
required stage's measured verdict); a permissive conversion stamps an uncertified one
and warns. The pipeline cache refuses, by name, a converted flow that is unstamped
(written before the probe existed, or converted outside ``convert_torch_model``), stale
(another converter version) or uncertified, so a resumed run cannot skip the probe.
"""

from __future__ import annotations

import dataclasses
import pickle
import warnings

import pytest
import torch
import torch.nn as nn

from mimarsinan.mapping.platform.packaging_contract import MVM_PACKAGING, SPIKING_PACKAGING
from mimarsinan.pipelining.cache.load_store_strategies import (
    TorchModelLoadStoreStrategy,
    UncertifiedCachedFlowError,
)
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.torch_mapping import converter
from mimarsinan.torch_mapping.converted_model_flow import ConvertedModelFlow
from mimarsinan.torch_mapping.converter import convert_torch_model, probe_stages, trace_and_convert
from mimarsinan.torch_mapping.value_parity import STAGE_CONVERTED, STAGE_VALUE_PROGRAM
from mimarsinan.torch_mapping.value_parity.certificate import (
    CERTIFICATE_HYPOTHESES,
    CONVERTER_VERSION,
    ConversionCertificate,
    UnverifiedConversionWarning,
    certificate_problem,
)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(12, 16)
        self.r = nn.ReLU()
        self.fc = nn.Linear(16, 6)

    def forward(self, x):
        return self.fc(self.r(self.l1(x)))


def _convert(packaging=MVM_PACKAGING, **kwargs):
    torch.manual_seed(0)
    return convert_torch_model(_MLP().eval(), (12,), 6, packaging=packaging, **kwargs)


def _negate_fc(flow) -> None:
    for node in flow.get_mapper_repr().execution_order():
        perceptron = getattr(node, "perceptron", None)
        if getattr(perceptron, "name", None) == "fc":
            with torch.no_grad():
                perceptron.layer.weight.neg_()


def _wrong_conversion(monkeypatch) -> None:
    original = converter.trace_and_convert

    def wrong(*args, **kwargs):
        traced = original(*args, **kwargs)
        _negate_fc(traced.flow)
        return traced

    monkeypatch.setattr(converter, "trace_and_convert", wrong)


class TestStamp:
    @pytest.mark.parametrize("packaging,stages", [
        (MVM_PACKAGING, (STAGE_CONVERTED, STAGE_VALUE_PROGRAM)), (SPIKING_PACKAGING, (STAGE_CONVERTED,)),
    ], ids=["mvm", "spiking"])
    def test_a_strict_conversion_is_stamped_certified(self, packaging, stages):
        flow = _convert(packaging, value_program_stage=True)
        certificate = flow.conversion_certificate
        assert isinstance(certificate, ConversionCertificate) and certificate.certified
        assert certificate.converter_version == CONVERTER_VERSION
        assert certificate.required_stages == stages == probe_stages(packaging)
        assert tuple(v.stage for v in certificate.stages) == stages
        assert all(v.ok and v.relative_error is not None and v.relative_error <= v.bound for v in certificate.stages)
        assert certificate.weight_layer_depth == 2
        assert flow.conversion_certificate_problem() is None

    def test_a_value_domain_conversion_without_stage_2_is_certified_but_incomplete(self):
        certificate = _convert(MVM_PACKAGING).conversion_certificate
        assert certificate.certified and certificate.missing_stages == (STAGE_VALUE_PROGRAM,)
        assert "skipped required probe stages (identity value program)" in certificate_problem(certificate)

    def test_the_pipeline_conversion_runs_every_required_stage(self, monkeypatch):
        from mimarsinan.pipelining.pipeline_steps.config import torch_mapping_step

        config = {"input_shape": (12,), "num_classes": 6, "device": "cpu", "target_tq": 8}
        calls = []
        monkeypatch.setattr(torch_mapping_step, "convert_torch_model", lambda *a, **k: calls.append(k))
        for packaging in (MVM_PACKAGING, SPIKING_PACKAGING):
            torch_mapping_step.convert_pipeline_model(_MLP().eval(), config, packaging)
        assert [k["value_program_stage"] for k in calls] == [True, False]
        monkeypatch.undo()
        flow = torch_mapping_step.convert_pipeline_model(_MLP().eval(), config, MVM_PACKAGING)
        assert flow.conversion_certificate_problem() is None

    def test_the_stamp_is_deterministic_and_survives_pickling(self):
        a = _convert(value_program_stage=True).conversion_certificate
        b = _convert(value_program_stage=True).conversion_certificate
        assert a == b
        assert pickle.loads(pickle.dumps(a)) == a

    def test_a_permissive_wrong_conversion_is_stamped_uncertified_and_warns(self, monkeypatch):
        _wrong_conversion(monkeypatch)
        with pytest.warns(UnverifiedConversionWarning, match="NOT certified"):
            flow = _convert(strict=False)
        assert not flow.conversion_certificate.certified
        assert "NOT certified" in flow.conversion_certificate_problem()

    def test_an_unprobed_conversion_carries_no_stamp(self):
        flow = trace_and_convert(_MLP().eval(), (12,), "cpu", packaging=MVM_PACKAGING).flow
        assert flow.conversion_certificate is None
        assert "no conversion certificate" in flow.conversion_certificate_problem()

    def test_a_stale_converter_version_is_a_problem(self):
        stale = ConversionCertificate("pre-wave0", "native_eval_identity", 2, (STAGE_CONVERTED,), ())
        assert "stale" in certificate_problem(stale)

    def test_the_certificate_states_trace_fidelity_as_a_hypothesis(self):
        """A stated boundary, not a tolerance (review round 5; type checks named in round 6): FX evaluates such Python
        operations on a Proxy while tracing, before the converter runs, and folds them (the known-wrong pins)."""
        certificate = _convert(value_program_stage=True).conversion_certificate
        trace_fidelity = (
            "trace fidelity: the native model's output must not depend on Python-level identity, hashing, type checks "
            "(isinstance, type, callable, hasattr) or string forms of tensor shapes"
        )
        assert trace_fidelity in CERTIFICATE_HYPOTHESES and certificate.hypotheses == CERTIFICATE_HYPOTHESES
        assert pickle.loads(pickle.dumps(certificate)).hypotheses == CERTIFICATE_HYPOTHESES

    def test_the_certificate_states_its_trust_base_beside_trace_fidelity(self):
        """The owner's decision after review round 10 (``owner_decision_trust_base.md``): the certificate assumes T1
        (Python and torch as shipped) and T2 (nn.Module's call protocol) beside trace fidelity; a program that violates
        either is outside the guarantee, and the forms the toolchain detects are boundary witnesses."""
        assert CERTIFICATE_HYPOTHESES == (
            "trace fidelity: the native model's output must not depend on Python-level identity, hashing, type checks "
            "(isinstance, type, callable, hasattr) or string forms of tensor shapes",
            "T1 shipped runtime: Python and torch run as shipped (nothing of torch is rebound or rewritten after "
            "mimarsinan is imported, and no user mode or subclass poses as torch's own)",
            "T2 nn.Module call protocol: the model is called through nn.Module's own call protocol (no class in its "
            "hierarchy redefines attribute lookup or nn.Module's private call machinery, and no user container "
            "replaces nn.Module's private tables or torch's global hook tables)",
            "T3 opaque-helper effects: an opaque user callable (a torch.fx.wrap function or a leaf module the trace "
            "does not enter) has the effects on its tensor arguments, writes and storage aliasing, that the probe "
            "observes, made through torch's version-counted operations",
        )  # [extended in review round 16: the owner's T3 beside T1 and T2]
        certificate = _convert(value_program_stage=True).conversion_certificate
        assert certificate.hypotheses == CERTIFICATE_HYPOTHESES
        assert pickle.loads(pickle.dumps(certificate)).hypotheses == CERTIFICATE_HYPOTHESES

    def test_the_certificate_states_the_opaque_helper_hypothesis(self):
        """The owner's decision after review round 15 (``owner_decision_trust_base.md``, last section): T3, an opaque
        user callable's effects on its tensor arguments are those the probe observes, made through torch's
        version-counted operations; ``statement()`` prints it, ``summary()`` names it."""
        certificate = _convert(value_program_stage=True).conversion_certificate
        t3 = [h for h in certificate.hypotheses if h.startswith("T3 opaque-helper effects:")]
        assert len(t3) == 1 and "version-counted" in t3[0] and "the probe observes" in t3[0], certificate.hypotheses
        assert f"- {t3[0]}" in certificate.statement().splitlines()
        assert "T2 nn.Module call protocol and T3 opaque-helper effects (converter " in certificate.summary()

    def test_the_certification_statement_prints_every_hypothesis(self):
        """``statement()``: the verdict, what the static shapes are proved by, and every hypothesis in full; the
        one-line ``summary()`` of a certified verdict names the four."""
        certificate = _convert(value_program_stage=True).conversion_certificate
        statement = certificate.statement()
        lines = statement.splitlines()
        assert lines[0] == certificate.summary()
        # [restated in review round 14: "static shape proof: no guard on the batch symbol"; the proof is the affine
        # batch domain's]
        assert lines[1] == ("shape proof: static shape proof: every batch-dependent size in the affine batch domain "
                            "(a*B + c, B >= 1)")
        assert lines[2:] == ["under the hypotheses:", *(f"- {h}" for h in CERTIFICATE_HYPOTHESES)]
        assert certificate.summary().startswith(  # [extended in review round 16: T3]
            "conversion certified under trace fidelity, T1 shipped runtime, T2 nn.Module call protocol and T3 "
            "opaque-helper effects (converter ")

    def test_an_uncertified_statement_claims_no_hypothesis(self, monkeypatch):
        _wrong_conversion(monkeypatch)
        with pytest.warns(UnverifiedConversionWarning, match="NOT certified"):
            certificate = _convert(strict=False).conversion_certificate
        assert certificate.summary().startswith("conversion NOT certified (converter ")
        assert certificate.statement().splitlines()[2:] == ["no hypothesis is claimed: the conversion is not certified"]


class TestCacheLoad:
    def _store(self, tmp_path, flow, name="model"):
        TorchModelLoadStoreStrategy(name).store(str(tmp_path), flow)
        return TorchModelLoadStoreStrategy(name)

    def test_a_certified_flow_round_trips(self, tmp_path):
        flow = _convert(value_program_stage=True)
        loaded = self._store(tmp_path, flow).load(str(tmp_path))
        assert isinstance(loaded, ConvertedModelFlow)
        assert loaded.conversion_certificate == flow.conversion_certificate

    def test_an_unstamped_converted_flow_is_refused_by_name(self, tmp_path):
        flow = trace_and_convert(_MLP().eval(), (12,), "cpu", packaging=MVM_PACKAGING).flow
        strategy = self._store(tmp_path, flow, name="torch_mapping_model")
        with pytest.raises(UncertifiedCachedFlowError, match="'torch_mapping_model'.*no conversion certificate"):
            strategy.load(str(tmp_path))

    def test_an_incomplete_certificate_is_refused(self, tmp_path):
        with pytest.raises(UncertifiedCachedFlowError, match="skipped required probe stages"):
            self._store(tmp_path, _convert()).load(str(tmp_path))

    def test_a_pre_probe_artifact_is_refused(self, tmp_path):
        """A pickle from before the stamp existed: the attribute is absent and the class default reads None."""
        flow = _convert()
        del flow.conversion_certificate
        assert flow.conversion_certificate is None
        with pytest.raises(UncertifiedCachedFlowError):
            self._store(tmp_path, flow).load(str(tmp_path))

    def test_a_stale_stamp_is_refused(self, tmp_path):
        flow = _convert(value_program_stage=True)
        flow.conversion_certificate = dataclasses.replace(flow.conversion_certificate, converter_version="old")
        with pytest.raises(UncertifiedCachedFlowError, match="stale"):
            self._store(tmp_path, flow).load(str(tmp_path))

    def test_the_pipeline_cache_propagates_the_refusal(self, tmp_path):
        cache = PipelineCache()
        cache.add("TorchMappingStep.model", trace_and_convert(
            _MLP().eval(), (12,), "cpu", packaging=SPIKING_PACKAGING).flow, "torch_model")
        cache.store(str(tmp_path))
        with pytest.raises(UncertifiedCachedFlowError, match="quarantine"):
            PipelineCache().load(str(tmp_path))

    def test_non_flow_models_are_not_checked(self, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = self._store(tmp_path, _MLP()).load(str(tmp_path))
        assert isinstance(loaded, _MLP)
