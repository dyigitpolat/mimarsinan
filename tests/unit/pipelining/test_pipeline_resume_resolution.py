"""A resume starts only where the run it starts can feed every step up to the requested one, and says so once.

Phase C defect D-G6-1 (SD-15, the release's own resume of the stock ODIN card run): the release reported one start
(Normalization Fusion), executed another (Weight Quantization, resolved a second time), and stopped at an anonymous
``AssertionError`` because the backward walk cleared a missing entry by the step that writes it without asking for
that step's own requirements. The rule these tests pin:

* a start index ``i`` for a requested step ``X`` is valid only if the entries the steps before ``i`` required and did
  not consume are in the cache (the precondition a direct start has today), and every requirement of every step from
  ``i`` to ``X`` is in the cache or written (promised or updated) by an earlier step at or after ``i``;
* the resolver returns the latest valid index, and a start it returns resolves to itself, so the session executes
  exactly the start it reports;
* no valid start, and a step whose requirements are absent when it runs, are refusals by name (a ``ToolchainRefusal``
  naming the step, the missing entries, the steps that write them and the remedy), never a bare assertion.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from conftest import MockDataProviderFactory

from mimarsinan.common.refusal import is_toolchain_refusal
from mimarsinan.pipelining.cache.pipeline_cache import PipelineCache
from mimarsinan.pipelining.core.engine import pipeline_resume
from mimarsinan.pipelining.core.engine.pipeline import Pipeline
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.session import PipelineSession

REPO = Path(__file__).resolve().parents[3]


class _Step(PipelineStep):
    """Reads every requirement, writes every promise and update (updates are a subset of requires)."""

    def __init__(self, pipeline, *, requires=(), promises=(), updates=()):
        super().__init__(requires=requires, promises=promises, updates=updates, clears=[], pipeline=pipeline)

    def process(self):
        for requirement in self.requires:
            self.get_entry(requirement)
        for promise in self.promises:
            self.add_entry(promise, self.name)
        for entry in self.updates:
            self.update_entry(entry, self.name)

    def validate(self):
        return 1.0


def _pipeline(tmp_path, spec, cached=()):
    """``spec``: (name, requires, promises, updates) per step; ``cached``: real keys present in the cache."""
    p = Pipeline(str(tmp_path / "cache"))
    for name, requires, promises, updates in spec:
        p.add_pipeline_step(name, _Step(p, requires=requires, promises=promises, updates=updates))
    for key in cached:
        p.cache.add(key, key)
    p.set_up_requirements()
    return p


# The plan's fixture: A promises m; B and C require and update m; D requires m and promises x; E requires x.
CHAIN = [
    ("A", [], ["m"], []),
    ("B", ["m"], [], ["m"]),
    ("C", ["m"], [], ["m"]),
    ("D", ["m"], ["x"], []),
    ("E", ["x"], [], []),
]


def _names(pipeline):
    return [name for name, _ in pipeline.steps]


class TestResolution:
    def test_the_start_is_closed_under_the_requirements_of_every_step_up_to_the_request(self, tmp_path):
        # B.m and C.m absent, A.m present: C's own input B.m is missing, so the latest start that feeds C, D and E is B.
        p = _pipeline(tmp_path, CHAIN, cached=["A.m"])
        assert _names(p)[pipeline_resume.find_starting_step_idx(p, "E")] == "B"
        assert p.get_resolved_start_step("E") == "B"

    def test_run_from_executes_the_closed_start_and_every_step_finds_its_inputs(self, tmp_path):
        p = _pipeline(tmp_path, CHAIN, cached=["A.m"])
        executed = []
        p.register_pre_step_hook(lambda name, step: executed.append(name))
        p.run_from("E")
        assert executed == ["B", "C", "D", "E"]

    def test_a_cached_requested_step_is_honoured_as_today(self, tmp_path):
        p = _pipeline(tmp_path, CHAIN, cached=["A.m", "B.m", "C.m", "D.x"])
        assert p.get_resolved_start_step("E") == "E"
        assert p.get_resolved_start_step("C") == "C"

    def test_no_valid_start_is_a_refusal_naming_the_request_the_entries_and_the_remedy(self, tmp_path):
        # B requires k and promises k again, so B's requirement is B.k: no step before B writes it, from any start.
        spec = [("A", [], ["k"], []), ("B", ["k"], ["k"], []), ("C", ["k"], [], [])]
        p = _pipeline(tmp_path, spec, cached=["A.k"])
        with pytest.raises(Exception) as info:
            pipeline_resume.find_starting_step_idx(p, "C")
        err = info.value
        assert is_toolchain_refusal(err), f"an anonymous {type(err).__name__} is not a refusal by name: {err}"
        assert isinstance(err, AssertionError), "the refusal keeps the builtin base of the assertion it replaces"
        assert err.requested_step == "C"
        assert err.step_name == "B"
        assert err.missing_keys == ("B.k",)
        assert err.writers == {"B.k": ("B",)}
        text = str(err)
        assert "'C'" in text and "B.k" in text and "quarantine" in text and "rerun" in text

    def test_an_unknown_requested_step_still_fails_loudly(self, tmp_path):
        p = _pipeline(tmp_path, CHAIN, cached=["A.m"])
        with pytest.raises(AssertionError):
            p.run_from("nonexistent")


class TestPresenceCheck:
    def test_a_step_whose_requirements_are_absent_refuses_by_name(self, tmp_path):
        p = _pipeline(tmp_path, [("produce", [], ["data"], []), ("consume", ["data"], ["result"], [])])
        with pytest.raises(Exception) as info:
            p._run_step("consume", p.steps[1][1])
        err = info.value
        assert is_toolchain_refusal(err), f"an anonymous {type(err).__name__} is not a refusal by name: {err}"
        assert isinstance(err, AssertionError)
        from mimarsinan.pipelining.core.engine.pipeline_resume import MissingRequirementsError

        assert isinstance(err, MissingRequirementsError)
        assert err.step_name == "consume"
        assert err.missing_keys == ("produce.data",)
        assert err.writers == {"produce.data": ("produce",)}
        assert err.requested_step is None
        text = str(err)
        assert "'consume'" in text and "produce.data" in text and "'produce'" in text


# ---------------------------------------------------------------------------
# Every graph: soundness, latest start, fixed point, today's direct starts
# ---------------------------------------------------------------------------

def _random_spec(rng):
    keys = ["a", "b", "c", "d"]
    spec, promised = [], set()
    for idx in range(rng.randint(3, 7)):
        requires = sorted(k for k in promised if rng.random() < 0.5)
        updates = sorted(k for k in requires if rng.random() < 0.5)
        promises = sorted(k for k in keys if rng.random() < (0.6 if idx == 0 else 0.25))
        spec.append((f"S{idx}", requires, promises, updates))
        promised |= set(promises)
    return spec


def _written_keys(pipeline):
    return sorted({pipeline._create_real_key(step.name, entry)
                   for _, step in pipeline.steps for entry in (*step.promises, *step.updates)})


def _feeds(pipeline, start, end):
    """The run from ``start`` reaches ``end`` with every requirement present (the engine's own reading)."""
    cache = set(pipeline.cache.keys())
    for _, step in pipeline.steps[start:end + 1]:
        if any(pipeline._translate_key(step.name, r) not in cache for r in step.requires):
            return False
        for entry in step.updates:
            cache.discard(pipeline._translate_key(step.name, entry))
        cache |= {pipeline._create_real_key(step.name, e) for e in (*step.promises, *step.updates)}
    return True


def _carried_present(pipeline, idx):
    return pipeline_resume.get_all_requirements(pipeline, pipeline.steps[idx][0]) <= set(pipeline.cache.keys())


def _valid(pipeline, start, end):
    return _carried_present(pipeline, start) and _feeds(pipeline, start, end)


def test_every_graph_resolves_to_the_latest_valid_start_or_refuses_by_name(tmp_path):
    rng = random.Random(20261101)
    checked = refused = moved = 0
    for case in range(300):
        spec = _random_spec(rng)
        probe = _pipeline(tmp_path / f"probe{case}", spec)
        cached = [k for k in _written_keys(probe) if rng.random() < 0.55]
        p = _pipeline(tmp_path / f"case{case}", spec, cached=cached)
        for requested in range(len(spec)):
            name = spec[requested][0]
            try:
                start = pipeline_resume.find_starting_step_idx(p, name)
            except Exception as err:  # noqa: BLE001 - the test reads the outcome's class
                assert is_toolchain_refusal(err), f"case {case} {name}: anonymous {type(err).__name__}: {err}"
                assert not any(_valid(p, j, requested) for j in range(requested + 1)), (case, name, spec, cached)
                refused += 1
                continue
            checked += 1
            assert 0 <= start <= requested
            assert _valid(p, start, requested), f"case {case}: start {start} for {name} does not feed it: {spec} {cached}"
            assert not any(_valid(p, j, requested) for j in range(start + 1, requested + 1)), (case, name, start)
            assert pipeline_resume.find_starting_step_idx(p, spec[start][0]) == start, "a resolved start moves"
            if _carried_present(p, requested):
                assert start == requested, "a direct start is decided as today"
            moved += start != requested
    assert checked > 500 and moved > 50 and refused > 0, (checked, moved, refused)


# ---------------------------------------------------------------------------
# The stock release-resume route of phase C group 6 (SD-15), as a fixture: no card, no data, no step runs
# ---------------------------------------------------------------------------

CARD_CONFIG = REPO / "scripts" / "hacc" / "odin_deployment_cell.json"
EXPORT = "HACC NUS - ODIN Deployment"
# The cache keys of the lineage stock card run (metadata.json of
# odin_narrowconv_lifs_wq_s16_nobias_phased_deployment_run, the run the HACC card sessions measured).
LINEAGE_KEYS = (
    "__target_metric", "Model Configuration.model_builder", "Model Configuration.model_config",
    "Model Configuration.platform_constraints_resolved", "Model Building.adaptation_manager", "Model Building.model",
    "Pretraining.model", "__mbh_retention_envelope", "Torch Mapping.model", "Torch Mapping.adaptation_manager",
    "Scale Migration.model", "Scale Migration.scale_migration_report", "Activation Analysis.activation_scales",
    "Activation Analysis.activation_scale_stats", "Activation Analysis.install_resolution_gauge",
    "Activation Adaptation.adaptation_manager", "Activation Adaptation.model", "Activation Adaptation.adaptation_ledger",
    "__mbh_dhat_highwater", "Clamp Adaptation.adaptation_manager", "Clamp Adaptation.model",
    "Clamp Adaptation.adaptation_ledger", "Activation Shifting.adaptation_manager", "Activation Shifting.model",
    "Activation Shifting.adaptation_ledger", "Activation Quantization.adaptation_manager",
    "Activation Quantization.model", "Activation Quantization.aq_reference_read",
    "Activation Quantization.adaptation_ledger", "__mbh_endpoint_steps_consumed", "LIF Adaptation.adaptation_manager",
    "LIF Adaptation.model", "LIF Adaptation.adaptation_ledger", "Weight Quantization.adaptation_manager",
    "Weight Quantization.model", "Weight Quantization.adaptation_ledger", "Normalization Fusion.model",
    "Normalization Fusion.fused_model", "Soft Core Mapping.ir_graph", "Soft Core Mapping.deployment_record_scm",
    "hybrid_mapping", "Hard Core Mapping.hard_core_mapping", "Hard Core Mapping.deployment_record_hcm",
    "Simulation.deployment_record_nevresim", "HACC NUS - ODIN Deployment.odin_hacc_deployment_bundle",
    "Deployment Record.deployment_record",
)
# The converted flows the release head refuses as uncertified (Wave 0 exactness) and the group quarantined with the
# release's own remedy, in the order it met them.
QUARANTINED = (
    "Torch Mapping.model", "Scale Migration.model", "Activation Adaptation.model", "Clamp Adaptation.model",
    "Activation Shifting.model", "Activation Quantization.model", "LIF Adaptation.model", "Weight Quantization.model",
    "Normalization Fusion.model", "Normalization Fusion.fused_model",
)


def _stock_session(tmp_path, *, quarantine):
    wd = tmp_path / "odin_narrowconv_lifs_wq_s16_nobias_phased_deployment_run"
    config = {k: v for k, v in json.loads(CARD_CONFIG.read_text()).items() if not k.startswith("_")}
    # [U-LIF2 r5] the stock run holds its saved document, as every run directory with entries does (entries beside no
    # point and no saved document are refused by name)
    (wd / "_RUN_CONFIG").mkdir(parents=True)
    (wd / "_RUN_CONFIG" / "config.json").write_text(json.dumps(config))
    cache = PipelineCache()
    for key in LINEAGE_KEYS:
        cache.add(key, 0.0)
    cache.store(str(wd))
    for key in QUARANTINED if quarantine else ():
        PipelineCache.quarantine_entry(str(wd), key)
    config.update(start_step=EXPORT, stop_step=None, _working_directory=str(wd))
    config["deployment_parameters"]["enable_nevresim_simulation"] = False
    return PipelineSession.from_config(
        config, data_provider_factory=MockDataProviderFactory(input_shape=(1, 28, 28), num_classes=10))


def _record_steps(pipeline, executed):
    """Stand-in for a step's work: the engine's own requirement check, then the step's writes (no step runs)."""

    def run_step(name, step):
        executed.append(name)
        pipeline._assert_requirements_present(step)
        for promise in step.promises:
            pipeline.add_entry(step, promise, 0.0)
        for entry in step.updates:
            pipeline.update_entry(step, entry, 0.0)

    pipeline._run_step = run_step


class TestStockReleaseResume:
    def test_the_quarantined_stock_run_resumes_from_its_last_certified_input(self, tmp_path):
        session = _stock_session(tmp_path, quarantine=True)
        try:
            assert session.resolved_start_step() == "Torch Mapping"
        finally:
            session.finish()

    def test_the_session_executes_exactly_the_start_it_reports(self, tmp_path):
        session = _stock_session(tmp_path, quarantine=True)
        executed = []
        _record_steps(session.pipeline, executed)
        try:
            reported = session.resolved_start_step()
            session.run()
        finally:
            session.finish()
        names = _names(session.pipeline)
        assert executed[:1] == [reported]
        assert executed == names[names.index(reported):]

    def test_the_unquarantined_lineage_cache_resolves_as_today(self, tmp_path):
        # Only R's new Soft Core Mapping.ir_graph_unbounded is absent: R restarts at Soft Core Mapping, as before.
        session = _stock_session(tmp_path, quarantine=False)
        try:
            assert session.resolved_start_step() == "Soft Core Mapping"
        finally:
            session.finish()
