"""Registry decision classes: one class per key and per nested field, a join, and a composite class."""

import importlib.util
import inspect
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import MockDataProviderFactory, MockPipeline, default_config
from mimarsinan.chip_simulation.sanafe.arch_synth.spec import CUSTOM_PRESET_NAME, _resolve_plugins
from mimarsinan.chip_simulation.sanafe.presets import PRESETS
from mimarsinan.chip_simulation.sanafe.runner import SanafeRunner
from mimarsinan.config_schema.registry import REGISTRY, serialize_registry
from mimarsinan.config_schema.registry.build import declared_paths, effective_class
from mimarsinan.config_schema.registry.decision_classes import (
    ADMISSIBILITY_BEARING,
    COMPOSITE,
    COST_ONLY,
    DECISION_CLASSES,
    FUNCTION_BEARING,
    GROUP_CLASSES,
    KEY_CLASSES,
    NOT_A_DECISION,
    DecisionClassError,
    join,
)
from mimarsinan.config_schema.registry.groups import VALID_GROUP_IDS
from mimarsinan.config_schema.registry.value_schema import nested_schemas
from mimarsinan.pipelining.core import simulation_factory
from mimarsinan.pipelining.core.steps.pipeline_step import PipelineStep
from mimarsinan.pipelining.pipeline_steps.verification.sanafe_simulation_step import (
    SanafeSimulationStep,
)
from mimarsinan.tuning.adaptation_target_adjuster import AdaptationTargetAdjuster

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SRC = Path(_REPO_ROOT) / "src" / "mimarsinan"
_EXPLICIT_GROUPS = ("hardware", "mapping_strategy")


def _load_script(name):
    path = os.path.join(_REPO_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested(entry):
    if entry.value_schema is None:
        return []
    return list(nested_schemas(entry.value_schema, entry.flat_key))


class TestCoverage:
    def test_every_key_carries_a_class_and_a_reason(self):
        for key, entry in REGISTRY.items():
            assert entry.decision_class in DECISION_CLASSES, key
            assert entry.decision_reason and entry.decision_reason.strip(), key

    def test_every_group_has_a_class(self):
        assert set(GROUP_CLASSES) == set(VALID_GROUP_IDS)
        for group, (cls, reason) in GROUP_CLASSES.items():
            assert cls in DECISION_CLASSES and cls != COMPOSITE, group
            assert reason.strip(), group

    def test_key_table_names_only_registry_keys(self):
        assert set(KEY_CLASSES) <= set(REGISTRY)

    def test_hardware_and_mapping_strategy_keys_are_classified_explicitly(self):
        keys = {k for k, e in REGISTRY.items() if e.group in _EXPLICIT_GROUPS}
        assert len(keys) >= 39
        assert keys <= set(KEY_CLASSES), sorted(keys - set(KEY_CLASSES))

    def test_explicit_rows_differ_from_their_group_outside_the_explicit_groups(self):
        for key, (cls, _reason) in KEY_CLASSES.items():
            group = REGISTRY[key].group
            if group not in _EXPLICIT_GROUPS:
                assert cls != GROUP_CLASSES[group][0], key

    def test_every_nested_field_carries_its_own_class(self):
        for key, entry in REGISTRY.items():
            for path, schema in _nested(entry):
                assert schema.decision_class in DECISION_CLASSES, path

    def test_a_nested_key_reports_the_join_of_its_fields(self):
        for key, entry in REGISTRY.items():
            nested = _nested(entry)
            if not nested or entry.decision_class == COMPOSITE:
                continue
            assert entry.decision_class == join(s.decision_class for _, s in nested), key


#: The cost-only set the owner reviewed at S0 (keys and nested fields): the theorem box covers exactly
#: this set, so it changes only through an S0 amendment. Review round 3 grounded every row in its
#: consumers; ``scheduling_latency_weight`` left it (no run reads it, only the wizard's suggester). The
#: polish round moved ``sanafe_arch_preset`` and ``sanafe_custom_arch_path`` out: both choose the
#: architecture the SANA-FE parity refusal simulates (plan section 20 item 4, TestClassEvidence). The
#: owner's S0 review moved ``decompose_leaf_modules`` out (strongest consumer: on spiking cores the moved
#: projection rides the activation grid), leaving the 26 rows the owner confirmed. The S0 amendment for
#: CL-SG (owner answer L1 = a, sign-off sheet section 5) adds ``cores.weight_sign_granularity``: the per-axon
#: sign split is exact per cycle (W = W+ + W- on the integer grid) and per event (the pair lemma), and a
#: float weight on a per-axon target is refused by name, so the set has 27 rows. W02's
#: ``position_multiplexing`` (owner approval of 2026-10-23, a cost-only mapping decision) is a further row.
#: Wave 2 W08 (owner Q9, 2026-10-23: lanes declared as a priced chip block) adds the lane block's two pricing-only
#: fields, read by the lane rows of the physics and the record's lane quantities only (a program never reads them).
#: CL9 final re-pin: the 32 rows are the union of the units' amendments (closure, N2, N2b, pack, lanes), equal to the
#: theorem box fixture's ``cost_only`` list (test_hypothesis_table.py) and to the X3a partition (test_cost_draws.py).
#: CL9 of integration pass 2: no unit of the pass (explore, exact, E1, E6, P1) amends the set, so it keeps its 32 rows.
#: U-F4 (owner answer of 2026-10-28, "Build it"): ``pipelined_multiplexing``, position multiplexing's pipelined form, a
#: cost-only mapping decision, is a further row.
#: U-C4 amendment (cells per stored weight, B3 intake open item 4): two pricing-only core-type fields, 34 rows on the
#: unit's base; 37 with the chain's rows (integration8/uc4).
#: [U-L2] (finalization plan item B-1, from U-L's finding): ``reduction_core_banks``, a pool lane's banked reduction
#: cores under position multiplexing, a cost-only mapping decision, is a further row; 35 rows on the unit's base, 38
#: with the chain's rows (integration8/ul2).
FROZEN_COST_ONLY = frozenset({
    "activity_factor", "allow_coalescing", "allow_neuron_splitting", "allow_scheduling",
    "core_value_granularity", "cores.count", "cores.max_axons", "cores.max_neurons",
    # U-C4 key amendment: the cells one stored weight occupies and those on adjacent rows (written cells, wear, area).
    "cores.cells_per_weight", "cores.weight_cell_rows",
    # E-6 key amendment (U-E, N2 follow-up F5): a core type's compute technology, priced per core type.
    "cores.compute_physics",
    # N2 (Wave 2) key amendment: who writes an operand core type's payloads, priced apart (design §2.9).
    "cores.operand.write_source",
    "cores.weight_sign_granularity",
    "cores.write_granularity",
    # N2b key amendment: a core type's write technology, priced per core type (design §2.9).
    "cores.write_physics",
    "cores.write_units_in_parallel", "cores_per_tile",
    # U-S key amendment: the event census's sample count, a measurement the record's energy axis prices.
    "event_census_samples",
    "inference_batch_K", "max_axons", "max_neurons",
    "max_schedule_passes", "onchip_arithmetic.lane_block", "onchip_arithmetic.lanes_per_core",
    "platform_physics_overrides", "platform_physics_profile",
    "residency_plan", "residency_plan.index", "residency_plan.step", "position_multiplexing", "pipelined_multiplexing",
    "reduction_core_banks",  # [U-L2]
    "layer_replication", "wavefront_group_cores",  # [thru:k] the layer copies of pipelined multiplexing
    "tile_grid_cols", "tile_grid_rows",
    "write_overlap", "write_overlap.mode", "write_overlap.shadow_capacity_fraction", "write_ports",
})


#: The not-a-decision set, audited against every consumer (review rounds 2 and 3): names, paths,
#: runtime facts, labels, a simulator transport, a trace, a wall-clock bound, a flag-only threshold,
#: and the reserved keys no run reads (``RESERVED_READERS``). ``inference_rate_hz`` is flagged for
#: the owner (owner_review.md).
FROZEN_NOT_A_DECISION = frozenset({
    "allow_per_layer_s", "datasets_path", "device", "experiment_name", "generate_visualizations",
    "generated_files_path", "inference_rate_hz", "input_shape", "input_size",
    "nevresim_connectivity_mode", "num_classes", "onchip_majority_fraction",
    "s_allocation", "s_allocation_budget", "s_allocation_explicit",
    "sanafe_log_potential_trace", "scheduling_latency_weight", "simulation_step_timeout_s",
    "start_step", "stop_step", "target_claim_class", "target_faithfulness_class",
    "target_overrides.*.note", "tuner_target_floor_ratio",
    # [U-H] the FPGA backend's transport (RTL co-simulation or the card, the same count refusal on either), the
    # card's index on the host and the bundle's file name: audited against the ODIN device step and the export.
    "odin_fpga_transport", "odin_fpga_device_index", "odin_hacc_bundle_name",
})

#: Reserved keys: every ``src/mimarsinan`` file outside ``config_schema/`` and ``gui/`` that names
#: the token today; none of them decides on it. A new reader fails the guard below, so the key's
#: class (not_a_decision) is revisited by an S0 amendment the moment a run starts to read it.
RESERVED_READERS = {
    r"\btuner_target_floor_ratio\b": frozenset(),
    r"\bscheduling_latency_weight\b": frozenset(),
    r"\ballow_per_layer_s\b": frozenset({"mapping/platform/mapping_structure.py"}),
    r"\bs_allocation\b": frozenset({
        "pipelining/core/deployment_plan.py", "pipelining/core/plan/predicates.py",
        "tuning/orchestration/temporal_allocation.py",
    }),
    r"\bs_allocation_(explicit|budget)\b": frozenset({"tuning/orchestration/temporal_allocation.py"}),
    # The per-depth S map reaches a run only through this call; nothing makes it.
    r"\.temporal_allocation\(": frozenset(),
}


def _every_path_class():
    for key, entry in REGISTRY.items():
        yield key, entry.decision_class
        for path, schema in _nested(entry):
            yield path, schema.decision_class


class TestPlanClasses:
    @pytest.mark.parametrize("key", [
        "inference_batch_K", "write_overlap", "write_ports", "allow_neuron_splitting",
        "allow_coalescing", "max_schedule_passes", "residency_plan",
    ])
    def test_cost_only_keys(self, key):
        assert REGISTRY[key].decision_class == COST_ONLY

    def test_leaf_decomposition_bears_on_the_function(self):
        """The strongest consumer rules (owner's S0 review): on spiking cores the moved projection rides
        the activation grid, so the outputs move; on value (MVM) cores the decomposition is exact, which the
        theorem box states as a finding, not as the key's class."""
        entry = REGISTRY["decompose_leaf_modules"]
        assert entry.decision_class == FUNCTION_BEARING
        assert "activation grid" in entry.decision_reason and "value" in entry.decision_reason

    def test_the_cost_only_set_is_the_owner_reviewed_set(self):
        cost_only = {path for path, cls in _every_path_class() if cls == COST_ONLY}
        # 26 owner-reviewed rows + the closure's CL-SG amendment (weight_sign_granularity) + the N2 key amendment
        # (write_source) + the N2b one (write_physics) + W02's position_multiplexing (pack) + W08's two lane-block
        # pricing fields (lanes) + E-6's compute_physics + U-S's event_census_samples (the event census) + U-F4's
        # pipelined_multiplexing + U-C4's two weight-cell fields (cells_per_weight, weight_cell_rows) + U-L2's
        # reduction_core_banks + the throughput branch's two layer-copy keys (thru:k)
        assert len(FROZEN_COST_ONLY) == 40
        assert cost_only == FROZEN_COST_ONLY, (
            sorted(cost_only - FROZEN_COST_ONLY), sorted(FROZEN_COST_ONLY - cost_only))

    @pytest.mark.parametrize("key", [
        "onchip_majority_gate", "nf_scm_parity_samples", "scm_torch_sim_parity_samples",
        "value_parity_samples", "spike_count_parity_samples",
        # The simulator steps refuse on a parity divergence (simulation_step.py, the integer
        # chip's window-count check; loihi_simulation_step.py and sanafe_simulation_step.py,
        # assert_spike_parity_or_raise), so their switches, sample count and sample bear on it.
        "enable_nevresim_simulation", "enable_loihi_simulation", "enable_sanafe_simulation",
        "sanafe_sample_count", "loihi_parity_sample_index",
        # The architecture the SANA-FE parity refusal simulates (TestClassEvidence): the preset picks
        # the neuron plugins (custom: the YAML's own), the path the YAML the runner builds from.
        "sanafe_arch_preset", "sanafe_custom_arch_path",
    ])
    def test_a_refusing_gate_s_switch_and_sample_count_bear_on_admissibility(self, key):
        """Disarming a gate (false, or fewer samples) admits what the armed gate would refuse."""
        assert REGISTRY[key].decision_class == ADMISSIBILITY_BEARING

    @pytest.mark.parametrize("key", [
        "enable_odin_fpga_simulation", "odin_fpga_sample_count", "odin_fpga_xclbin_path",
        "enable_odin_hacc_export", "odin_hacc_bundle_samples", "odin_hacc_certification_samples",
        # [U-HV] the split both ODIN steps draw those samples from (the development split unless declared)
        "odin_sample_split",
    ])
    def test_the_odin_backends_refusals_bear_on_admissibility(self, key):
        """[U-H] The ODIN device step refuses a count divergence from the HCM reference on the samples it runs,
        on the device image it is given (``odin_fpga_simulation_step.py``); the export refuses a twin-versus-HCM
        divergence on every shipped sample and writes no bundle past it, and the certified subset is what the
        board's exact-count check judges (``odin_hacc_deployment_step.py``, ``odin_hacc/artifact.py``)."""
        assert REGISTRY[key].decision_class == ADMISSIBILITY_BEARING

    @pytest.mark.parametrize("key", [
        "membrane_bits", "membrane_signed", "bias_row_splitting", "bias_rows_per_perceptron",
        "firing_granularity", "membrane_arithmetic",
    ])
    def test_the_soma_point_and_the_bias_rows_bear_on_the_function(self, key):
        """[U-H] A saturating register moves counts at its rails; the event-serial law moves the fire order; a
        bias spread over always-on rows takes the weight grid from max|w| alone (``soma_law.py``,
        ``weight_quantization_step.py``)."""
        assert REGISTRY[key].decision_class == FUNCTION_BEARING

    def test_the_odin_fabric_sign_claim_bears_on_admissibility(self):
        """[U-H] The top-level claim names the ODIN fabric (row pairs in logical slots); a bundle or a bitstream
        of another sign placement is refused (``odin_fpga/chip_selection.py``). The per-core-type field stays the
        release head's cost-only physical-row declaration."""
        assert REGISTRY["weight_sign_granularity"].decision_class == ADMISSIBILITY_BEARING
        assert REGISTRY["cores"].value_schema is not None

    @pytest.mark.parametrize("key", [
        "deployment_metric_full_eval", "max_simulation_samples", "simulation_batch_count",
        "simulation_batch_size",
    ])
    def test_the_sample_a_refusing_read_takes_bears_on_admissibility(self, key):
        """These keys choose the test samples of the deployed-metric reads (soft-core and hard-core
        mapping, after every tuner) that the retention gate (scm_degradation_tolerance) judges, and
        max_simulation_samples also the samples of the nevresim window-count refusal
        (simulation_factory.run_trainer_metric, SimulationRunner). With simulation_batch_count
        declared, simulation_batch_size sets how many samples a read takes (TestClassEvidence)."""
        assert REGISTRY[key].decision_class == ADMISSIBILITY_BEARING

    @pytest.mark.parametrize("key", [
        "eval_max_samples", "eval_subsample_target", "paired_confirm_batches",
        "activation_analysis_batch_size", "target_metric_override", "pretrain_floor_chance_multiple",
    ])
    def test_a_measurement_a_tuner_decides_on_bears_on_the_function(self, key):
        """The measurement rule: a sample count or threshold whose measured number a tuner's target,
        accept or rollback reads (or an analysis whose statistics set the scales) changes the deployed
        weights. eval_max_samples sets every step metric (PipelineStep._capped_test), which becomes
        the pipeline's target metric the tuners seed from (Pipeline._record_step_metric)."""
        assert REGISTRY[key].decision_class == FUNCTION_BEARING

    def test_data_loader_parallelism_bears_on_the_function(self):
        """num_workers reorders the random augmentation stream of the training loader (torch seeds a
        worker per id; in-process loading draws from the main generator), so the trained weights
        move with it (evidence/num_workers_probe.txt)."""
        assert REGISTRY["num_workers"].decision_class == FUNCTION_BEARING

    @pytest.mark.parametrize("key", [
        "tuner_target_floor_ratio", "scheduling_latency_weight", "allow_per_layer_s",
        "s_allocation", "s_allocation_explicit", "s_allocation_budget",
    ])
    def test_a_key_no_run_reads_is_not_a_decision(self, key):
        """Reserved keys are classed by what they change today: nothing a run computes, costs or
        admits (TestClassEvidence.test_a_reserved_key_gains_no_reader pins the readers)."""
        assert REGISTRY[key].decision_class == NOT_A_DECISION

    def test_a_disarming_zero_is_stated_exactly_where_the_bounds_admit_it(self):
        """A row's reason says '0 disarms' only when 0 is a legal value of that row."""
        for key, entry in REGISTRY.items():
            reason = entry.decision_reason
            low = None if entry.bounds is None else entry.bounds[0]
            admits_zero = low is not None and (low < 0 or (low == 0 and not entry.exclusive_bounds[0]))
            if "0 disarms" in reason:
                assert admits_zero, key
            if reason.startswith("sample count of a refusing") and admits_zero:
                assert "0 disarms" in reason, key

    def test_the_not_a_decision_set_is_the_audited_set(self):
        """A key reaching not_a_decision by its group's class was audited against every refusing
        step; a new key in that set needs the same audit (and an S0 amendment)."""
        not_a_decision = {path for path, cls in _every_path_class() if cls == NOT_A_DECISION}
        assert not_a_decision == FROZEN_NOT_A_DECISION, (
            sorted(not_a_decision - FROZEN_NOT_A_DECISION),
            sorted(FROZEN_NOT_A_DECISION - not_a_decision))

    def test_no_group_reason_claims_a_classification_it_does_not_check(self):
        for group, (_cls, reason) in GROUP_CLASSES.items():
            assert "every refusing gate" not in reason, group

    @pytest.mark.parametrize("key", [
        "readout_law", "activation_bits", "weight_bits", "simulation_steps", "has_bias",
        "core_semantics",
    ])
    def test_function_bearing_keys(self, key):
        assert REGISTRY[key].decision_class == FUNCTION_BEARING

    def test_admissibility_and_reporting_keys(self):
        assert REGISTRY["min_lifetime_hours"].decision_class == ADMISSIBILITY_BEARING
        assert REGISTRY["inference_rate_hz"].decision_class == NOT_A_DECISION
        for key in ("target_claim_class", "target_faithfulness_class"):
            assert REGISTRY[key].decision_class == NOT_A_DECISION

    def test_composite_keys(self):
        for key in ("target", "target_overrides"):
            assert REGISTRY[key].decision_class == COMPOSITE

    def test_core_fields(self):
        classes = {path: s.decision_class for path, s in _nested(REGISTRY["cores"])}
        for field in ("max_axons", "max_neurons", "count", "weight_sign_granularity", "write_granularity",
                      "write_units_in_parallel"):
            assert classes[f"cores.{field}"] == COST_ONLY, field
        assert classes["cores.has_bias"] == FUNCTION_BEARING
        assert REGISTRY["cores"].decision_class == FUNCTION_BEARING

    def test_the_weight_sign_granularity_field_takes_its_consumers_legal_set(self):
        """The S0 amendment for CL-SG: a core type's sign granularity, per synapse (absent, today's behaviour) or
        one sign per axon row; the legal set is the mapper's (``WEIGHT_SIGN_GRANULARITIES``), cost-only."""
        from mimarsinan.mapping.platform.platform_constraints import PER_SYNAPSE, WEIGHT_SIGN_GRANULARITIES

        schema = {path: s for path, s in _nested(REGISTRY["cores"])}["cores.weight_sign_granularity"]
        assert schema.decision_class == COST_ONLY
        assert tuple(schema.options) == WEIGHT_SIGN_GRANULARITIES == ("per_synapse", "per_axon")
        assert schema.absent_means.startswith(PER_SYNAPSE)
        assert "rows" in schema.doc


class TestClassEvidence:
    """The code facts the non-obvious rows rest on (owner_review.md, Class Evidence): if one moves,
    the row's class is revisited through an S0 amendment."""

    class _Counting(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = 0
            self.linear = torch.nn.Linear(64, 4)

        def forward(self, x):
            self.seen += int(x.shape[0])
            return self.linear(x.flatten(1))

    @pytest.mark.parametrize("batch_size,expected", [(None, 300), (8, 16), (64, 128)])
    def test_a_deployed_metric_read_takes_batch_size_times_batch_count_samples(
            self, batch_size, expected):
        """With simulation_batch_count declared, the first read's batch (the census batch
        max(test batch, 1024), capped by a declared simulation_batch_size) fixes how many test
        samples the deployed metric reads."""
        config = default_config()
        config["simulation_batch_count"] = 2
        if batch_size is not None:
            config["simulation_batch_size"] = batch_size
        pipeline = MockPipeline(config=config, data_provider_factory=MockDataProviderFactory(size=300))
        model = self._Counting()
        simulation_factory.run_trainer_metric(pipeline, model, device="cpu")
        assert model.seen == expected

    @pytest.mark.parametrize("declared,expected", [(None, 10000), (123, 123), (0, "full")])
    def test_every_step_metric_reads_the_eval_max_samples_subsample(self, declared, expected):
        """Every trainer or tuner step measures its metric (the next tuner's target) on the
        eval_max_samples subsample; 0 is no cap, the full test set (a legal value since S0 amendment B)."""
        calls = []

        class _Trainer:
            def test(self):
                calls.append("full")
                return 0.5

            def test_on_subsample(self, *, max_samples, seed):
                calls.append(max_samples)
                return 0.5

        config = default_config()
        if declared is not None:
            config["eval_max_samples"] = declared
        PipelineStep._capped_test(SimpleNamespace(pipeline=MockPipeline(config=config)), _Trainer())
        assert calls == [expected]

    @pytest.mark.parametrize("ratio", [None, 0.5, 0.99])
    def test_the_tuner_floor_never_reads_tuner_target_floor_ratio(self, ratio):
        config = {**default_config(), "degradation_tolerance": 0.05}
        if ratio is not None:
            config["tuner_target_floor_ratio"] = ratio
        adjuster = AdaptationTargetAdjuster.from_pipeline(1.0, MockPipeline(config=config))
        assert adjuster.floor == pytest.approx(0.95)

    def test_a_sanafe_preset_prices_events_only(self):
        """A built-in sanafe_arch_preset picks per-event energy and latency constants, nothing behavioural."""
        for name, preset in PRESETS.items():
            assert preset and all(k.endswith(("_energy_j", "_latency_s")) for k in preset), name

    def test_the_sanafe_architecture_keys_choose_what_the_parity_refusal_simulates(self):
        """The class of sanafe_arch_preset and sanafe_custom_arch_path (admissibility_bearing): the custom
        preset drops the built mimarsinan neuron plugins for the YAML's own, the step hands the path to
        the runner, the runner builds the architecture from the YAML whenever a path is set, and the
        step refuses a spike-parity divergence of that simulation."""
        assert set(_resolve_plugins(CUSTOM_PRESET_NAME).values()) == {""}
        step = inspect.getsource(SanafeSimulationStep.process)
        for fact in ('config.get("sanafe_arch_preset"', 'config.get("sanafe_custom_arch_path")',
                     "arch_preset=arch_preset", "custom_arch_path=custom_arch_path",
                     "assert_spike_parity_or_raise(ref, subset)"):
            assert fact in step, fact
        build = inspect.getsource(SanafeRunner._ensure_arch)
        assert build.count("custom_arch_path=self.custom_arch_path") == 2, build

    @pytest.mark.parametrize("token", sorted(RESERVED_READERS))
    def test_a_reserved_key_gains_no_reader(self, token):
        pattern, allowed = re.compile(token), RESERVED_READERS[token]
        readers = set()
        for path in _SRC.rglob("*.py"):
            rel = path.relative_to(_SRC).as_posix()
            if rel.startswith(("config_schema/", "gui/")):
                continue
            if pattern.search(path.read_text(encoding="utf-8")):
                readers.add(rel)
        assert readers <= allowed, (
            f"{token}: a run now reads it in {sorted(readers - allowed)}; revisit its class "
            "(not_a_decision) through an S0 amendment")


class TestJoin:
    HAND_TABLE = (
        ((), NOT_A_DECISION),
        (("inference_rate_hz",), NOT_A_DECISION),
        (("write_ports", "inference_rate_hz"), COST_ONLY),
        (("cores.max_axons", "cores.count", "write_ports"), COST_ONLY),
        (("cores.max_axons", "min_lifetime_hours"), ADMISSIBILITY_BEARING),
        (("cores.max_axons", "cores.has_bias"), FUNCTION_BEARING),
        (("write_ports", "min_lifetime_hours", "weight_bits"), FUNCTION_BEARING),
        (("residency_plan.step", "write_overlap.mode"), COST_ONLY),
        (("cores",), FUNCTION_BEARING),
    )

    @pytest.mark.parametrize("keys,expected", HAND_TABLE)
    def test_effective_class_is_the_join(self, keys, expected):
        assert effective_class(keys) == expected

    def test_a_composite_or_unknown_member_refuses_by_name(self):
        with pytest.raises(DecisionClassError, match="target"):
            effective_class(("target", "write_ports"))
        with pytest.raises(DecisionClassError, match="no_such_key"):
            effective_class(("no_such_key",))
        with pytest.raises(DecisionClassError, match="cores.wat"):
            effective_class(("cores.wat",))
        with pytest.raises(DecisionClassError, match=r"target_overrides\.\*"):
            effective_class(("target_overrides.*",))

    def test_a_declared_value_sets_every_nested_field_of_its_key(self):
        """A declaration replaces its key's whole value: the fields it names take their values and the
        rest their absent meanings, so every nested field of the key is set."""
        values = {
            "cores": [{"max_axons": 256, "max_neurons": 256, "count": 4}],
            "weight_bits": 8,
            "write_overlap": {"mode": "serial"},
        }
        assert declared_paths(values) == (
            "cores.cells_per_weight",  # U-C4: absent, one cell per weight (cost-only)
            "cores.compute_physics", "cores.count", "cores.has_bias", "cores.max_axons", "cores.max_neurons",
            "cores.operand.payload_bits", "cores.operand.write_source",
            "cores.weight_bits",  # Wave 2 WI-5g: absent, the platform weight_bits (function-bearing)
            "cores.weight_cell_rows",  # U-C4: absent, required above one cell per weight (cost-only)
            "cores.weight_sign_granularity",
            "cores.write_granularity", "cores.write_physics", "cores.write_units_in_parallel",
            "weight_bits", "write_overlap.mode", "write_overlap.shadow_capacity_fraction",
        )
        assert effective_class(declared_paths(values)) == FUNCTION_BEARING
        assert effective_class(declared_paths({"write_overlap": {"mode": "serial"}})) == COST_ONLY
        assert declared_paths({"weight_bits": 8, "cores": None}) == ("cores", "weight_bits")

    def test_a_cores_override_without_has_bias_is_function_bearing(self):
        """Replacing ``cores`` resets each core's bias to the platform bias mode, so an override that
        names only geometry still sets has_bias (a bundle declaring a core type's has_bias false loses it)."""
        geometry_only = {"cores": [{"max_axons": 64, "max_neurons": 64, "count": 2}]}
        assert "cores.has_bias" in declared_paths(geometry_only)
        assert effective_class(declared_paths(geometry_only)) == FUNCTION_BEARING


class TestPrinted:
    def test_the_class_is_in_the_wizard_schema_payload(self):
        payload = serialize_registry()
        json.dumps(payload)
        for key, record in payload["keys"].items():
            assert record["decision_class"] == REGISTRY[key].decision_class, key

    def test_the_declaration_table_prints_every_key_and_nested_field(self, capsys):
        script = _load_script("declaration_table")
        script.main(["--json"])
        rows = json.loads(capsys.readouterr().out)
        by_key = {row["key"]: row for row in rows}
        assert set(REGISTRY) <= set(by_key)
        assert by_key["cores.write_granularity"]["decision_class"] == COST_ONLY
        assert by_key["readout_law.range_value"]["legal_values"] == "FLOAT > 0"
        assert by_key["target"]["decision_class"] == COMPOSITE
        order = [row["decision_class"] for row in rows]
        assert order[: order.count(COST_ONLY)] == [COST_ONLY] * order.count(COST_ONLY)

    def test_every_row_states_its_absent_meaning(self, capsys):
        script = _load_script("declaration_table")
        script.main(["--json"])
        rows = json.loads(capsys.readouterr().out)
        silent = [row["key"] for row in rows
                  if not row["absent_means"].strip() or row["absent_means"] == "\u2014"]
        assert silent == []
        by_key = {row["key"]: row["absent_means"] for row in rows}
        # The consumer's frozen fallback, the derivation, the runtime fact, the resolver's bias lane.
        assert by_key["core_value_granularity"] == "frozen default {}"
        assert by_key["platform_physics_overrides"] == "frozen default {}"
        assert by_key["capacity_gate"] == "frozen default true"
        assert by_key["onchip_min_fraction"] == "frozen default 0.2"
        assert by_key["max_axons"] == "derived from cores"
        assert by_key["spiking_mode"] == "derived from spiking_family, spiking_variant; not declarable"
        assert by_key["device"] == "runtime fact; not declarable"
        assert by_key["has_bias"].startswith("true")
        assert by_key["generate_visualizations"].startswith("unset")
        # The first read's batch, not the retry fallback alone (simulation_factory, F3).
        assert by_key["simulation_batch_size"].startswith("8 on the spiking hard-core read")
        assert "max(test batch, 1024)" in by_key["simulation_batch_size"]

    def test_reserved_values_print_as_reserved_and_legal_sets_are_the_consumers(self, capsys):
        """The table prints a key's legal set (the consumer's) and names its reserved values as reserved;
        a value the consumer refuses is printed nowhere."""
        script = _load_script("declaration_table")
        script.main(["--json"])
        by_key = {row["key"]: row for row in json.loads(capsys.readouterr().out)}
        assert by_key["s_allocation"]["legal_values"] == "ENUM uniform (reserved: explicit, budget)"
        assert by_key["nevresim_connectivity_mode"]["legal_values"] == "ENUM compile_time | runtime"
        script.main([])
        markdown = capsys.readouterr().out
        assert "codegen" not in markdown
        assert "ENUM uniform (reserved: explicit, budget)" in markdown

    @pytest.mark.parametrize("key", ["cores", "training_recipe", "tuning_recipe"])
    def test_an_absent_value_is_printed_in_full(self, key, capsys):
        """A long default is never cut: the JSON rows and the markdown table carry all of it."""
        script = _load_script("declaration_table")
        script.main(["--json"])
        by_key = {row["key"]: row["absent_means"] for row in json.loads(capsys.readouterr().out)}
        full = "default " + json.dumps(REGISTRY[key].default)
        assert by_key[key] == full
        script.main([])
        assert full in capsys.readouterr().out

    def test_no_cell_is_cut(self, capsys):
        script = _load_script("declaration_table")
        script.main(["--json"])
        rows = json.loads(capsys.readouterr().out)
        cut = [(row["key"], column) for row in rows for column, cell in row.items()
               if cell.endswith("...")]
        assert cut == []
