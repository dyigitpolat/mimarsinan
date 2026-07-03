"""Host-contract classes must stay annotation-only and in sync with their hosts."""

import dataclasses

from mimarsinan.search.optimizers.agent_evolve.batch_eval import BatchEvalMixin
from mimarsinan.search.optimizers.agent_evolve.host_contract import EvolveHostContract
from mimarsinan.search.optimizers.agent_evolve.optimizer import AgentEvolveOptimizer
from mimarsinan.search.optimizers.agent_evolve.prompting import PromptingMixin
from mimarsinan.search.problems.joint.evaluate import JointEvaluateMixin
from mimarsinan.search.problems.joint.layout_hook import JointLayoutMixin
from mimarsinan.search.problems.joint.problem import JointArchHwProblem
from mimarsinan.search.problems.joint.types import JointHostContract
from mimarsinan.search.problems.joint.validate import JointValidateMixin

EVOLVE_CONTRACT_METHODS = [
    "_log",
    "_report_search_event",
    "_llm_call",
    "_coerce_llm_text",
    "_generate_initial_candidates",
    "_regenerate_candidates",
    "_generate_offspring",
    "_regenerate_offspring",
    "_apply_failure_insights",
    "_refresh_constraint_instruction",
]

JOINT_CONTRACT_METHODS = [
    "objectives",
    "validate_detailed",
    "_penalty_objectives",
    "_ensure_hw_only_cache",
    "_build_raw_model",
    "_ensure_mapper_repr",
    "_collect_softcores",
    "_compute_hw_objectives",
]


def _runtime_members(cls):
    return {name for name in vars(cls) if not name.startswith("__")}


class TestContractsAreAnnotationOnly:
    def test_evolve_contract_has_no_runtime_members(self):
        assert _runtime_members(EvolveHostContract) == set()

    def test_joint_contract_has_no_runtime_members(self):
        assert _runtime_members(JointHostContract) == set()


class TestMixinsInheritContracts:
    def test_agent_evolve_mixins(self):
        assert issubclass(BatchEvalMixin, EvolveHostContract)
        assert issubclass(PromptingMixin, EvolveHostContract)
        assert issubclass(AgentEvolveOptimizer, EvolveHostContract)

    def test_joint_mixins(self):
        assert issubclass(JointEvaluateMixin, JointHostContract)
        assert issubclass(JointLayoutMixin, JointHostContract)
        assert issubclass(JointValidateMixin, JointHostContract)
        assert issubclass(JointArchHwProblem, JointHostContract)


class TestContractsCoveredByHosts:
    def test_evolve_attributes_are_host_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(AgentEvolveOptimizer)}
        for name in EvolveHostContract.__annotations__:
            assert name in field_names, f"contract attr {name!r} missing on host"

    def test_evolve_methods_exist_on_host(self):
        for name in EVOLVE_CONTRACT_METHODS:
            assert callable(getattr(AgentEvolveOptimizer, name)), name

    def test_joint_attributes_are_host_dataclass_fields(self):
        field_names = {f.name for f in dataclasses.fields(JointArchHwProblem)}
        for name in JointHostContract.__annotations__:
            assert name in field_names, f"contract attr {name!r} missing on host"

    def test_joint_methods_exist_on_host(self):
        for name in JOINT_CONTRACT_METHODS:
            assert getattr(JointArchHwProblem, name, None) is not None, name


class TestContractsDoNotLeakIntoDataclassFields:
    def test_contract_annotations_do_not_add_init_fields(self):
        init_params = {
            f.name for f in dataclasses.fields(AgentEvolveOptimizer) if f.init
        }
        assert "pop_size" in init_params
        joint_init = {f.name for f in dataclasses.fields(JointArchHwProblem) if f.init}
        assert "_cache" not in joint_init
