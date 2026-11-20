"""[W15, W33] A backend runs only the execution extensions it implements: the function-bearing contract fields the
reference executor honours beyond the semantics (phase dither, the membrane guard) are declared per backend, and an
enabled backend that does not implement a declared extension refuses the plan by name at assembly, instead of running
another function (``RA/11_wave2/exact/review_soundness.md`` M3: 15 of 16 programs differed on every backend)."""

import pytest

from mimarsinan.chip_simulation.backend import (
    BACKEND_REGISTRY,
    EXECUTION_EXTENSIONS,
    BackendExtensionRefusal,
    declared_extensions,
)
from mimarsinan.common.refusal import ToolchainRefusal
from mimarsinan.config_schema.registry.decision_classes import FUNCTION_BEARING, class_of_path
from mimarsinan.pipelining.core.deployment_plan import DeploymentPlan
from spiking_points import declaring_fire_while_integrating


def _plan(**cfg) -> DeploymentPlan:
    base = {"configuration_mode": "user", "spiking_mode": "lif", "model_type": "mlp_mixer"}
    base.update(cfg)
    return DeploymentPlan.resolve(declaring_fire_while_integrating(base))  # [U-LIF2 W16] the backends fire per cycle


@pytest.mark.parametrize("backend", ["nevresim", "sanafe", "loihi"])
@pytest.mark.parametrize("key,value", [("spike_phase_dither", True), ("lif_membrane_init", 0.5)])
def test_an_enabled_backend_refuses_an_extension_it_does_not_run(backend, key, value):
    plan = _plan(**{key: value})
    with pytest.raises(BackendExtensionRefusal, match=key) as caught:
        BACKEND_REGISTRY.get(backend).require_supported(plan, context="ctx")
    assert isinstance(caught.value, ToolchainRefusal)


def test_the_assembly_refuses_before_any_step_is_appended():
    with pytest.raises(BackendExtensionRefusal, match="nevresim"):
        BACKEND_REGISTRY.selected_step_specs(_plan(spike_phase_dither=True, enable_nevresim_simulation=True))
    specs = BACKEND_REGISTRY.selected_step_specs(_plan(spike_phase_dither=True, enable_nevresim_simulation=False))
    assert [name for name, _ in specs] == []


def test_neutral_values_declare_no_extension():
    assert declared_extensions(_plan().config) == ()
    assert declared_extensions(_plan(lif_membrane_init=0.0, spike_phase_dither=False).config) == ()
    assert declared_extensions(_plan(spike_phase_dither=True, lif_membrane_init=-0.25).config) == (
        "lif_membrane_init", "spike_phase_dither")
    BACKEND_REGISTRY.get("nevresim").require_supported(_plan(), context="ctx")


def test_every_extension_is_function_bearing_in_the_decision_class_registry():
    from mimarsinan.config_schema.registry import REGISTRY

    for key in EXECUTION_EXTENSIONS:
        assert class_of_path(REGISTRY, key) == FUNCTION_BEARING, key
