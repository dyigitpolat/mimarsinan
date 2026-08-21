"""[ODIN P7a] the third backend-enable state: supported, but OPT-IN.

Capability and AVAILABILITY are different questions. A software simulator that
the mode supports is on by default, because running it costs nothing a config
cannot supply. A backend that needs a DEVICE — an RTL simulator, an Alveo card
— is admitted by the same capability derivation and still stays OFF until the
document asks for it, because no config can assert that a board exists.

Collapsing the two would give exactly one of two wrong answers: every run
reaching for hardware, or hardware unreachable by declaration.
"""

from __future__ import annotations

import pytest

from mimarsinan.config_schema.deployment_derivation import (
    derive_deployment_parameters,
)
from mimarsinan.config_schema.recipe_fold import resolve_backend_enable
from mimarsinan.tuning.orchestration.conversion_policy import (
    DEVICE_OPT_IN_ENABLES,
    ConversionPolicy,
)

ODIN_KEY = "enable_odin_fpga_simulation"


def _resolved(**document):
    base = {"configuration_mode": "user", "model_type": "mlp_mixer",
            "spiking_family": "lif", "spiking_variant": "streamed"}
    base.update(document)
    derive_deployment_parameters(base, explicit_keys=set(base))
    return base


class TestTheRuleItself:
    def test_capability_off_is_off_whatever_the_document_says(self):
        assert resolve_backend_enable(
            supported=False, declared=True, opt_in=False) is False
        assert resolve_backend_enable(
            supported=False, declared=True, opt_in=True) is False

    def test_a_supported_software_backend_defaults_on(self):
        assert resolve_backend_enable(
            supported=True, declared=None, opt_in=False) is True
        assert resolve_backend_enable(
            supported=True, declared=False, opt_in=False) is False

    def test_a_supported_device_backend_defaults_off_and_honors_an_explicit_on(self):
        assert resolve_backend_enable(
            supported=True, declared=None, opt_in=True) is False
        assert resolve_backend_enable(
            supported=True, declared=True, opt_in=True) is True
        assert resolve_backend_enable(
            supported=True, declared=False, opt_in=True) is False


class TestTheOdinDeviceIsTheOptInMember:
    def test_the_recipe_admits_it_under_the_streamed_lif_law(self):
        recipe = ConversionPolicy.derive("lif", None)
        assert recipe.sim_enables[ODIN_KEY] is True
        assert ODIN_KEY in recipe.sim_opt_in
        assert recipe.sim_opt_in == DEVICE_OPT_IN_ENABLES

    def test_every_ttfs_family_refuses_it_by_capability(self):
        for mode, schedule in (("ttfs", None), ("ttfs_quantized", None),
                               ("ttfs_cycle_based", "cascaded"),
                               ("ttfs_cycle_based", "synchronized")):
            recipe = ConversionPolicy.derive(mode, schedule)
            assert recipe.sim_enables[ODIN_KEY] is False, (mode, schedule)

    def test_a_silent_document_resolves_it_off(self):
        assert _resolved()[ODIN_KEY] is False

    def test_an_explicit_declaration_turns_it_on(self):
        assert _resolved(**{ODIN_KEY: True})[ODIN_KEY] is True

    def test_an_explicit_off_stays_off(self):
        assert _resolved(**{ODIN_KEY: False})[ODIN_KEY] is False

    def test_an_explicit_on_against_a_mode_with_no_executor_is_a_keyed_error(self):
        with pytest.raises(ValueError, match=ODIN_KEY):
            _resolved(spiking_family="ttfs", spiking_variant="analytical",
                      **{ODIN_KEY: True})

    def test_the_software_simulators_keep_their_default_on_behaviour(self):
        resolved = _resolved()
        assert resolved["enable_nevresim_simulation"] is True
        assert resolved["enable_sanafe_simulation"] is True
