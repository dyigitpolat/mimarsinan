"""[ODIN6] Template expansion: every parameter substituted, nothing left behind.

The RTL is not synthesized in Python — it is one reviewed template in ``hw/gen``
whose parameters are filled from the spec. Two failure modes matter and both are
gated here: a placeholder that survives (a template shipped into a simulator)
and a substitution table that names a placeholder the template no longer has (a
renamed parameter silently keeping its old default).

The rendered file also carries the licence header and the SHL-2.0 §4(b)
statement of changes the plan requires of derivative RTL.
"""

import pytest

from mimarsinan.chip_simulation.soma_law import SomaLaw
from mimarsinan.mapping.export.odin_gen.render import (
    CORE_MODULE,
    render_core_rtl,
    spec_flags_word,
    substitution_table,
)
from mimarsinan.mapping.export.odin_gen.spec import CoreSpec
from mimarsinan.mapping.export.odin_gen.templates import (
    TemplateError,
    assert_no_placeholders,
    expand,
    placeholders_in,
    template_path,
)

_LIF = {"spiking_family": "lif", "spiking_variant": "streamed",
        "firing_mode": "Novena", "thresholding_mode": "<="}
_PER_EVENT = SomaLaw.resolve({**_LIF, "firing_granularity": "per_event",
                              "membrane_bits": 8})
_SYNC_FIRE = SomaLaw.resolve({**_LIF, "membrane_bits": 16,
                              "membrane_signed": True})


def _spec(law=_PER_EVENT, axons=128, neurons=128):
    return CoreSpec.project(
        {"max_axons": axons, "max_neurons": neurons, "count": 1,
         "has_bias": False},
        soma_law=law, weight_bits=4, weight_sign_granularity="per_synapse")


class TestTheExpanderIsCheckedBothWays:
    def test_it_substitutes_the_declared_placeholders(self):
        assert expand("a=@X@ b=@Y@", {"X": 1, "Y": 2}) == "a=1 b=2"

    def test_a_missing_value_refuses_instead_of_shipping_a_template(self):
        with pytest.raises(TemplateError, match="Y"):
            expand("a=@X@ b=@Y@", {"X": 1})

    def test_a_value_the_template_does_not_use_refuses(self):
        with pytest.raises(TemplateError, match="Z"):
            expand("a=@X@", {"X": 1, "Z": 2})

    def test_verilog_concatenation_braces_are_not_placeholders(self):
        """The marker is ``@NAME@`` precisely because ``{{`` is Verilog."""
        text = "wire [3:0] w = {{ {2{1'b0}}, x }};"
        assert placeholders_in(text) == ()
        assert expand(text, {}) == text

    def test_an_unknown_template_name_refuses_rather_than_synthesizing_text(self):
        with pytest.raises(TemplateError, match="no template at"):
            template_path("not_a_template.v")


class TestTheCoreTemplateIsFullyParameterised:
    def test_every_placeholder_has_a_value_for_every_shipped_spec(self):
        text = template_path("odin_gen_core.v").read_text(encoding="utf-8")
        names = set(placeholders_in(text))
        assert names, "the core template declares no parameters at all"
        for law in (_PER_EVENT, _SYNC_FIRE):
            assert set(substitution_table(_spec(law))) == names

    def test_the_rendered_file_carries_no_placeholder(self):
        assert_no_placeholders(render_core_rtl(_spec()))
        with pytest.raises(TemplateError, match="LEFTOVER"):
            assert_no_placeholders("parameter X = @LEFTOVER@;")

    def test_it_declares_the_module_the_testbench_binds(self):
        assert f"module {CORE_MODULE}" in render_core_rtl(_spec())

    def test_the_licence_header_and_statement_of_changes_survive(self):
        rendered = render_core_rtl(_spec())
        assert "Solderpad Hardware" in rendered
        assert "http://solderpad.org/licenses/SHL-2.0/" in rendered
        assert "UCLouvain" in rendered
        assert "STATEMENT OF CHANGES (Solderpad Hardware License v2.0, "\
            "section 4(b))" in rendered
        assert "ChFrenkel/ODIN" in rendered
        assert "1781931" in rendered


class TestTheGeometryReachesTheEmittedParameters:
    @pytest.mark.parametrize("axons,neurons,aw,nw", [
        (128, 128, 7, 7), (512, 256, 9, 8), (256, 256, 8, 8),
    ])
    def test_the_address_widths_are_emitted(self, axons, neurons, aw, nw):
        table = substitution_table(_spec(axons=axons, neurons=neurons))
        assert table["AXONS"] == axons and table["NEURONS"] == neurons
        assert table["AW"] == aw and table["NW"] == nw
        assert table["NWM1"] == nw - 1

    def test_the_two_laws_emit_different_parameters(self):
        event = substitution_table(_spec(_PER_EVENT))
        sync = substitution_table(_spec(_SYNC_FIRE))
        assert (event["PER_EVENT"], event["MSIGNED"], event["ASSERT_NO_SAT"]) \
            == (1, 0, 0)
        assert (sync["PER_EVENT"], sync["MSIGNED"], sync["ASSERT_NO_SAT"]) \
            == (0, 1, 1)
        assert (event["V_LO"], event["V_HI"]) == (0, 255)
        assert (sync["V_LO"], sync["V_HI"]) == (-32768, 32767)

    def test_the_two_laws_render_to_different_rtl(self):
        assert render_core_rtl(_spec(_PER_EVENT)) != render_core_rtl(_spec(_SYNC_FIRE))

    def test_rendering_is_deterministic(self):
        assert render_core_rtl(_spec()) == render_core_rtl(_spec())


class TestTheSpecFlagsWordIsTheTestbenchContract:
    """The word the generated core drives on ``SPEC_FLAGS`` and the harness
    compares against; a disagreement here is what makes SPECFAIL fire."""

    def test_the_bit_order_is_the_templates_concatenation(self):
        assert spec_flags_word(_spec(_PER_EVENT)) == 0b01110
        assert spec_flags_word(_spec(_SYNC_FIRE)) == 0b11101

    def test_every_bit_is_reachable_from_a_declared_law(self):
        strict = SomaLaw(
            firing_mode="Default", thresholding_mode="<",
            firing_granularity="per_cycle",
            membrane_arithmetic="saturating_unsigned", membrane_bits=8)
        assert spec_flags_word(_spec(strict)) == 0
