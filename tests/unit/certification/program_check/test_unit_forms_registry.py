"""[E6] The registry knows every program form the integrated units build (the form census of the integrated head,
``RA/11_wave2/integration2/e6/census``): each unit's program scans with no unknown and no unread form, so W15
establishes it or fails it by a named condition; an IR weight bank is an element of its own (an attribute a unit adds
to a bank fails by name); an element without an instance dict is read by its slots and its class, never a crash."""

from __future__ import annotations

import copy

import pytest

from mimarsinan.certification.program_check.forms import registered_attributes, scan_forms
from mimarsinan.certification.program_check.forms import base as B

from unit.certification.program_check import programs as P
from unit.certification.program_check import unit_vehicles as UV


@pytest.mark.parametrize("name", sorted(UV.BUILDERS))
def test_every_integrated_units_program_scans_with_no_unknown_or_unread_form(name):
    program, ir, _decl = UV.BUILDERS[name]()
    found = scan_forms(program, ir)
    assert not found, [v.text() for v in found]


@pytest.mark.parametrize("name", sorted(UV.BUILDERS))
def test_no_integrated_units_verdict_names_an_unknown_form(name):
    program, ir, decl = UV.BUILDERS[name]()
    report = P.run(program, ir, decl)
    assert not report.unknown_forms and not report.unread_forms
    assert "unknown program form" not in report.theorem1 and "unread program form" not in report.theorem1


#: Each integrated unit's attributes with the readers the registry gives them (the census rows and their readers).
EXPECTED = {
    ("program", "dynamic_bank_bindings"): {"tiles", "H9", "H8"},
    ("hard_core", "weight_bits"): {"H8"},
    ("hard_core", "accepted_roles"): {B.UNREAD},
    ("hard_core", "lane_operand_axons"): {B.UNREAD},
    ("ir_graph", "dynamic_bank_bindings"): {"tiles", B.IDENTITY},
    ("ir_graph", "lane_fusion_report"): {B.UNREAD},
    ("ir_node", "core_role"): {B.IDENTITY},
    ("ir_node", "weight_bits"): {B.IDENTITY},
    ("ir_node", "lane_operand_axons"): {B.IDENTITY},
    ("ir_node", "lane_operand_rows"): {B.UNREAD},
    ("ir_node", "lane_owner"): {B.UNREAD},
    ("ir_node", "lane_outputs"): {"ir-tiling"},
    ("ir_node", "lane_source_name"): {"ir-tiling"},
    ("ir_node", "lane_producer_theta"): {B.UNREAD},
    ("ir_node", "lane_reduced_instances"): {B.UNREAD},
    ("ir_node", "pruned_row_magnitude"): {"N-identity"},
    ("ir_node", "_depth_relay"): {"ir-tiling"},
    ("ir_bank", "weight_bits"): {B.IDENTITY},
    ("ir_bank", "core_matrix"): {B.IDENTITY},
}


@pytest.mark.parametrize("element,attribute", sorted(EXPECTED))
def test_each_integrated_attribute_is_registered_with_its_readers(element, attribute):
    readers = registered_attributes()[element].get(attribute)
    assert readers is not None, f"{element}.{attribute} is not registered"
    assert set(readers) == EXPECTED[(element, attribute)], readers


def test_the_columnar_axon_source_is_a_registered_class():
    assert "ColumnSpikeSource" in B.CLASSES["axon_source"]
    assert "WeightBank" in B.CLASSES["ir_bank"]


def test_an_ir_bank_attribute_a_unit_adds_fails_by_name():
    program, ir = P.resident_chain()
    bad = copy.deepcopy(ir)
    setattr(next(iter(bad.weight_banks.values())), "zz_unit_bank_field", 3)
    verdict = P.run(program, bad, P.decl("synchronized_lif", core_types=UV.CHAIN_CORES)).theorem1
    assert verdict.startswith("failed: unknown program form") and "ir_bank.zz_unit_bank_field" in verdict, verdict


def test_an_element_without_an_instance_dict_is_scanned_by_its_slots_and_class():
    """A slotted stand-in for an I/O slice (no ``__dict__``): the scan names its class and reads its slots; it never
    raises ``vars()``'s TypeError (the census met one on a GUI test's malformed program)."""

    class SlotSlice:
        __slots__ = ("node_id", "offset", "size", "zz_slot_field")

        def __init__(self, s):
            self.node_id, self.offset, self.size, self.zz_slot_field = s.node_id, s.offset, s.size, 1

    program, ir = P.streamed_multipass()
    bad = copy.deepcopy(program)
    stage = P.neural(bad)[0]
    stage.input_map = [SlotSlice(s) for s in stage.input_map]
    found = {v.rule for v in scan_forms(bad, ir)}
    assert "io_slice:SlotSlice" in found and "io_slice.zz_slot_field" in found, found
