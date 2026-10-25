"""Chip-row declarations: a row whose core types name several write technologies, or an operand core the registry
does not declare, refuses by name; a one-technology row is buildable."""

from __future__ import annotations

import pytest

from mimarsinan.config_schema.registry.entries_write import CORE_TYPE_SCHEMA
from mimarsinan.deployment_record.design_point.chip_rows import (
    OPERAND_CORE_TYPE, PER_CORE_TYPE_WRITE_TECHNOLOGY, RowDeclarationError, core_type_record, declaration_refusals,
)

STATIC = {"role": "static", "max_axons": 256, "max_neurons": 256, "count": 6, "write_technology": "neurram"}
OPERAND = {"role": "operand", "max_axons": 128, "max_neurons": 1024, "count": 2, "write_technology": "sram_cim_macro",
           "operand": {"payload_bits": 8, "write_source": "host"}}


def test_two_write_technologies_refuse_by_name_naming_both():
    refusals = declaration_refusals([STATIC, {**OPERAND, "operand": None}])
    assert [r["name"] for r in refusals] == [PER_CORE_TYPE_WRITE_TECHNOLOGY]
    assert "neurram" in refusals[0]["reason"] and "sram_cim_macro" in refusals[0]["reason"]


def test_an_operand_core_type_refuses_while_the_registry_declares_none(monkeypatch):
    """The rule reads the registry. [integration of CL7 with N2] N2 registers the core-type ``operand`` field, so on
    the integrated tree the operand refusal lifts, and it returns wherever the schema declares no such field."""
    import mimarsinan.deployment_record.design_point.chip_rows as chip_rows

    assert "operand" in CORE_TYPE_SCHEMA.fields
    assert [r["name"] for r in declaration_refusals([STATIC, OPERAND])] == [PER_CORE_TYPE_WRITE_TECHNOLOGY]
    without = type("Schema", (), {"fields": {k: v for k, v in CORE_TYPE_SCHEMA.fields.items() if k != "operand"}})
    monkeypatch.setattr(chip_rows, "CORE_TYPE_SCHEMA", without)
    names = [r["name"] for r in declaration_refusals([STATIC, OPERAND])]
    assert names == [PER_CORE_TYPE_WRITE_TECHNOLOGY, OPERAND_CORE_TYPE]


def test_one_technology_is_buildable():
    assert declaration_refusals([STATIC, {**STATIC, "role": "any"}]) == []
    assert declaration_refusals([{"max_axons": 128, "max_neurons": 1024}]) == []


def test_a_core_type_record_keeps_role_geometry_and_technology():
    record = core_type_record(OPERAND)
    assert record["role"] == "operand" and record["write_technology"] == "sram_cim_macro"
    assert record["operand"] == {"payload_bits": 8, "write_source": "host"}
    with pytest.raises(RowDeclarationError):
        core_type_record({**STATIC, "role": "lane"})
    with pytest.raises(RowDeclarationError):
        core_type_record({"role": "static"})
