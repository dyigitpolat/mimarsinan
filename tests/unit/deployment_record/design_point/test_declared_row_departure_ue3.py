"""[U-E3 Z-1] A row declared by its core types (E-6) under the chip-description input width (U-C C-5).

The instance's build declares the 8-bit value grid of its integer readout law (``BUILD_DECLARATIONS``). A declaration
over a target whose chip sets another input width (the hybrid's ``neurram``, 4 bits) may not override it silently: the
build's width is a declared departure (``target_overrides`` with its note), which the resolved platform's faithfulness
label and every record name. The test vehicle of the per-core-type rows declares it the same way."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from mimarsinan.mapping.platform.target_bundles import target_departures

from . import programs as P

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "design_point_instance.py"
OPERAND = {"payload_bits": 8, "write_source": "host"}
ROW = {"row": "mixed", "role": "figure", "title": "Two technologies, declared", "beside_of": None,
       "core_types": [{"role": "static", "target": "neurram"},
                      {"role": "operand", "target": "sram_cim_macro", "operand": OPERAND}],
       "declaration": "tests/fixtures/design_point/hybrid_neurram_sram_cim.json", "core_counts": [8, 4],
       "declare": {"allow_scheduling": True, "max_schedule_passes": 4096}, "arithmetic_class": "test"}


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("design_point_instance_ue3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _width(departures):
    return [d for d in departures if d["key"] == "activation_bits"]


def test_the_declared_row_resolves_with_the_build_width_as_a_named_departure(driver):
    platform = driver.declared_platform({}, ROW, None, 1)
    assert platform["activation_bits"] == 8
    (width,) = _width(target_departures(platform))
    assert (width["bundle"], width["declared"]) == (4, 8) and "BUILD_DECLARATIONS" in width["note"]
    assert "activation_bits declared" in platform["target_faithfulness_class"]


def test_a_build_declaration_the_target_shares_stays_a_plain_declaration(driver, monkeypatch):
    """Only a departing build key goes through ``target_overrides``: one equal to the target's is declared as is."""
    monkeypatch.setitem(driver.BUILD_DECLARATIONS, "activation_bits", 4)
    platform = driver.declared_platform({}, ROW, None, 1)
    assert platform["activation_bits"] == 4 and not _width(target_departures(platform))
    assert "activation_bits declared" not in platform["target_faithfulness_class"]


def test_the_judged_chip_block_states_the_resolved_class_and_each_core_type_s_departure(driver):
    platform = driver.declared_platform({}, ROW, None, 1)
    block = driver.chip_block({}, ROW, 12, resolved=platform)
    assert block["faithfulness"]["class"] == platform["target_faithfulness_class"]
    # [U-CHIPKEYS] the build's lane-free program departs from NeuRRAM's stated ReLU in its neuron (on-chip class L)
    assert block["faithfulness"]["named"] == [
        "activation_bits declared (static core type, bundle neurram)",
        "onchip_arithmetic declared (static core type, bundle neurram)",
        "activation_bits declared (operand core type, bundle sram_cim_macro)"]
    assert block["weight_bits"] == int(platform["weight_bits"])


def test_the_per_core_type_vehicle_declares_its_width_as_a_departure():
    platform = P.resolved_hybrid(1, 1, scheduling=True)
    (width,) = _width(target_departures(platform))
    assert (width["bundle"], width["declared"]) == (4, 8)
