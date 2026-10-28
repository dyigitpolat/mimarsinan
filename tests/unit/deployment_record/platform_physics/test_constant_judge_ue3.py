"""[U-E3 Z-2] One constant judge below the converter and the loader, every import at module top.

A converter table's rows (E-5) are judged as a profile's constants are. The judge (``constant_from_declaration``) and
the value it builds (``PhysicsConstantValue``) live in ``platform_physics.constants``, below both the converter and
the loader: ``profile`` imports the converter at top (a profile carries its converter table), so a judge that lives
in the loader, or a value type that lives in ``profile``, would close an init cycle (package init, loader, profile,
converter, loader) that only a function-level import could break, which the lazy-import ratchet forbids.

Each module of the cycle, and each module whose function-level import moved to its top, imports first in a fresh
interpreter: the order of first import can no longer fail."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import mimarsinan.deployment_record.platform_physics.constants as constants
import mimarsinan.deployment_record.platform_physics.converter as converter
import mimarsinan.deployment_record.platform_physics.loader as loader
import mimarsinan.deployment_record.platform_physics.profile as profile
import mimarsinan.deployment_record.platform_physics.resolve as resolve

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"

#: The modules of the converter <-> loader cycle and of the moved imports (U-E's five files and their targets).
FIRST_IMPORTS = (
    "mimarsinan.deployment_record.platform_physics",
    "mimarsinan.deployment_record.platform_physics.constants",
    "mimarsinan.deployment_record.platform_physics.converter",
    "mimarsinan.deployment_record.platform_physics.loader",
    "mimarsinan.deployment_record.platform_physics.profile",
    "mimarsinan.deployment_record.platform_physics.resolve",
    "mimarsinan.deployment_record.quantities.compute_technology",
    "mimarsinan.deployment_record.cost.absolute.formulas",
    "mimarsinan.deployment_record.cost.absolute.report",
)


class TestOneJudge:
    def test_the_converter_and_the_loader_read_the_one_judge_below_them(self):
        assert converter.constant_from_declaration is constants.constant_from_declaration
        assert loader.constant_from_declaration is constants.constant_from_declaration
        assert resolve.constant_from_declaration is constants.constant_from_declaration

    def test_the_value_type_is_one_class_re_exported_by_the_profile(self):
        assert profile.PhysicsConstantValue is constants.PhysicsConstantValue
        assert profile.EVIDENCE_KINDS is constants.EVIDENCE_KINDS

    def test_a_converter_row_is_judged_as_a_profile_constant(self):
        row = {"nominal": 2.0, "unit": "pJ", "evidence_kind": "derived", "derivation": "test arithmetic"}
        judged = converter.constant_from_declaration("e_adc_conversion", row)
        assert isinstance(judged, profile.PhysicsConstantValue)
        assert judged == loader.constant_from_declaration("e_adc_conversion", row)
        with pytest.raises(ValueError, match="unknown fields"):
            converter.constant_from_declaration("e_adc_conversion", {**row, "stray": 1})


@pytest.mark.parametrize("module", FIRST_IMPORTS)
def test_each_module_imports_first_in_a_fresh_interpreter(module):
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run([sys.executable, "-B", "-c", f"import {module}"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
