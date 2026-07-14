"""target_tq / simulation_steps consistency: the documented ``Tq | S`` contract
is FAIL-LOUD, not honor-system.

target_tq (Tq) is the QAT activation-quantization level (the AQ capacity); it is
consumed by a different subsystem than simulation_steps (S), the deployment spike
window. The deployment identity requires the QAT grid to tile the window, so the
registry doc says "target_tq must divide simulation_steps" — enforced here at
platform-constraint resolution so a silent misconfig (e.g. Tq=6, S=8) cannot
resolve into a broken deployment.
"""

from __future__ import annotations

import pytest

from mimarsinan.config_schema.deployment_derivation import derive_platform_constraints


class TestTqDividesS:
    def test_equal_is_consistent(self):
        # The tightest identity (Tq == S), used by every tier-0/0.1 config.
        for v in (4, 8, 16, 32, 64):
            pc = {"target_tq": v, "simulation_steps": v}
            derive_platform_constraints(pc, cores_declared=False)  # no raise

    def test_proper_divisor_is_consistent(self):
        # Tq strictly divides S — the separate-axes regime the contract allows.
        pc = {"target_tq": 4, "simulation_steps": 8}
        derive_platform_constraints(pc, cores_declared=False)  # no raise

    def test_non_divisor_fails_loud(self):
        pc = {"target_tq": 6, "simulation_steps": 8}
        with pytest.raises(ValueError, match="target_tq.*divide.*simulation_steps|divide"):
            derive_platform_constraints(pc, cores_declared=False)

    def test_tq_greater_than_s_fails_loud(self):
        pc = {"target_tq": 16, "simulation_steps": 8}
        with pytest.raises(ValueError, match="divide"):
            derive_platform_constraints(pc, cores_declared=False)

    def test_only_tq_present_does_not_fire(self):
        # The architecture-search path sets target_tq without simulation_steps;
        # a one-operand contract is not checkable and must not raise.
        pc = {"target_tq": 32}
        derive_platform_constraints(pc, cores_declared=False)  # no raise

    def test_only_s_present_does_not_fire(self):
        pc = {"simulation_steps": 16}
        derive_platform_constraints(pc, cores_declared=False)  # no raise

    def test_error_names_both_values(self):
        pc = {"target_tq": 3, "simulation_steps": 8}
        with pytest.raises(ValueError) as exc:
            derive_platform_constraints(pc, cores_declared=False)
        assert "3" in str(exc.value) and "8" in str(exc.value)
