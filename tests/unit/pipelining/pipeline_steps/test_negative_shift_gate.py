"""negative_value_shift mode gate: unsupported spiking modes must fail loud."""

from __future__ import annotations

import pytest

from conftest import MockPipeline
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


class _EmptyTrainer:
    def iter_validation_batches(self, n):
        return iter([])


def _step(*, negative_value_shift: bool, spiking_mode: str) -> SoftCoreMappingStep:
    pipeline = MockPipeline()
    pipeline.config["negative_value_shift"] = negative_value_shift
    pipeline.config["spiking_mode"] = spiking_mode
    pipeline.config.setdefault("simulation_steps", 4)
    step = SoftCoreMappingStep(pipeline)
    step.trainer = _EmptyTrainer()
    return step


@pytest.mark.parametrize("mode", ["rate", "bogus"])
def test_negative_shift_unsupported_mode_fails_loud(mode):
    step = _step(negative_value_shift=True, spiking_mode=mode)
    with pytest.raises(NotImplementedError, match="negative_value_shift"):
        step._apply_negative_value_shift_compensation(model=object())


@pytest.mark.parametrize(
    "mode", ["lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"]
)
def test_negative_shift_disabled_is_silent_noop(mode):
    step = _step(negative_value_shift=False, spiking_mode=mode)
    step._apply_negative_value_shift_compensation(model=object())


@pytest.mark.parametrize(
    "mode", ["lif", "ttfs", "ttfs_quantized", "ttfs_cycle_based"]
)
def test_negative_shift_supported_modes_pass_gate(mode):
    step = _step(negative_value_shift=True, spiking_mode=mode)
    step._apply_negative_value_shift_compensation(model=object())
