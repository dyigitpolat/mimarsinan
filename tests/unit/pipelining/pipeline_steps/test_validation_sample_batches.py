"""SoftCoreMappingStep._validation_sample_batches: inputs-only sampling helper contract."""

from __future__ import annotations

import pytest
import torch

from conftest import MockPipeline
from mimarsinan.pipelining.pipeline_steps.mapping.soft_core_mapping_step import (
    SoftCoreMappingStep,
)


class _StubTrainer:
    def __init__(self, batches):
        self._batches = batches
        self.requested = None

    def iter_validation_batches(self, n_batches):
        self.requested = n_batches
        yield from self._batches[:n_batches]


def _step():
    return SoftCoreMappingStep(MockPipeline(config={}))


def test_returns_inputs_only_in_order():
    x0, x1 = torch.zeros(2, 3), torch.ones(2, 3)
    step = _step()
    step.trainer = _StubTrainer([(x0, torch.tensor([0, 1])), (x1, torch.tensor([1, 0]))])

    batches = step._validation_sample_batches(2)

    assert step.trainer.requested == 2
    assert len(batches) == 2
    assert batches[0] is x0 and batches[1] is x1


def test_empty_validation_cache_yields_empty_list():
    step = _step()
    step.trainer = _StubTrainer([])

    assert step._validation_sample_batches(4) == []


def test_requires_constructed_trainer():
    step = _step()

    assert step.trainer is None
    with pytest.raises(AssertionError):
        step._validation_sample_batches(1)
