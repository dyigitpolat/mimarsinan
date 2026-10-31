"""[U-HV] A provider whose test split counts every construction and every read, beside a validation split.

The two splits hold different values (and, where a classifier needs them, different labels), so a read of one cannot
pass for the other. ``_get_test_dataset`` is overridden on the provider itself, so the fix round's guard (which
redirects the base class's test split) cannot hide a test read from the counters, and no torchvision dataset is
built, so no test file of any real dataset is opened.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from mimarsinan.data_handling.data_provider import ClassificationMode, DataProvider

Split = Tuple[torch.Tensor, torch.Tensor]


class _CountedSplit(torch.utils.data.Dataset):
    """A dataset that counts every length query and every item read into ``events['reads']``."""

    def __init__(self, split: Split, events: Dict[str, int]):
        self._xs, self._ys = split
        self._events = events

    def __len__(self) -> int:
        self._events["reads"] += 1
        return int(self._xs.shape[0])

    def __getitem__(self, index):
        self._events["reads"] += 1
        return self._xs[index], self._ys[index]


class SplitCountingProvider(DataProvider):
    """Validation and test splits that differ; each test-split construction and read is counted."""

    def __init__(self, validation: Split, test: Split, *, num_classes: int):
        super().__init__("", seed=0)
        self._validation = torch.utils.data.TensorDataset(*validation)
        self._test = test
        self._num_classes = int(num_classes)
        self.test_events: Dict[str, int] = {"constructions": 0, "reads": 0}

    def _get_training_dataset(self):
        return self._validation

    def _get_validation_dataset(self):
        return self._validation

    def _get_test_dataset(self):
        self.test_events["constructions"] += 1
        return _CountedSplit(self._test, self.test_events)

    def get_prediction_mode(self):
        return ClassificationMode(self._num_classes)

    def get_training_batch_size(self):
        return 4


class SplitCountingFactory:
    """One provider per factory (so its counters cover every read the run made) and a count of constructions."""

    def __init__(self, validation: Split, test: Split, *, num_classes: int):
        self.validation, self.test, self._num_classes = validation, test, int(num_classes)
        self.built = 0
        self._provider: SplitCountingProvider | None = None

    def create(self) -> SplitCountingProvider:
        if self._provider is None:
            self.built += 1
            self._provider = SplitCountingProvider(self.validation, self.test, num_classes=self._num_classes)
        return self._provider

    def test_events(self) -> Dict[str, int]:
        """The test split's constructions and reads so far (zero for a factory whose provider was never built)."""
        return dict(self._provider.test_events) if self._provider is not None else {"constructions": 0, "reads": 0}


def ramp_splits(count: int = 6, width: int = 2) -> Tuple[Split, Split]:
    """Validation values 0, 1, 2, ... and test values -1, -2, ..., labels 0 and 1 by split, so either read is plain."""
    base = torch.arange(count * width, dtype=torch.float32).reshape(count, width)
    validation = (base, torch.zeros(count, dtype=torch.long))
    test = (-(base + 1.0), torch.ones(count, dtype=torch.long))
    return validation, test


def ramp_factory(count: int = 6, width: int = 2) -> SplitCountingFactory:
    validation, test = ramp_splits(count, width)
    return SplitCountingFactory(validation, test, num_classes=2)
