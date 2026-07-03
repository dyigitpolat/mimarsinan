import os

import torch
import torch.nn as nn
import torchvision.transforms as _T

from mimarsinan.data_handling.dataset_views import ApplyTransform
from mimarsinan.data_handling.preprocessing import resolve_preprocessing

FFCV_DISABLE_ENV = "MIMARSINAN_DISABLE_FFCV"


class ClassificationMode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def mode(self):
        return "classification"

    def create_loss(self):
        """Loss for this mode, callable as ``loss(model, x, y) -> torch.Tensor``."""
        from mimarsinan.model_training.training_utilities import BasicClassificationLoss
        return BasicClassificationLoss()

class RegressionMode:
    def mode(self):
        return "regression"

    def create_loss(self):
        """Default MSE regression loss (regression training is not yet generalized)."""

        class _MSELossWrapper:
            def __call__(self, model, x, y):
                return nn.MSELoss()(model(x), y)

        return _MSELossWrapper()


class DataProvider:
    SUPPORTS_PREPROCESSING = True

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        self.datasets_path = datasets_path
        self.seed = int(seed) if seed is not None else None
        self._preprocessing_spec = resolve_preprocessing(preprocessing)
        self._batch_size_override = int(batch_size) if batch_size else None

        self._input_shape = None
        self._output_shape = None

    def _wrap_with_preprocessing(self, base_transforms):
        """Wrap a raw transform list with the configured preprocessing into a Compose."""
        if self._preprocessing_spec is None:
            return _T.Compose(list(base_transforms))
        return self._preprocessing_spec.compose(base_transforms)

    def _get_split_generator(self):
        """Seeded torch generator so dataset splits are deterministic (or None if unseeded)."""
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(int(self.seed))
        return g

    def raw_datasets(self) -> dict:
        return {}

    def torch_transforms(self) -> dict:
        return {}

    def ffcv_transforms(self) -> dict:
        return {}

    def enable_ffcv(self) -> bool:
        if os.environ.get(FFCV_DISABLE_ENV) == "1":
            return False
        return bool(self.ffcv_transforms())

    def _assemble_split(self, split: str):
        raw = self.raw_datasets().get(split)
        if raw is None:
            raise NotImplementedError(
                f"{type(self).__name__}: no raw_datasets()[{split!r}] and no "
                f"override for _get_{split}_dataset()."
            )
        tf_list = self.torch_transforms().get(split)
        if tf_list is None:
            return raw
        return ApplyTransform(raw, self._wrap_with_preprocessing(tf_list))

    def _get_training_dataset(self):
        return self._assemble_split("train")

    def _get_validation_dataset(self):
        return self._assemble_split("val")

    def _get_test_dataset(self):
        return self._assemble_split("test")

    def get_prediction_mode(self):
        """Prediction mode for this provider (subclasses must override)."""
        raise NotImplementedError()

    def create_loss(self):
        """Loss function for this provider, from its prediction mode."""
        return self.get_prediction_mode().create_loss()

    def get_training_batch_size(self):
        if self._batch_size_override is not None:
            return self._batch_size_override
        return self.get_training_set_size() // 100

    def get_validation_batch_size(self):
        """Match training minibatch size; never exceed the validation set size."""
        train_bs = self.get_training_batch_size()
        n_val = self.get_validation_set_size()
        if n_val <= 0:
            return max(1, train_bs)
        return min(max(1, train_bs), n_val)

    def get_test_batch_size(self):
        """Training batch size, capped at the test-set size (a full-set batch OOMs on large inputs)."""
        train_bs = self.get_training_batch_size()
        n_test = self.get_test_set_size()
        if n_test <= 0:
            return max(1, train_bs)
        return min(max(1, train_bs), n_test)

    def get_training_set_size(self):
        return len(self._get_training_dataset())

    def get_validation_set_size(self):
        return len(self._get_validation_dataset())

    def get_test_set_size(self):
        return len(self._get_test_dataset())

    def get_input_shape(self):
        if self._input_shape is None:
            self._input_shape = self._get_test_dataset()[0][0].shape

        return self._input_shape

    def get_output_shape(self):
        return self.get_prediction_mode().num_classes

    def is_mp_safe(self):
        return True
