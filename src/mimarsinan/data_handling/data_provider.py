import torch

class ClassificationMode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def mode(self):
        return "classification"

    def create_loss(self):
        """
        Create the appropriate loss function for this prediction mode.

        The returned object must be callable as: loss(model, x, y) -> torch.Tensor
        (see BasicTrainer usage).
        """
        # Lazy import to avoid importing training code when only inspecting datasets.
        from mimarsinan.model_training.training_utilities import BasicClassificationLoss
        return BasicClassificationLoss()

class RegressionMode:
    def mode(self):
        return "regression"

    def create_loss(self):
        """
        Default regression loss.

        Note: Regression training/metrics are not yet fully generalized in BasicTrainer
        (validate/test currently assume classification accuracy). This loss is provided
        for future expansion and for evaluators that use only the loss.
        """
        import torch.nn as nn

        class _MSELossWrapper:
            def __call__(self, model, x, y):
                return nn.MSELoss()(model(x), y)

        return _MSELossWrapper()


class DataProvider:
    # Providers whose transforms hard-code the input shape (e.g. ImageNet
    # uses RandomResizedCrop(224) unconditionally) set this to False so
    # the wizard disables the resize / normalize controls for them.
    SUPPORTS_PREPROCESSING = True

    def __init__(self, datasets_path, *, seed: int | None = 0, preprocessing=None, batch_size=None):
        from mimarsinan.data_handling.preprocessing import resolve_preprocessing

        self.datasets_path = datasets_path
        self.seed = int(seed) if seed is not None else None
        self._preprocessing_spec = resolve_preprocessing(preprocessing)
        self._batch_size_override = int(batch_size) if batch_size else None

        self._input_shape = None
        self._output_shape = None

    def _wrap_with_preprocessing(self, base_transforms):
        """Wrap a raw transform list with the configured preprocessing.

        Returns a :class:`torchvision.transforms.Compose`. With no
        preprocessing configured, the list is returned unchanged inside
        a Compose.
        """
        import torchvision.transforms as _T

        if self._preprocessing_spec is None:
            return _T.Compose(list(base_transforms))
        return self._preprocessing_spec.compose(base_transforms)

    def _get_split_generator(self):
        """
        Torch generator to make dataset splits deterministic.

        Providers that use torch.utils.data.random_split should pass this generator to it.
        """
        if self.seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(int(self.seed))
        return g

    # ----- Per-split dataset / transform contract -----------------------------
    # Three method overrides cover the full data surface. Each returns a dict
    # keyed by split name ("train" / "val" / "test").
    #   raw_datasets()     - raw datasets, shared by both data paths.
    #   torch_transforms() - raw transform lists for the torch DataLoader path.
    #                        The base class wraps each list with the configured
    #                        preprocessing.
    #   ffcv_transforms()  - FFCV CPU op chains [(op_name, kwargs), ...] for
    #                        the FFCV path. Providing a non-empty dict opts
    #                        the provider into FFCV; the empty default keeps
    #                        it on the torch path regardless of the global
    #                        FFCV toggle. enable_ffcv() is derived.

    def raw_datasets(self) -> dict:
        return {}

    def torch_transforms(self) -> dict:
        return {}

    def ffcv_transforms(self) -> dict:
        return {}

    def ffcv_image_field_kwargs(self) -> dict:
        """kwargs forwarded to FFCV's ``RGBImageField`` at beton-write time.

        Typical use: ``{"max_resolution": 224}`` so the beton stores the
        image at the model-input resolution and no post-decode resize is
        needed (FFCV doesn't ship a post-decode upscale op).
        """
        return {}

    def enable_ffcv(self) -> bool:
        return bool(self.ffcv_transforms())

    def _assemble_split(self, split: str):
        from mimarsinan.data_handling.dataset_views import ApplyTransform

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
        """
        Returns the classification mode
        """
        raise NotImplementedError()

    def create_loss(self):
        """
        Convenience helper: create the loss function for this provider based on its prediction mode.
        """
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
        """Default to the training batch size, capped at the test-set size.

        The legacy default (full test set in one batch) OOMs on any non-trivial
        input (e.g. ViT-B/16 at 224x224 -> ~22 GiB for 10k CIFAR-10 test images).
        The trainer's ``test()`` method iterates the loader, so a sub-full-set
        batch size is always correct.
        """
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
