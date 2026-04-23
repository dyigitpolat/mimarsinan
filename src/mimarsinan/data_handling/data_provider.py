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

    def _apply_preprocessing(self, base_transforms, train: bool = False):
        """Wrap a provider's native transform list with the configured preprocessing.

        Providers should call this on their ``train_transform`` / ``eval_transform``
        lists; when no preprocessing is configured the list is returned as a
        :class:`torchvision.transforms.Compose` unchanged.
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

    def _get_training_dataset(self):
        """
        Dataset: Training - Validation
        Transformation: Augmentation
        """
        raise NotImplementedError()
    
    def _get_validation_dataset(self):
        """
        Dataset: Validation
        Transformation: None
        """
        raise NotImplementedError()
    
    def _get_test_dataset(self):
        """
        Dataset: Test
        Transformation: None
        """
        raise NotImplementedError()
    
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