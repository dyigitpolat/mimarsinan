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
    def __init__(self, datasets_path, *, seed: int | None = 0):
        self.datasets_path = datasets_path
        self.seed = int(seed) if seed is not None else None

        self._input_shape = None
        self._output_shape = None

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
        return self.get_training_set_size() // 100
    
    def get_validation_batch_size(self):
        return self.get_validation_set_size()
    
    def get_test_batch_size(self):
        return self.get_test_set_size()
    
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