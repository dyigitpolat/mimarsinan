import torch

class ClassificationMode:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def mode(self):
        return "classification"
    
class RegressionMode:
    def mode(self):
        return "regression"
    

class DataProvider:
    def __init__(self, datasets_path):
        self.datasets_path = datasets_path

        self._input_shape = None
        self._output_shape = None

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