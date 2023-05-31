import torch

class DataProvider:
    datasets_path = "../datasets"
    
    def __init__(self):
        self._num_workers = 4
        self._pin_memory = True
        self._training_loaders = {}
        self._validation_loaders = {}
        self._test_loaders = {}
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
    
    def _get_torch_dataloader(
            self, dataset, batch_size, shuffle):
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=self._num_workers, pin_memory=self._pin_memory)
    
    def get_training_loader(self, batch_size):
        if batch_size not in self._training_loaders:
            self._training_loaders[batch_size] = self._get_torch_dataloader(
                self._get_training_dataset(),
                batch_size=batch_size, shuffle=True
            )
            
        return self._training_loaders[batch_size]
    
    def get_validation_loader(self, batch_size):
        if batch_size not in self._validation_loaders:
            self._validation_loaders[batch_size] = self._get_torch_dataloader(
                self._get_validation_dataset(),
                batch_size=batch_size, shuffle=True
            )
            
        return self._validation_loaders[batch_size]
    
    def get_test_loader(self, batch_size):
        if batch_size not in self._test_loaders:
            self._test_loaders[batch_size] = self._get_torch_dataloader(
                self._get_test_dataset(),
                batch_size=batch_size, shuffle=False
            )
            
        return self._test_loaders[batch_size]
    
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
            self._input_shape = next(iter(self.get_test_loader(1)))[0].shape[1:]
        
        return self._input_shape
    
    def get_output_shape(self):
        if self._output_shape is None:
            self._output_shape = next(iter(self.get_test_loader(1)))[1].shape[1:]
        
        return self._output_shape