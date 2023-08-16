from mimarsinan.data_handling.data_provider import DataProvider

import torch

class DataLoaderFactory:
    def __init__(self, data_provider_factory, num_workers=4):
        self._data_provider_factory = data_provider_factory
        self._num_workers = num_workers

        self._persistent_workers = num_workers > 0
        self._pin_memory = True

    def _get_torch_dataloader(
            self, dataset, batch_size, shuffle):
        
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            num_workers=self._num_workers, pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers)
    
    def create_data_provider(self) -> DataProvider:
        return self._data_provider_factory.create()
    
    def create_training_loader(self, batch_size, data_provider):
        return self._get_torch_dataloader(
            data_provider._get_training_dataset(),
            batch_size=batch_size, shuffle=True)
    
    def create_validation_loader(self, batch_size, data_provider):
        return self._get_torch_dataloader(
            data_provider._get_validation_dataset(),
            batch_size=batch_size, shuffle=True)
    
    def create_test_loader(self, batch_size, data_provider):
        return self._get_torch_dataloader(
            data_provider._get_test_dataset(),
            batch_size=batch_size, shuffle=False)