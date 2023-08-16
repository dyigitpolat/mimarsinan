from typing import Any
from mimarsinan.data_handling.data_provider import DataProvider

import importlib.util

def _import_class_from_path(path_to_module, class_name):
    spec = importlib.util.spec_from_file_location(class_name, path_to_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _Class_ = getattr(module, class_name)

    return _Class_

class ImportedDataProvider:
    def __init__(self, dataprovider_path, dataprovider_class_name, datasets_path):
        self._datasets_path = datasets_path
        self._dataprovider_class_path = dataprovider_path
        self._dataprovider_class_name = dataprovider_class_name
        self._dataprovider = _import_class_from_path(
            dataprovider_path, dataprovider_class_name)(datasets_path)

        self._num_workers = None
        self._persistent_workers = None
        self._update_state()

    def _update_state(self):
        self._num_workers = self._dataprovider._num_workers
        self._persistent_workers = self._dataprovider._persistent_workers

    def set_num_workers(self, count):
        self._dataprovider.set_num_workers(count)
        self._update_state()
    
    def get_prediction_mode(self):
        return self._dataprovider.get_prediction_mode()
    
    def get_training_loader(self, batch_size):
        return self._dataprovider.get_training_loader(batch_size)
    
    def get_validation_loader(self, batch_size):
        return self._dataprovider.get_validation_loader(batch_size)
    
    def get_test_loader(self, batch_size):
        return self._dataprovider.get_test_loader(batch_size)
    
    def get_training_batch_size(self):
        return self._dataprovider.get_training_batch_size()
    
    def get_validation_batch_size(self):
        return self._dataprovider.get_validation_batch_size()
    
    def get_test_batch_size(self):
        return self._dataprovider.get_test_batch_size()
    
    def get_training_set_size(self):
        return self._dataprovider.get_training_set_size()
    
    def get_validation_set_size(self):
        return self._dataprovider.get_validation_set_size()
    
    def get_test_set_size(self):
        return self._dataprovider.get_test_set_size()
    
    def get_input_shape(self):
        return self._dataprovider.get_input_shape()
    
    def get_output_shape(self):
        return self._dataprovider.get_output_shape()
    
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_dataprovider']
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._dataprovider = _import_class_from_path(
            state['_dataprovider_class_path'], state['_dataprovider_class_name'])(state['_datasets_path'])
        self._dataprovider.set_num_workers(self._num_workers)