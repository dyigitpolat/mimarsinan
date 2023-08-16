from mimarsinan.data_handling.data_provider import DataProvider

import importlib.util

class DataProviderFactory:
    def create(self) -> DataProvider:
        raise NotImplementedError

class ImportedDataProviderFactory(DataProviderFactory):
    def __init__(self, dataprovider_path, dataprovider_class_name, datasets_path):
        self._datasets_path = datasets_path
        self._dataprovider_class_path = dataprovider_path
        self._dataprovider_class_name = dataprovider_class_name

    def create(self) -> DataProvider:
        def _import_class_from_path(path_to_module, class_name):
            spec = importlib.util.spec_from_file_location(class_name, path_to_module)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _Class_ = getattr(module, class_name)

            return _Class_
        
        DataProvider = _import_class_from_path(self._dataprovider_class_path, self._dataprovider_class_name)
        return DataProvider(self._datasets_path)