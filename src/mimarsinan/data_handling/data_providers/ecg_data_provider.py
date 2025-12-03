from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

from torch.utils.data import Dataset

import torch
import numpy as np
import requests
import os
    
@BasicDataProviderFactory.register("ECG_DataProvider")
class ECG_DataProvider(DataProvider):
    def __init__(self, datasets_path):
        super().__init__(datasets_path)
        
        training_dataset, validation_dataset, test_dataset = self._load_data()
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

    def _load_data(self):
        filename = self._download_data()
        f = np.load(filename)

        train_x = torch.FloatTensor(f['x_train']).reshape(-1, 1, 180, 1)
        train_y = torch.LongTensor(f['y_train'])

        train_x_new = []
        train_y_new = []
        c0_count = 0
        c1_count = 0
        for idx, y in enumerate(train_y):
            if y == 1:
                c1_count += 1
                train_x_new.append(train_x[idx])
                train_y_new.append(y)
        
        for idx, y in enumerate(train_y):
            if y == 0:
                if c0_count < c1_count:
                    c0_count += 1
                    train_x_new.append(train_x[idx])
                    train_y_new.append(y)

        train_x = torch.stack(train_x_new)
        train_y = torch.stack(train_y_new)

        # shuffle for validation
        torch.random.manual_seed(42)
        shuffle_indices = torch.randperm(len(train_x))
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]

        test_x = torch.FloatTensor(f['x_test']).reshape(-1, 1, 180, 1)
        test_y = torch.LongTensor(f['y_test'])

        class AugmentedDataset(Dataset):
            def __init__(self, tensor_dataset, transform):
                self.tensor_dataset = tensor_dataset
                self.transform = transform

            def __len__(self):
                return len(self.tensor_dataset)

            def __getitem__(self, idx):
                x, y = self.tensor_dataset[idx]
                return self.transform(x, y), y

        training_dataset = AugmentedDataset(torch.utils.data.TensorDataset(train_x, train_y), ECG_DataProvider._augmentation)
        validation_dataset = torch.utils.data.TensorDataset(train_x, train_y)

        training_validation_split = 0.95
        length = len(training_dataset)
        training_length = int(len(training_dataset) * training_validation_split)

        training_dataset = torch.utils.data.Subset(
            training_dataset, range(0, training_length))
        
        validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(training_length, length))
        
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

        return training_dataset, validation_dataset, test_dataset

    @staticmethod
    def _augmentation(x, y):
        x = x.clone()

        y = y.unsqueeze(-1)
        for idx, x_i in enumerate(x):
            shift_amt = np.random.uniform(-0.1, 0.1)
            shift = int(shift_amt * x_i.shape[-2])

            x[idx] = x[idx] * np.random.uniform(0.9, 1.2)

            x[idx] = torch.clamp(x[idx], min=0, max=1)
            x[idx] = torch.roll(x[idx], shift, dims=1)

            x[idx] = x[idx] * np.random.uniform(0.9, 1.0)
        
        return x

    def _download_data(self):
        filename = self.datasets_path + "/ecg/raw_ecg_shuffled_normalized.npz"
        url = "https://github.com/dyigitpolat/mimarsinan/releases/download/intrapatient_data_normalized/raw_ecg_shuffled_normalized.npz"

        if not os.path.exists(filename):
            print("Downloading ECG data...")
            os.makedirs(self.datasets_path + "/ecg/", exist_ok=True)
            r = requests.get(url, allow_redirects=True)
            open(filename, 'wb').write(r.content)
        
        return filename

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset
    
    def get_test_batch_size(self):
        return 10000
    
    def get_prediction_mode(self):
        return ClassificationMode(2)
    
    def is_mp_safe(self):
        return False