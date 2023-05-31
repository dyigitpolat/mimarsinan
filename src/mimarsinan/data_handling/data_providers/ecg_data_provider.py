from mimarsinan.data_handling.data_provider import DataProvider

import torchvision.transforms as transforms
import torchvision

import torch
import numpy as np
import requests
import os

class ECG_DataProvider(DataProvider):
    def __init__(self):
        super().__init__()
        filename = self._download_data()
        f = np.load(filename)

        train_x = torch.FloatTensor(f['x_train']).reshape(-1, 1, 180, 1)
        train_y = torch.LongTensor(f['y_train'])

        test_x = torch.FloatTensor(f['x_test']).reshape(-1, 1, 180, 1)
        test_y = torch.LongTensor(f['y_test'])

        class AugmentedDataset(torch.utils.data.Dataset):
            def __init__(self, x, y, transform):
                self.x = x
                self.y = y
                self.transform = transform

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                return self.transform(self.x[idx]), self.y[idx]
        
        training_dataset = AugmentedDataset(train_x, train_y, self._augmentation)
        validation_dataset = torch.utils.data.TensorDataset(train_x, train_y)

        training_validation_split = 0.99
        training_length = int(len(training_dataset) * training_validation_split)

        self.training_dataset = torch.utils.data.Subset(
            training_dataset, range(0, training_length))
        
        self.validation_dataset = torch.utils.data.Subset(
            validation_dataset, range(training_length, len(training_dataset)))
        
        self.test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
            

    def _augmentation(self, x):
        shift_amt = np.random.uniform(-0.1, 0.1)
        shift = int(shift_amt * x.shape[1])
        x = x * np.random.uniform(0.9, 1.1)
        x = torch.clamp(x, min=0, max=1)
        return torch.roll(x, shift, dims=1)

    def _download_data(self):
        filename = self.datasets_path + "/ecg/ecg_data_normalized_smoked_two_classes.npz"
        url = "https://github.com/dyigitpolat/mimarsinan/releases/download/data/ecg_data_normalized_smoked_two_classes.npz"
        
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
