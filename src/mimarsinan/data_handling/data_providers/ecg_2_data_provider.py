from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode

import torchvision.transforms as transforms
import torchvision

import torch
import numpy as np
import requests
import os

class ECG_2_DataProvider(DataProvider):
    def __init__(self):
        super().__init__()
        f = np.load(self.datasets_path + "/ecg_chuping.npz")

        train_x = torch.FloatTensor(f['x_train']).reshape(-1, 1, 180, 1)
        train_y = torch.LongTensor(f['y_train'])
        print(f"training class distribution: {np.unique(train_y, return_counts=True)}")

        test_x = torch.FloatTensor(f['x_test']).reshape(-1, 1, 180, 1)
        test_y = torch.LongTensor(f['y_test'])
        print(f"test class distribution: {np.unique(test_y, return_counts=True)}")

        max_value = torch.max(train_x.flatten())
        min_value = torch.min(train_x.flatten())

        train_x = (train_x - min_value) / (max_value - min_value)
        test_x = (test_x - min_value) / (max_value - min_value)

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
        shift_amt = np.random.uniform(-0.3, 0.3)
        shift = int(shift_amt * x.shape[1])
        x = x * np.random.uniform(0.8, 1.3)
        x = torch.clamp(x, min=0, max=1)
        return torch.roll(x, shift, dims=1)

    def _get_training_dataset(self):
        return self.training_dataset

    def _get_validation_dataset(self):
        return self.validation_dataset

    def _get_test_dataset(self):
        return self.test_dataset
    
    def get_prediction_mode(self):
        return ClassificationMode(2)