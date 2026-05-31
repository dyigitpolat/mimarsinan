from mimarsinan.data_handling.data_provider import DataProvider, ClassificationMode
from mimarsinan.data_handling.data_provider_factory import BasicDataProviderFactory

import torch
import numpy as np
import requests
import os


@BasicDataProviderFactory.register("ECG_DataProvider")
class ECG_DataProvider(DataProvider):
    DISPLAY_LABEL = "ECG (1D signal, multi-class)"

    def __init__(self, datasets_path, *, seed: int | None = 0):
        super().__init__(datasets_path, seed=seed)

        self._train_raw, self._val_raw, self._test_raw = self._load_data()

    def _load_data(self):
        filename = self._download_data()
        f = np.load(filename)

        train_x = torch.FloatTensor(f['x_train']).reshape(-1, 1, 180, 1)
        train_y = torch.LongTensor(f['y_train'])

        # Class-balance the training set.
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
            if y == 0 and c0_count < c1_count:
                c0_count += 1
                train_x_new.append(train_x[idx])
                train_y_new.append(y)
        train_x = torch.stack(train_x_new)
        train_y = torch.stack(train_y_new)

        g = torch.Generator()
        g.manual_seed(int(self.seed if self.seed is not None else 42))
        shuffle_indices = torch.randperm(len(train_x), generator=g)
        train_x = train_x[shuffle_indices]
        train_y = train_y[shuffle_indices]

        full_tds = torch.utils.data.TensorDataset(train_x, train_y)
        cut = int(len(full_tds) * 0.95)
        train_raw = torch.utils.data.Subset(full_tds, range(0, cut))
        val_raw   = torch.utils.data.Subset(full_tds, range(cut, len(full_tds)))

        test_x = torch.FloatTensor(f['x_test']).reshape(-1, 1, 180, 1)
        test_y = torch.LongTensor(f['y_test'])
        test_raw = torch.utils.data.TensorDataset(test_x, test_y)

        return train_raw, val_raw, test_raw

    @staticmethod
    def _augment(x):
        x = x.clone()
        for c in range(x.shape[0]):
            shift = int(np.random.uniform(-0.1, 0.1) * x.shape[-2])
            x[c] = x[c] * np.random.uniform(0.9, 1.2)
            x[c] = torch.clamp(x[c], min=0, max=1)
            x[c] = torch.roll(x[c], shift, dims=0)
            x[c] = x[c] * np.random.uniform(0.9, 1.0)
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

    def raw_datasets(self) -> dict:
        return {"train": self._train_raw, "val": self._val_raw, "test": self._test_raw}

    def torch_transforms(self) -> dict:
        return {
            "train": [self._augment],
            "val":   [],
            "test":  [],
        }

    # No FFCV opt-in: ECG is 1D signal data; FFCV's RGBImageField doesn't apply.

    def get_test_batch_size(self):
        return 10000

    def get_prediction_mode(self):
        return ClassificationMode(2)

    def is_mp_safe(self):
        return False
