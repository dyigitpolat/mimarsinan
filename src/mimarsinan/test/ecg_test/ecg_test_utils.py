import wfdb
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MITBIHECGDataset(Dataset):
    def __init__(self, data_dir, record_list, transform=None):
        self.data_dir = data_dir
        self.record_list = record_list
        self.transform = transform

    def __len__(self):
        return len(self.record_list)

    def __getitem__(self, idx):
        record_name = self.record_list[idx]
        signals, fields = wfdb.rdsamp(os.path.join(self.data_dir, record_name))
        ecg_data = np.array(signals).astype(np.float32)
        ecg_data = ecg_data.T

        if self.transform:
            ecg_data = self.transform(ecg_data)

        return ecg_data


def get_mit_bih_ecg_dataset(data_dir, transform=None):
    # Download the dataset from PhysioNet
    wfdb.dl_database('mitdb', data_dir)

    # Create a list of record names
    record_list = [str(100 + i) for i in range(10)]

    return MITBIHECGDataset(data_dir, record_list, transform=transform)


# Example usage
if __name__ == "__main__":
    data_dir = '../../../datasets/mit_bih_ecg_data'
    transform = ToTensor()
    dataset = get_mit_bih_ecg_dataset(data_dir, transform=transform)

    # Print the shape of the first ECG data
    print(dataset[0].shape)
