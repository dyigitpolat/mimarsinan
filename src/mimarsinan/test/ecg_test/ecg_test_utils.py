import wfdb
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import os

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
        anns = wfdb.rdann(os.path.join(self.data_dir, record_name), 'atr')
        ecg_data = np.array(signals).astype(np.float32)
        ecg_data = ecg_data.T

        if self.transform:
            ecg_data = self.transform(ecg_data)

        return ecg_data, anns


def get_mit_bih_ecg_dataset(data_dir, transform=None):
    record_list = ['100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                     '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
                     '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
                     '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
                     '222', '223', '228', '230', '231', '232', '233', '234']
    dataset_exists = all(os.path.exists(os.path.join(data_dir, record + '.atr')) for record in record_list)

    # Download the dataset from PhysioNet if it doesn't exist
    if not dataset_exists:
        wfdb.dl_database('mitdb', data_dir)

    return MITBIHECGDataset(data_dir, record_list, transform=transform)


def plot_array(array, title=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(array)
    if title is not None:
        plt.title(title)
    #mark x = 720
    plt.axvline(x=720, color='r')
    plt.show()

# Example usage
if __name__ == "__main__":
    data_dir = '../../../datasets/mit_bih_ecg_data'
    transform = ToTensor()
    dataset = get_mit_bih_ecg_dataset(data_dir, transform=transform)

    # Print the shape of the first ECG data
    print(len(dataset))

    for x, y in dataset:
        print(x.shape)
        print(y.ann_len)
        print(y.__dict__)

        for idx, t in enumerate(y.__dict__['sample']):
            if y.__dict__['symbol'][idx] == 'f' and y.__dict__['symbol'][idx-1] == 'N':
                start = t-720
                end = t +720
                plot_array(x[0][0][start:end], title='ECG')
