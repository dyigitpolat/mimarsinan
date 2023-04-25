import wfdb
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import torch

import os

path = '/home/yigit/data/ecg_data_normalize_smoke_2c.npz'  
f = np.load(path)
train_x, train_y = f['x_train'], f['y_train']
test_x, test_y = f['x_test'], f['y_test']

y_test_ = test_y
X_train_ = torch.FloatTensor(train_x)
print(X_train_.shape)

X_train_ = X_train_.reshape(-1,1,180,1)
y_train_ = torch.FloatTensor(train_y)
    
X_test = torch.FloatTensor(test_x)
X_test = X_test.reshape(-1,1,180,1)
print(X_test.shape)

y_test = torch.FloatTensor(test_y)

#define a transform that shifts x within the range of 10% of its length
def shift(x):
    shift = int(0.5 * x.shape[1])
    shift = np.random.randint(-shift, shift)

    x = x * np.random.uniform(0.5, 2.5)
    random_mask = torch.rand(x.shape) > 0.5
    x[random_mask] = 0
    random_mask = torch.rand(x.shape) > 0.2
    x[random_mask] *= 2

    return torch.roll(x, shift, dims=1)

torch_dataset_train = torch.utils.data.TensorDataset(X_train_, y_train_)
torch_dataset_test = torch.utils.data.TensorDataset(X_test, y_test)

class CustomTensorDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        assert data.size(0) == targets.size(0)  # Ensure data and targets have the same length
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.data.size(0)
    
custom_train_set = CustomTensorDataset(X_train_, y_train_, transform=shift)

train_loader = torch.utils.data.DataLoader(custom_train_set,shuffle=True,batch_size=4296, num_workers=16, pin_memory=True)
test_loader = torch.utils.data.DataLoader(torch_dataset_test, shuffle=False,batch_size=49691, num_workers=16, pin_memory=True)



def plot_array(array, title=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(array)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_spectrogram(array, title=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.specgram(array, NFFT=64, noverlap=63)
    if title is not None:
        plt.title(title)
    plt.show()

def plot_2d_img_array(array, markings, title=None):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(array, cmap='gray')
    if title is not None:
        plt.title(title)
    #put red lines where markigs ==1
    # for i in range(len(markings)):
    #     if markings[i] == 0:
    #         plt.axvline(x=i, color='r', alpha=0.1)
    #     else:
    #         plt.axvline(x=i, color='b', alpha=0.1)

    plt.show()

def normalize_array(array):
    array = (array - array.mean()) / array.std()
    # return torch.where(array > 0, array ** 0.9, -((-array) ** 0.9))
    return 0.6*torch.tanh(0.5*torch.pi*array) / (0.5*torch.pi) + 0.4*array

# Example usage
if __name__ == "__main__":
    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        plot_2d_img_array(x.squeeze().numpy().T, y.numpy(), title="Raw ECG")

        break

    import torch.nn as nn

    net = nn.Sequential(

        nn.Linear(180, 4),
        nn.LeakyReLU(),

        nn.Linear(4, 4),
        nn.LeakyReLU(),
        nn.Dropout(0.5),


        nn.Linear(4, 4),
        nn.LeakyReLU(),
        nn.Dropout(0.5),


        nn.Linear(4, 2),
        nn.LeakyReLU(),
        nn.Dropout(0.5),

        nn.Linear(2, 2),
        nn.LeakyReLU(),

        nn.Softmax(dim=1)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    def test():
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                x, y = x.to('cuda'), y.to('cuda')
                out = net(x.squeeze())
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            print("Accuracy: {}".format(correct / total))

    def test_on_trainset():
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in train_loader:
                x, y = x.to('cuda'), y.to('cuda')
                out = net(x.squeeze())
                _, predicted = torch.max(out.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            print("Train Accuracy: {}".format(correct / total))

    import time

    net.to('cuda')
    for epoch in range(100):
        start_time = time.time()
        for x, y in train_loader:
            x, y = x.to('cuda'), y.to('cuda')

            optimizer.zero_grad()
            out = net(x.squeeze())
            loss = criterion(out, y.long())
            loss.backward()
            optimizer.step()
        print("Time: {}".format(time.time() - start_time))

        if epoch % 10 == 0:
            print("Epoch: {} Loss: {}".format(epoch, loss.item()))
            test()
            test_on_trainset()
