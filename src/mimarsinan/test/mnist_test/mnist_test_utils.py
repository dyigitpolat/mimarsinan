from mimarsinan.models.core_flow import *
from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *

import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms

datasets_path = "../datasets"

def get_mnist_data(batch_size=1):
    train_dataset = torchvision.datasets.MNIST(
        root=datasets_path, train=True, download=True,
        transform=transforms.ToTensor())

    test_set = torchvision.datasets.MNIST(
        root=datasets_path, train=False, download=True,
        transform=transforms.ToTensor())
                            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=32)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=32)
    validation_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=32)

    return train_loader, test_loader, validation_loader

