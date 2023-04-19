from mimarsinan.models.core_flow import *
from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.search.patch_borders import *

import torch.nn as nn

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from mimarsinan.transformations.weight_quantization import *
from mimarsinan.transformations.weight_clipping import *

datasets_path = "../datasets"

def get_cifar10_data(batch_size=1):
    train_transform = transforms.Compose([
        #transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])


    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0, 0, 0], [1, 1, 1])
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=datasets_path, train=True, download=False,
        transform=train_transform)

    test_set = torchvision.datasets.CIFAR10(
        root=datasets_path, train=False, download=False,
        transform=test_transform)
                            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=32, 
        pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=32,
        pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader, validation_loader




