from mimarsinan.models.core_flow import *
from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.search.patch_borders import *

import torch.nn as nn

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from mimarsinan.mapping.weight_quantization import *

datasets_path = "../datasets"

def get_cifar10_data(batch_size=1):
    train_transform = transforms.Compose([
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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

    return train_loader, test_loader

def train_on_cifar10_for_one_epoch(ann, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)
    for (x, y) in train_loader:
        optimizer.zero_grad()
        ann.train()
        y.to(device)
        outputs = ann.forward(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
        optimizer.step()

def train_on_cifar10(ann, device, epochs, lr = 0.001, batch_size = 5000):
    train_loader, _ = get_cifar10_data(batch_size)
    optimizer = torch.optim.Adam(ann.parameters(), lr = lr, weight_decay=0.00005)
    
    for epoch in range(epochs):
        train_on_cifar10_for_one_epoch(
            ann, device, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test_on_cifar10(ann, device)
            print(correct, '/', total)

def test_on_cifar10(ann, device):
    total = 0
    correct = 0
    with torch.no_grad():
        _, test_loader = get_cifar10_data(10000)
        for (x, y) in test_loader:
            y.to(device)
            outputs = ann.forward(x)
            _, predicted = outputs.cpu().max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    
    return correct, total

def get_mlp_mixer_model(parameters):
    cifar10_h = 32
    cifar10_w = 32
    cifar10_c = 3
    cifar10_output_size = 10

    region_borders_x = get_region_borders(
        int(parameters['patch_cols']), 
        int(parameters['patch_center_x']), 
        int(parameters['patch_lensing_exp_x']),
        cifar10_w)

    region_borders_y = get_region_borders(
        int(parameters['patch_rows']), 
        int(parameters['patch_center_y']), 
        int(parameters['patch_lensing_exp_y']),
        cifar10_h)
        
    return SimpleMLPMixer(
        int(parameters['patch_rows']), int(parameters['patch_cols']),
        int(parameters['features_per_patch']),
        int(parameters['mixer_channels']),
        int(parameters['mixer_features']),
        int(parameters['inner_mlp_width']),
        int(parameters['inner_mlp_count']),
        region_borders_x,
        region_borders_y,
        cifar10_h,cifar10_w,cifar10_c, 
        cifar10_output_size)

def quantize_model(ann, bits):
    quantizer = TensorQuantization(bits, clipping_p=0.01)
    for param in ann.parameters():
        param.data = quantizer.quantize_tensor(param.data)

def update_model_weights(ann, qnn):
    for param, q_param in zip(ann.parameters(), qnn.parameters()):
        q_param.data = nn.Parameter(param).data

def update_quantized_model(ann, qnn):
    update_model_weights(ann, qnn)
    quantize_model(qnn, bits=4)

def transfer_gradients(a, b):
    for a_param, b_param in zip(a.parameters(), b.parameters()):
        a_param.grad = b_param.grad

def train_on_cifar10_for_one_epoch_quantized(ann, qnn, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)
    for (x, y) in train_loader:
        update_quantized_model(ann, qnn)
        optimizer.zero_grad()
        ann.train()
        qnn.train()
        loss = nn.CrossEntropyLoss()(qnn(x), y)
        loss.backward()
        transfer_gradients(ann, qnn)
        optimizer.step()

import copy 
def train_on_cifar10_quantized(ann, device, epochs, lr=0.001, weight_decay=0.00005, batch_size = 10000):
    qnn = copy.deepcopy(ann)

    train_loader, _ = get_cifar10_data(batch_size)
    optimizer = torch.optim.Adam(ann.parameters(), lr = lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        train_on_cifar10_for_one_epoch_quantized(
            ann, qnn, device, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test_on_cifar10(qnn, device)
            print(correct, '/', total)


def clip_model_weights(ann, bits):
    quantizer = TensorQuantization(bits)
    quantizer.p = 0.01

    for param in ann.parameters():
        param.data = quantizer.get_clipped_weights(param.data)

def update_clipped_model(ann, qnn):
    update_model_weights(ann, qnn)
    clip_model_weights(qnn, bits=4)

def train_on_cifar10_for_one_epoch_weight_clipped(ann, qnn, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)
    for (x, y) in train_loader:
        update_clipped_model(ann, qnn)
        optimizer.zero_grad()
        ann.train()
        qnn.train()
        loss = nn.CrossEntropyLoss()(qnn(x), y)
        loss.backward()
        transfer_gradients(ann, qnn)
        optimizer.step()

import copy 
def train_on_cifar10_weight_clipped(ann, device, epochs, lr=0.001, weight_decay=0.00005, batch_size = 10000):
    qnn = copy.deepcopy(ann)

    train_loader, _ = get_cifar10_data(batch_size)
    optimizer = torch.optim.Adam(ann.parameters(), lr = lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        train_on_cifar10_for_one_epoch_weight_clipped(
            ann, qnn, device, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test_on_cifar10(qnn, device)
            print(correct, '/', total)