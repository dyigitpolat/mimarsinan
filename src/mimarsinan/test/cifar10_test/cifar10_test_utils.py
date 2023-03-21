from mimarsinan.models.core_flow import *
from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.search.patch_borders import *

import torch.nn as nn

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

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

def train_on_cifar10(ann, device, epochs):
    train_loader, _ = get_cifar10_data(500)
    optimizer = torch.optim.Adam(ann.parameters(), lr = 0.001, weight_decay=0.00005)
    
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
        _, test_loader = get_cifar10_data(4096)
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


def quantize_weight_tensor(weight_tensor, bits):
    q_min = -( 2 ** (bits - 1) )
    q_max = ( 2 ** (bits - 1) ) - 1

    max_weight = weight_tensor.max().item()
    min_weight = weight_tensor.max().item()

    return torch.where(
        weight_tensor > 0,
        torch.round(((q_max) * (weight_tensor)) / (max_weight)) / (q_max / max_weight),
        torch.round(((q_min) * (weight_tensor)) / (min_weight)) / (q_min / min_weight))

def quantize_model(ann, bits):
    assert isinstance(ann, CoreFlow)
    for core_param in ann.core_params:
        core_param.data = quantize_weight_tensor(core_param.data, bits)

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
    b = 0
    for (x, y) in train_loader:
        print(100* b / len(train_loader))
        b += 1
        update_quantized_model(ann, qnn)
        optimizer.zero_grad()
        ann.train()
        qnn.train()
        loss = nn.CrossEntropyLoss()(qnn(x), y)
        loss.backward()
        print(loss)
        transfer_gradients(ann, qnn)
        optimizer.step()

import copy 
def train_on_cifar10_quantized(ann, device, epochs, lr=0.001, weight_decay=0.00005):
    qnn = copy.deepcopy(ann)

    train_loader, _ = get_cifar10_data(10000)
    optimizer = torch.optim.Adam(ann.parameters(), lr = lr)
    for epoch in range(epochs):
        train_on_cifar10_for_one_epoch_quantized(
            ann, qnn, device, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test_on_cifar10(ann, device)
            print(correct, '/', total)