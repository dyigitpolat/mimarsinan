import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms

from mimarsinan.models.core_flow import *

datasets_path = "../datasets"

def get_mnist_data(batch_size=1):
    train_dataset = torchvision.datasets.MNIST(
        root=datasets_path, train=True, download=True,
        transform=transforms.ToTensor())

    test_set = torchvision.datasets.MNIST(
        root=datasets_path, train=False, download=True,
        transform=transforms.ToTensor())
                            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def train_on_mnist_for_one_epoch(ann, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)

    for (x, y) in train_loader:
        optimizer.zero_grad()
        ann.train()
        y.to(device)
        loss = nn.CrossEntropyLoss()(ann(x), y)
        loss.backward()
        optimizer.step()       

def train_on_mnist(ann, device, epochs, lr=0.01):
    train_loader, _ = get_mnist_data(5000)
    optimizer = torch.optim.Adam(ann.parameters(), lr = lr)
    
    for epoch in range(epochs):
        train_on_mnist_for_one_epoch(
            ann, device, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test_on_mnist(ann, device)
            print(correct, '/', total)

def test_on_mnist(ann, device):
    total = 0
    correct = 0
    with torch.no_grad():
        _, test_loader = get_mnist_data(50000)
        for (x, y) in test_loader:
            y.to(device)
            outputs = ann.forward(x)
            _, predicted = outputs.cpu().max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    
    return correct, total

def avg_top(weight_tensor, p):
    q = max(1, int(p * weight_tensor.numel()))
    return torch.mean(torch.topk(weight_tensor.flatten(), q)[0])

def avg_bottom(weight_tensor, p):
    q = max(1, int(p * weight_tensor.numel()))
    return -torch.mean(torch.topk(-weight_tensor.flatten(), q)[0])

def quantize_weight_tensor(weight_tensor, bits):
    q_min = -( 2 ** (bits - 1) )
    q_max = ( 2 ** (bits - 1) ) - 1

    max_weight = avg_top(weight_tensor, 0.01).item()
    min_weight = avg_bottom(weight_tensor, 0.01).item()

    neg_scale = 1.0
    if abs(min_weight) > 0: neg_scale = abs(q_max/min_weight)
    pos_scale = 1.0
    if abs(max_weight) > 0: pos_scale = abs(q_max/max_weight)

    scale = min(neg_scale, pos_scale)
    clipped_weights = torch.clamp(weight_tensor, min_weight, max_weight)

    return torch.round(clipped_weights * scale) / scale

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

def train_on_mnist_for_one_epoch_quantized(ann, qnn, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)
    for (x, y) in train_loader:
        update_quantized_model(ann, qnn)
        optimizer.zero_grad()
        ann.train()
        qnn.train()
        nn.CrossEntropyLoss()(qnn(x), y).backward()
        transfer_gradients(ann, qnn)
        optimizer.step()

import copy 
def train_on_mnist_quantized(ann, device, epochs, lr=0.001):
    qnn = copy.deepcopy(ann)

    train_loader, _ = get_mnist_data(10000)
    optimizer = torch.optim.Adam(ann.parameters(), lr = lr)
    for epoch in range(epochs):
        train_on_mnist_for_one_epoch_quantized(
            ann, qnn, device, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test_on_mnist(qnn, device)
            print(correct, '/', total)