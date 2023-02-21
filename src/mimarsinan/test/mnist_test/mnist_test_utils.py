import torch.nn as nn

import torch
import torchvision
import torchvision.transforms as transforms

from mimarsinan.models.simple_mlp import *

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
        _, test_loader = get_mnist_data(5000)
        for (x, y) in test_loader:
            y.to(device)
            outputs = ann.forward(x)
            _, predicted = outputs.cpu().max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
    
    return correct, total


q_max = 7
q_min = -8

def quantize_weight_tensor(weight_tensor):
    max_weight = torch.max(weight_tensor)
    min_weight = torch.min(weight_tensor)

    return torch.where(
        weight_tensor > 0,
        torch.round(((q_max) * (weight_tensor)) / (max_weight)) / (q_max / max_weight),
        torch.round(((q_min) * (weight_tensor)) / (min_weight)) / (q_min / min_weight))

def quantize_model(ann):
    for layer in ann.layers:
        if isinstance(layer, nn.Linear):
            layer.weight = nn.Parameter(quantize_weight_tensor(layer.weight))

def update_model_weights(ann, qnn):
    for layer, q_layer in zip(ann.layers, qnn.layers):
        if isinstance(layer, nn.Linear):
            q_layer.weight = nn.Parameter(layer.weight)

def update_quantized_model(ann, qnn):
    update_model_weights(ann, qnn)
    quantize_model(qnn)

def train_on_mnist_for_one_epoch_quantized(ann, qnn, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)
    for (x, y) in train_loader:
        update_quantized_model(ann, qnn)
        optimizer.zero_grad()
        ann.train()
        y_to_one_hot = torch.nn.functional.one_hot(y, 10) * 1.0
        loss = nn.MSELoss()(ann(x), y_to_one_hot)
        loss *= nn.MSELoss()(qnn(x), y_to_one_hot) ** 5.5
        loss.backward()
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
            correct, total = test_on_mnist(ann, device)
            print(correct, '/', total)