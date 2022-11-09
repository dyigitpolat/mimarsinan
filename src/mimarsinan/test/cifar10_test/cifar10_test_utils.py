import torch.nn as nn

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

datasets_path = "../datasets"

def get_cifar10_data(batch_size=1):
    train_dataset = torchvision.datasets.CIFAR10(
        root=datasets_path, train=True, download=True,
        transform=transforms.ToTensor())

    test_set = torchvision.datasets.CIFAR10(
        root=datasets_path, train=False, download=True,
        transform=transforms.ToTensor())
                            
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def train_on_cifar10_for_one_epoch(ann, device, optimizer, train_loader, epoch):
    print("Training epoch:", epoch)
    for (x, y) in train_loader:
        optimizer.zero_grad()
        ann.train()
        y.to(device)
        outputs = ann.forward(x)
        loss = nn.CrossEntropyLoss()(outputs.cpu(), y)
        loss.backward()
        optimizer.step()

def train_on_cifar10(ann, device, epochs):
    train_loader, _ = get_cifar10_data(4096)
    optimizer = torch.optim.Adam(ann.parameters(), lr = 0.001)
    
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
