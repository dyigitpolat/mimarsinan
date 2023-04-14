from mimarsinan.common.file_utils import *

import numpy as np

def chip_output_to_predictions(chip_output, number_of_classes):
    prediction_count = int(len(chip_output) / number_of_classes)
    output_array = np.array(chip_output).reshape((prediction_count, number_of_classes))

    predictions = np.zeros(prediction_count, dtype=int)
    for i in range(prediction_count):
        predictions[i] = np.argmax(output_array[i])
    return predictions

def evaluate_chip_output(
    predictions, test_loader, verbose = False):
    targets = np.array([y.item() for (_, y) in test_loader], dtype=int)

    if verbose:

        confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])
        for (y, p) in zip(targets, predictions):
            confusion_matrix[y.item()][p] += 1
        print("Confusion matrix:")
        print(confusion_matrix)

    total = 0
    correct = 0
    for (y, p) in zip(targets, predictions):
        correct += int(y.item() == p)
        total += 1
    
    return float(correct) / total

def almost_equal(a, b, epsilon=0.00001):
    eq1 = abs(a - b) < epsilon
    eq2 = True
    if((b != 0.0) and (b != -0.0) and (a != 0.0) and (a != -0.0)):
        eq2 = abs(1.0 - a/b) < 0.1

    return eq1 and eq2

import torch
import torch.nn as nn
import copy
    
def train_with_weight_trasformation(
    model, device, 
    train_dataloader, test_dataloader, 
    weight_transformation, epochs, lr):

    # b = a
    def update_model_weights(from_model, to_model):
        for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
            to_param.data[:] = from_param.data[:]

    def update_model(a, b):
        update_model_weights(a, b)
        weight_transformation(b)
    
    # d_a = d_b
    def transfer_gradients(a, b):
        for a_param, b_param in zip(a.parameters(), b.parameters()):
            if a_param.requires_grad: 
                a_param.grad = b_param.grad

    def train_one_epoch(model_a, model_b, optimizer, train_loader, epoch):
        print("    Training epoch:", epoch)
        for (x, y) in train_loader:
            update_model(model_a, model_b)
            optimizer.zero_grad()
            model_a.train()
            model_b.train()
            model_b.loss(x, y).backward()
            transfer_gradients(model_a, model_b)
            optimizer.step()

    def test(model, device, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                y.to(device)
                outputs = model.forward(x)
                _, predicted = outputs.cpu().max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())
        return correct, total

    print("    LR:", lr)

    model_b = copy.deepcopy(model)

    train_loader = train_dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        train_one_epoch(
            model, model_b, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 4) == 0):
            correct, total = test(model_b, device, test_dataloader)
            print("    Test acc:", correct, '/', total)

    update_model_weights(model_b, model)

    correct, total = test(model, device, test_dataloader)
    print("    Test acc:", correct, '/', total)

    correct, total = test(model, device, train_dataloader)
    print("    Train acc:", correct, '/', total)
    return correct / total


def train_until_target_accuracy_with_weight_transformation (
    model, device, 
    train_dataloader, test_dataloader, 
    weight_transformation, max_epochs, 
    lr, 
    target_accuracy):

    # b = a
    def update_model_weights(from_model, to_model):
        for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
            to_param.data[:] = from_param.data[:]

    def update_model(a, b):
        update_model_weights(a, b)
        weight_transformation(b)
    
    # d_a = d_b
    def transfer_gradients(a, b):
        for a_param, b_param in zip(a.parameters(), b.parameters()):
            if a_param.requires_grad: 
                a_param.grad = b_param.grad

    def train_one_epoch(model_a, model_b, optimizer, scheduler, train_loader, epoch):
        print("    Training epoch:", epoch)
        total = 0
        correct = 0
        for (x, y) in train_loader:
            update_model(model_a, model_b)
            optimizer.zero_grad()
            model_a.train()
            model_b.train()
            loss = model_b.loss(x, y)
            loss.backward()
            transfer_gradients(model_a, model_b)
            optimizer.step()
            
            _, predicted = model_b.out.max(1)
            total += float(y.size(0))
            correct += float(predicted.eq(y).sum().item())
        
        scheduler.step(correct / total)
        return correct / total

    def test(model, device, test_loader):
        total = 0
        correct = 0
        with torch.no_grad():
            for (x, y) in test_loader:
                y.to(device)
                outputs = model.forward(x)
                _, predicted = outputs.cpu().max(1)
                total += float(y.size(0))
                correct += float(predicted.eq(y).sum().item())
        return correct, total

    print("    LR:", lr)

    model_b = copy.deepcopy(model)

    train_loader = train_dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5, min_lr=lr/100, verbose=True)

    epochs = 0
    acc = 0
    while acc < target_accuracy and epochs < max_epochs:
        acc = train_one_epoch(
            model, model_b, optimizer, scheduler, train_loader, epochs)

        if(epochs % 5 == 0):
            correct, total = test(model_b, device, test_dataloader)
            print("    LR:", round(optimizer.param_groups[0]['lr'], 7))
            print("    Train acc:", acc * 100, '%') 
            print("    Test acc:", correct, '/', total)
        epochs += 1

    update_model_weights(model_b, model)

    correct, total = test(model, device, test_dataloader)
    print("    Test acc:", correct, '/', total)

    correct, total = test(model, device, train_dataloader)
    print("    Train acc:", correct, '/', total)
    return lr, correct / total