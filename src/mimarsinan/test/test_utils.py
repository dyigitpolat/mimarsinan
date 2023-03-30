from mimarsinan.common.file_utils import *

import numpy as np

def input_to_file(
    input, target, filename:str):
    result = ""
    result += str(target) + ' '
    result += '1' + ' '
    result += str(len(input)) + ' '
    for x_i in input.tolist():
        result += str(x_i) + ' '

    f = open(filename, "w")
    f.write(result)
    f.close()


def save_inputs_to_files(generated_files_path, loader, input_count):
    input_files_path = "{}/inputs/".format(generated_files_path)
    prepare_containing_directory(input_files_path)

    for batch_idx, (x, y) in enumerate(loader):
        if(batch_idx >= input_count): break
        input_to_file(
            x.flatten(), np.argmax(y.tolist()), 
            "{}{}.txt".format(input_files_path, batch_idx))
        

def save_weights_and_chip_code(chip, generated_files_path):
    weight_file_path = "{}/weights/".format(generated_files_path)
    chip_file_path = "{}/chip/".format(generated_files_path)

    prepare_containing_directory(chip_file_path)
    f = open("{}generate_chip.hpp".format(chip_file_path), "w")
    f.write(chip.get_string())
    f.close()

    prepare_containing_directory(weight_file_path)
    f = open("{}chip_weights.txt".format(weight_file_path), "w")
    f.write(chip.get_weights_string())
    f.close()


def chip_output_to_predictions(chip_output, number_of_classes):
    return [ 
        np.argmax(chip_output[i:i+number_of_classes]) 
            for i in range(0, len(chip_output), number_of_classes)]

def evaluate_chip_output(
    chip_output, test_loader, number_of_classes, verbose = False):
    if verbose:
        total_spikes = sum(chip_output)
        print("Total spikes: {}".format(total_spikes))

        confusion_matrix = np.array([[0 for i in range(10)] for j in range(10)])
        for ((_, y), (p)) in zip(test_loader, chip_output_to_predictions(chip_output, number_of_classes)):
            confusion_matrix[y.item()][p] += 1
        print("Confusion matrix:")
        print(confusion_matrix)

    

    predictions = chip_output_to_predictions(chip_output, number_of_classes)

    total = 0
    correct = 0
    for ((_, y), (p)) in zip(test_loader, predictions):
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
    def update_model_weights(model_a, model_b):
        for a_param, b_param in zip(model_a.parameters(), model_b.parameters()):
            b_param.data = nn.Parameter(a_param).data

    def update_model(a, b):
        update_model_weights(a, b)
        weight_transformation(b)
    
    # d_a = d_b
    def transfer_gradients(a, b):
        for a_param, b_param in zip(a.parameters(), b.parameters()):
            a_param.grad = b_param.grad

    def train_one_epoch(model_a, model_b, optimizer, train_loader, epoch):
        print("Training epoch:", epoch)
        for (x, y) in train_loader:
            update_model(model_a, model_b)
            optimizer.zero_grad()
            model_a.train()
            model_b.train()
            nn.CrossEntropyLoss()(model_b(x), y).backward()
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

    model_b = copy.deepcopy(model)

    train_loader = train_dataloader
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    for epoch in range(epochs):
        train_one_epoch(
            model, model_b, optimizer, train_loader, epoch)

        if(epoch % max(epochs // 10, 1) == 0):
            correct, total = test(model_b, device, test_dataloader)
            print(correct, '/', total)