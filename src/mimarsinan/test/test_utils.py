from mimarsinan.common.file_utils import *

import numpy as np
import torch

def input_to_file(
    input: torch.Tensor, target: torch.Tensor, filename:str):
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
    chip_output, test_loader, number_of_classes):

    predictions = chip_output_to_predictions(chip_output, number_of_classes)

    total = 0
    correct = 0
    for ((_, y), (p)) in zip(test_loader, predictions):
        correct += int(y.item() == p)
        total += 1
    
    return float(correct) / total