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
    predictions, targets, num_classes, verbose = False):
    if verbose:
        confusion_matrix = np.array([[0 for i in range(num_classes)] for j in range(num_classes)])
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