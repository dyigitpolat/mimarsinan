import numpy as np

import os

def prepare_containing_directory(filename):
    os.makedirs(filename[:-filename[::-1].find('/')-1], exist_ok=True) 

def input_to_file(
    input, target, filename:str):

    values = input.tolist()
    parts = [str(target), '1', str(len(values))]
    parts.extend(str(x_i) for x_i in values)
    with open(filename, "w") as f:
        f.write(' '.join(parts) + ' ')


def save_inputs_to_files(generated_files_path, loader, input_count):
    print("Saving input data to files...")
    
    input_files_path = "{}/inputs/".format(generated_files_path)
    prepare_containing_directory(input_files_path)

    for batch_idx, (x, y) in enumerate(loader):
        if(batch_idx >= input_count): break
        input_to_file(
            x.flatten(), np.argmax(y.tolist()), 
            "{}{}.txt".format(input_files_path, batch_idx))
        

def save_weights_and_chip_code(chip, generated_files_path, verbose=True):
    if verbose:
        print("Saving trained weights and chip generation code...")

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