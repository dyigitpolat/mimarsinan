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


def spike_train_to_file(
    spikes,
    target,
    input_size: int,
    simulation_length: int,
    filename: str,
) -> None:
    """Write a flattened cycle-major spike train for ``SpikeTrain`` nevresim runs."""
    flat = np.asarray(spikes, dtype=np.float64).reshape(-1)
    expected = int(input_size) * int(simulation_length)
    if flat.size != expected:
        raise ValueError(
            f"spike train size {flat.size} != input_size * T ({expected})"
        )
    parts = [str(target), "1", str(input_size), str(simulation_length)]
    parts.extend(str(float(x)) for x in flat.tolist())
    with open(filename, "w") as f:
        f.write(" ".join(parts) + " ")


def save_inputs_to_files(generated_files_path, loader, input_count):
    print("Saving input data to files...")
    
    input_files_path = "{}/inputs/".format(generated_files_path)
    prepare_containing_directory(input_files_path)

    for batch_idx, (x, y) in enumerate(loader):
        if(batch_idx >= input_count): break
        input_to_file(
            x.flatten(), np.argmax(y.tolist()), 
            "{}{}.txt".format(input_files_path, batch_idx))


def save_spike_train_inputs_to_files(
    generated_files_path,
    loader,
    input_count,
    *,
    input_size: int,
    simulation_length: int,
):
    """Save per-sample spike-train inputs for ``SpikeTrainSpikeGenerator``."""
    print("Saving spike-train input data to files...")

    input_files_path = "{}/inputs/".format(generated_files_path)
    prepare_containing_directory(input_files_path)

    for batch_idx, (spikes, y) in enumerate(loader):
        if batch_idx >= input_count:
            break
        spike_train_to_file(
            spikes,
            np.argmax(y.tolist()),
            input_size,
            simulation_length,
            "{}{}.txt".format(input_files_path, batch_idx),
        )
        

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