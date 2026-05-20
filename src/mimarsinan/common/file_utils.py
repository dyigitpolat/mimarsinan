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
        

def spike_train_to_file(spikes, target, filename: str, *, simulation_length: int):
    """Serialise ``(T, D)`` or ``(D, T)`` spike train cycle-major into a text file.

    Format (matches ``nevresim::SpikeTrainInputLoader``):
        target batch_size input_size simulation_length
        spike[c=0,n=0] spike[c=0,n=1] ... spike[c=0,n=D-1]
        spike[c=1,n=0] ...
    Strict cycle-major flat ordering: ``data[cycle * D + neuron]``.
    """
    arr = np.asarray(spikes, dtype=np.float64)
    T = int(simulation_length)
    if arr.ndim == 1:
        raise ValueError(
            f"spike_train_to_file: expected 2D (T,D) or (D,T) spike train, got shape {arr.shape}"
        )
    if arr.ndim != 2:
        raise ValueError(
            f"spike_train_to_file: expected 2D spike train, got ndim={arr.ndim}"
        )
    # Accept either layout: prefer T as the dim matching simulation_length.
    if arr.shape[0] == T:
        td_arr = arr  # (T, D)
    elif arr.shape[1] == T:
        td_arr = arr.T  # transpose (D, T) -> (T, D)
    else:
        raise ValueError(
            f"spike_train_to_file: spike train shape {arr.shape} has no axis matching "
            f"simulation_length={T}"
        )
    input_size = int(td_arr.shape[1])
    parts = [str(int(target)), "1", str(input_size), str(T)]
    for cycle in range(T):
        for neuron in range(input_size):
            parts.append(str(float(td_arr[cycle, neuron])))
    with open(filename, "w") as f:
        f.write(" ".join(parts) + " ")


def save_spike_train_inputs_to_files(
    generated_files_path,
    spike_trains_loader,
    input_count,
    *,
    simulation_length: int,
):
    """Save each sample's spike train + target as ``inputs/<i>.txt``.

    ``spike_trains_loader`` yields ``(spike_train, y_target)`` pairs where
    ``spike_train`` is ``(T, D)`` or ``(D, T)``.
    """
    print("Saving spike-train input data to files...")
    input_files_path = "{}/inputs/".format(generated_files_path)
    prepare_containing_directory(input_files_path)
    for batch_idx, (spikes, y) in enumerate(spike_trains_loader):
        if batch_idx >= input_count:
            break
        target = int(np.argmax(np.asarray(y).flatten()))
        spike_train_to_file(
            spikes, target,
            "{}{}.txt".format(input_files_path, batch_idx),
            simulation_length=simulation_length,
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