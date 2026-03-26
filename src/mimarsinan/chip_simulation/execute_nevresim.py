import os
import subprocess
import time


def execute_simulator(simulator_filename, input_count, num_proc=0):
    """Run the nevresim binary. Supports both absolute and relative paths.

    When *num_proc* is 0 (default), uses ``os.cpu_count() // 2`` workers
    (at least 1).
    """
    if num_proc <= 0:
        num_proc = max(1, (os.cpu_count() or 2) // 2)
    num_proc = min(num_proc, input_count) if input_count > 0 else 1
    print(f"Executing simulator ({num_proc} processes)...")

    start_time = time.time()

    # Use path as-is when absolute; prepend ./ for relative paths
    executable = simulator_filename if os.path.isabs(simulator_filename) else "./{}".format(simulator_filename)

    pipes = []
    for i in range(num_proc):
        start = i * input_count // num_proc
        end = (i + 1) * input_count // num_proc
        if i == num_proc - 1:
            end = input_count

        cmd = [executable, str(start), str(end)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        pipes.append(p)

    output_values = []
    for p in pipes:
        line = p.stdout.readline()
        for val in line.split():
            output_values.append(float(val)) 
        p.wait()

    end_time = time.time()
    print("  Simulation time:", end_time - start_time)

    return output_values