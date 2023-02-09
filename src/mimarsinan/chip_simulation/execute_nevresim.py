import subprocess

def execute_simulator(simulator_filename, input_count, num_proc = 4):
    pipes = []
    for i in range(num_proc):
        start = i * input_count // num_proc
        end = (i + 1) * input_count // num_proc
        if(i == num_proc - 1):
            end = input_count

        cmd = ["./{}".format(simulator_filename), str(start), str(end)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        pipes.append(p)

    output_values = []
    for p in pipes:
        line = p.stdout.readline()
        for val in line.split():
            output_values.append(float(val)) 
        p.wait()

    return output_values