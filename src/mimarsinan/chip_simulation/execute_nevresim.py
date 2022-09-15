import subprocess

def execute_simulator(simulator_filename):
    cmd = ["./{}".format(simulator_filename)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    
    line = p.stdout.readline()
    output_values = [int(val) for val in line.split()]
    p.wait()

    return output_values