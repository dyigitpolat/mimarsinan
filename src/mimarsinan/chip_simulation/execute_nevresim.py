import subprocess

def execute_simulator(simulator_filename):
    cmd = ["./{}".format(simulator_filename)]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    while(True):
        line = p.stdout.readline()
        print(line.decode("utf-8"))
        if not line:
            break

    return p.wait()