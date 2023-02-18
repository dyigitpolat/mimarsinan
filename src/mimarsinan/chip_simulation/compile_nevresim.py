import subprocess
from mimarsinan.common.file_utils import *

cc_command = "clang++-14"
cpp_standard = "c++20"
optimization_flag = "-O3"

def compile_simulator(generated_files_path, nevresim_path):
    simulator_filename = "../bin/simulator"
    prepare_containing_directory(simulator_filename)

    cmd = [
        cc_command, 
        optimization_flag, 
        "{}/main/main.cpp".format(generated_files_path), 
        "-I", "{}/include".format(nevresim_path), 
        "-I", "{}/chip".format(generated_files_path), 
        "-std={}".format(cpp_standard), 
        "-o", simulator_filename]

    p = subprocess.Popen(cmd)
    if( p.wait() == 0 ):
        return simulator_filename
    else:
        return "COMPILATION FAILED"
