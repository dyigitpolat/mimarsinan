import subprocess
from mimarsinan.common.file_utils import *
from mimarsinan.common.build_utils import find_cpp20_compiler

def compile_simulator(generated_files_path, nevresim_path):
    print("Compiling nevresim for mapped chip...")
    
    cc_command, family = find_cpp20_compiler()
    if cc_command is None:
        print("No C++20-capable compiler found.")
        return None

    print(f"  Using compiler: {cc_command} (family={family})")

    cpp_standard = "c++20"
    optimization_flag = "-O3"

    # constexpr budget – flag name differs between compilers
    step_limit = 256 * 256 * 10000
    if family == "gcc":
        step_limit_option = f"-fconstexpr-ops-limit={step_limit}"
    else:
        step_limit_option = f"-fconstexpr-steps={step_limit}"

    simulator_filename = "./bin/simulator"
    prepare_containing_directory(simulator_filename)

    cmd = [
        cc_command,
        optimization_flag,
        step_limit_option,
        "{}/main/main.cpp".format(generated_files_path),
        "-I", "{}/include".format(nevresim_path),
        "-I", "{}/chip".format(generated_files_path),
        "-std={}".format(cpp_standard),
    ]

    # Only use libc++ with modern Clang (>= 17) where it actually works.
    # Older Clang or GCC should use the default libstdc++.
    if family == "clang":
        cmd.append("-stdlib=libc++")

    cmd += ["-o", simulator_filename]

    p = subprocess.Popen(cmd)
    if( p.wait() == 0 ):
        print("Compilation outcome:", simulator_filename)
        return simulator_filename
    else:
        print("Compilation failed.")
        return None
