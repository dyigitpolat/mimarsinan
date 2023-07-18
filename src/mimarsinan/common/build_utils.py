import subprocess

def find_latest_clang_version():
    for i in range(20, 0, -1):
        try:
            version_cmd = f'clang++-{i} --version'
            subprocess.check_output(version_cmd, shell=True)
            return f'clang++-{i}'
        except subprocess.CalledProcessError:
            continue
    return None

def verify_clang_version(cc_command):
    if cc_command is None:
        print("Clang++ not found.")
        return False
    
    if int(cc_command.split('-')[-1]) < 14:
        print("Clang++ version must be at least 14.")
        return False
    
    return True