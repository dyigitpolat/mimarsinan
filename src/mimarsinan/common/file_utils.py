import os

def prepare_containing_directory(filename):
    os.makedirs(filename[:-filename[::-1].find('/')-1], exist_ok=True) 