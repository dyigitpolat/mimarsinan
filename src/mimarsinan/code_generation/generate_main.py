from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.main_cpp_template import *

def generate_main_function(
    generated_files_path,
    input_count,
    simulation_length,
    cpp_code_template = main_cpp_template):

    main_cpp_code = \
        cpp_code_template.format(
            generated_files_path, 
            input_count,
            simulation_length)

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)

    prepare_containing_directory(main_cpp_filename)
    f = open(main_cpp_filename, "w")
    f.write(main_cpp_code)
    f.close()
