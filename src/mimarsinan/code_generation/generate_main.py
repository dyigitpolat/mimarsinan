from mimarsinan.common.file_utils import *
from mimarsinan.code_generation.main_cpp_template import *
from mimarsinan.code_generation.main_cpp_template_real_valued_exec import *

def get_config(spike_gen_mode = "Stochastic", firing_mode = "Default", weight_type = "double"):
    return {
        "spike_gen_mode": spike_gen_mode,
        "firing_mode": firing_mode,
        "weight_type": weight_type
    }

def generate_main_function(
    generated_files_path,
    input_count,
    output_count,
    simulation_length,
    cpp_code_template = main_cpp_template,
    simulation_config = get_config()):
    print("Generating main function code...")

    main_cpp_code = \
        cpp_code_template.format(
            generated_files_path, 
            input_count,
            simulation_length,
            simulation_config["spike_gen_mode"],
            simulation_config["firing_mode"],
            simulation_config["weight_type"],
            output_count)

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)

    prepare_containing_directory(main_cpp_filename)
    f = open(main_cpp_filename, "w")
    f.write(main_cpp_code)
    f.close()

def generate_main_function_for_real_valued_exec(
    generated_files_path,
    cpp_code_template = main_cpp_template_real_valued_exec):

    main_cpp_code = \
        cpp_code_template.format(generated_files_path)

    main_cpp_filename = "{}/main/main.cpp".format(generated_files_path)

    prepare_containing_directory(main_cpp_filename)
    f = open(main_cpp_filename, "w")
    f.write(main_cpp_code)
    f.close()