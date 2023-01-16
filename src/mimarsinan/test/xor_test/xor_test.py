from mimarsinan.test.xor_test.xor_test_utils import *
from mimarsinan.test.test_utils import *
from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *

def test_xor():
    generated_files_path = "../generated/xor/"
    simulation_length = 1000
    input_count = 4

    print("Training model...")
    xor_model = train_xor_model()

    print("Mapping trained model to chip...")
    chip = simple_mlp_to_chip(xor_model)

    print("Saving inputs to file...")
    save_inputs_to_files(
        generated_files_path, 
        zip(*get_xor_train_data()), 
        input_count)

    print("Saving trained weights and chip generation code...")
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function(generated_files_path, input_count, simulation_length)

    print("Compiling nevresim for mapped chip...")
    compiled_simulator_filename = compile_simulator(
        generated_files_path,
        "../nevresim/")
    print("Compilation outcome:", compiled_simulator_filename)


    print("Executing simulator...")
    chip_output = execute_simulator(compiled_simulator_filename)

    print("Evaluating simulator output...")
    x, y = get_xor_train_data()
    y = [np.argmax(y_i) for y_i in y]
    accuracy = evaluate_chip_output(
        chip_output, zip(x, y), 2)

    print("SNN accuracy on XOR is:", accuracy*100, "%")

    export_json_to_file(chip, generated_files_path + "chip.json")
    print("XOR test done.")

