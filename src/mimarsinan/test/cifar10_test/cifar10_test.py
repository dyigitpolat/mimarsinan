from mimarsinan.test.cifar10_test.cifar10_test_utils import *

from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

from mimarsinan.models.core_flow import *
from mimarsinan.visualization.hardcore_visualization import *

from mimarsinan.models.polat_mlp_mixer import *
from mimarsinan.mapping.polat_mlp_mixer_mapper import *

import torch
import time

def test_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    cifar10_input_shape = (3, 32, 32)

    pretraining_epochs = 1
    cq_quantize_epochs = 1
    cf_cq_quantize_epochs = 1
    
    ann_model = PolatMLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=8, 
        hidden_size=63, 
        hidden_s=47,
        hidden_c=15, 
        num_layers=1, 
        num_classes=10, 
        drop_p=0.05)
    #ann_model.load_state_dict(torch.load("../saved_models/model.state_dict"))

    _ = polat_mlp_mixer_to_chip(ann_model, leak=0, quantize=False, weight_type=int)

    print("---")
    generated_files_path = "../generated/cifar10/"

    print("Pretraining model...")
    PolatMLPMixer.polat_activation = nn.LeakyReLU()
    train_on_cifar10(ann_model, device, pretraining_epochs, batch_size=1000, lr = 0.001)

    print("Tuning model with CQ and weight clipping...")
    Tq = 30
    PolatMLPMixer.polat_activation = CQ_Activation(Tq)
    train_on_cifar10_weight_clipped(ann_model, device, cq_quantize_epochs, batch_size=1000, lr = 0.001)
    torch.save(ann_model.state_dict(), "../generated/cifar10/_cq_quantized_state_dict.pt")

    print("Mapping model to soft cores...")
    model_repr = get_polat_mlp_repr(cifar10_input_shape, ann_model)
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)
    print("  Number of soft cores:", len(soft_core_mapping.cores))
    print("  Soft core mapping delay: ", ChipLatency(soft_core_mapping).calculate())

    # print("Testing CoreFlow with soft cores...")
    # cf = CoreFlow(cifar10_input_shape, soft_core_mapping)
    # cf.set_activation(nn.LeakyReLU())
    # correct, total = test_on_cifar10(cf, device)
    # print(f"{correct}/{total}")

    calculate_core_thresholds(soft_core_mapping.cores, 4)

    print("Mapping soft cores to hard cores...")
    axons_per_core = 256
    neurons_per_core = 256
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)
    print("  Number of hard cores:", len(hard_core_mapping.cores))
    print("  Hard core mapping delay: ", ChipLatency(hard_core_mapping).calculate())
    HardCoreMappingVisualizer(hard_core_mapping).visualize("../generated/cifar10/hard_core_mapping.png")


    # print("Testing CoreFlow with hard cores...")
    # cf = CoreFlow(cifar10_input_shape, hard_core_mapping)
    # cf.set_activation(nn.LeakyReLU())
    # correct, total = test_on_cifar10(cf, device)
    # print(f"{correct}/{total}")

    print("Tuning hard core mapping with CQ and weight quantization...")
    Tq = 30
    cf.set_activation(CQ_Activation(Tq))
    train_on_cifar10_quantized(cf, device, cf_cq_quantize_epochs, lr = 0.0001, batch_size=10000)
    torch.save(cf.state_dict(), "../generated/cifar10/cf_cq_quantized_state_dict.pt")

    print("Updating model weights...")
    cf.update_cores()
    hard_core_mapping.cores = cf.cores

    print("Quantizing model weights...")
    quantize_cores(hard_core_mapping.cores, bits=4, clipping_p=0.01)

    # final mapping
    HardCoreMappingVisualizer(hard_core_mapping).visualize("../generated/cifar10/final_hard_core_mapping.png")
    print("Final hard core mapping latency: ", ChipLatency(hard_core_mapping).calculate())

    ###### 

    print("Mapping hard cores to chip...")
    chip = hard_cores_to_chip(
        sum(cifar10_input_shape),
        hard_core_mapping, 
        axons_per_core,
        neurons_per_core, 
        leak=0,
        weight_type=int)

    print("Saving trained weights and chip generation code...")
    #save_inputs_to_files(generated_files_path, test_loader, input_count)
    save_weights_and_chip_code(chip, generated_files_path)

    input_count = 300
    out_size = 10
    _, test_loader = get_cifar10_data(1)

    print("Generating main function code...")
    generate_main_function(generated_files_path, input_count, out_size, 200, 
        main_cpp_template, get_config("Stochastic", "Novena", "int"))
    
    # print("Generating main function code...")
    # generate_main_function_for_real_valued_exec(generated_files_path)

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    start_time = time.time()
    chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)

    # tot = 0
    # inc = 0
    # layer_debug_tensor = ann_model.debug.detach().cpu().numpy().flatten()
    # for i in range(len(chip_output)):
    #     if(not almost_equal(chip_output[i], layer_debug_tensor[i])):
    #         inc +=1
    #     tot += 1
    
    # print("inc", inc)
    # print("tot", tot)

    # print(chip_output)
    # print(layer_debug_tensor)
    
    print("Evaluating simulator output...")
    accuracy = evaluate_chip_output(chip_output, test_loader, out_size, verbose=True)
    print("SNN accuracy on CIFAR-10 is:", accuracy*100, "%")
    
    export_json_to_file(chip, generated_files_path + "chip.json")
    print("cifar10 test done.")