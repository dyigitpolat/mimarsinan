from mimarsinan.test.cifar10_test.cifar10_test_utils import *

from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

from mimarsinan.models.core_flow import *
from mimarsinan.models.polat_mlp_mixer import *
from mimarsinan.mapping.polat_mlp_mixer_mapper import *

import torch
import time

def almost_equal(a, b, epsilon=0.001):
    eq1 = abs(a - b) < epsilon
    eq2 = True
    if(b != 0.0 and b != -0.0 and a != 0.0 and a != -0.0):
        eq2 = abs(1.0 - a/b) < 0.1

    return eq1 and eq2

def get_ann_model(args):
    return PolatMLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=8, 
        hidden_size=32, 
        hidden_s=192, 
        hidden_c=16, 
        num_layers=1, 
        num_classes=10, 
        drop_p=0.3).to(args.device)

def train_model(args):
    print("Training model...")
    wandb.login()

    experiment_name = f"{args.seed}_{args.model}_{args.dataset}_{args.optimizer}_{args.scheduler}"
    if args.autoaugment:
        experiment_name += "_aa"
    if args.clip_grad:
        experiment_name += f"_cg{args.clip_grad}"
    if args.off_act:
        experiment_name += f"_noact"
    if args.cutmix_prob>0.:
        experiment_name += f'_cm'
    if args.is_cls_token:
        experiment_name += f"_cls"

    ann_model = get_ann_model(args)
    
    with wandb.init(project='mlp_mixer_1000_run', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        prepare_containing_directory("../saved_models/")
        #ann_model.load_state_dict(torch.load("../saved_models/model.state_dict"))

        wandb.config.update(args)

        print("Pretraining model...")
        trainer = Trainer(ann_model, args)
        trainer.fit(train_dl, test_dl)

        print("Tuning with CQ...")
        ann_model.enable_cq(15)
        trainer = Trainer(ann_model, args, trainer.num_steps)
        trainer.fit(train_dl, test_dl)

        print("Tuning with CQ and weight quantization...")
        trainer = Trainer(ann_model, args, trainer.num_steps)
        trainer.fit_q(train_dl, test_dl)

        torch.save(ann_model.state_dict(), f"../saved_models/model.state_dict")

    return ann_model
    
def test_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    cifar10_input_shape = (3, 32, 32)

    args = Args()
    args.dataset = "c10"
    args.batch_size = 128
    args.epochs = 10
    args.device = torch.device("cuda")

    pretrain_epochs = 10
    cq_only_epochs = 1
    cq_quantize_epochs = 1
    
    #ann_model = train_model(args)
    ann_model = PolatMLPMixer(
        in_channels=3,
        img_size=32, 
        patch_size=8, 
        hidden_size=32, 
        hidden_s=32, 
        hidden_c=128, 
        num_layers=1, 
        num_classes=10, 
        drop_p=0.05)
    #ann_model.load_state_dict(torch.load("../saved_models/model.state_dict"))

    _ = polat_mlp_mixer_to_chip(ann_model, leak=0, quantize=False, weight_type=int)

    print("---")
    generated_files_path = "../generated/cifar10/"

    # print("Preparing test input...")
    # test_input = torch.randn(input_count, 3, 32, 32)
    # test_loader = [(test_input, torch.ones(1,out_size))]

    # print("Forward pass on test input...")
    # ann_model(test_input.to(device))

    print("Pretraining model...")
    train_on_cifar10(ann_model, device, pretrain_epochs)

    #ann_model.load_state_dict(torch.load("../saved_models/model_pt100.state_dict"))
    #torch.save(ann_model.state_dict(), f"../saved_models/model_pt{pretrain_epochs}.state_dict")

    print("Mapping model to soft cores...")
    model_repr = get_polat_mlp_repr(cifar10_input_shape, ann_model)
    soft_core_mapping = SoftCoreMapping()
    soft_core_mapping.map(model_repr)
    print("  Number of soft cores:", len(soft_core_mapping.cores))
    print("  Soft core mapping delay: ", ChipLatency(soft_core_mapping).calculate())

    print("Testing CoreFlow with soft cores...")
    cf = CoreFlow(cifar10_input_shape, soft_core_mapping)
    cf.set_activation(nn.LeakyReLU())
    correct, total = test_on_cifar10(cf, device)
    print(f"{correct}/{total}")

    print("Mapping soft cores to hard cores...")
    axons_per_core = 256
    neurons_per_core = 256
    hard_core_mapping = HardCoreMapping(axons_per_core, neurons_per_core)
    hard_core_mapping.map(soft_core_mapping)
    print("  Number of hard cores:", len(hard_core_mapping.cores))
    print("  Hard core mapping delay: ", ChipLatency(hard_core_mapping).calculate())

    print("Testing CoreFlow with hard cores...")
    cf = CoreFlow(cifar10_input_shape, hard_core_mapping)
    cf.set_activation(nn.LeakyReLU())
    correct, total = test_on_cifar10(cf, device)
    print(f"{correct}/{total}")

    print("Tuning model with CQ...")
    Tq = 30
    cf.set_activation(CQ_Activation(Tq))
    train_on_cifar10(cf, device, cq_only_epochs)

    print("Tuning model with CQ and weight quantization...")
    train_on_cifar10_quantized(cf, device, cq_quantize_epochs)

    print("Updating model weights...")
    cf.update_cores()

    print("Quantizing model weights...")
    quantize_cores(hard_core_mapping.cores, bits=4)

    ###### 

    print("Calculating delay for hard core mapping...")
    print(f"delay: {ChipLatency(hard_core_mapping).calculate()}")

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