from mimarsinan.test.cifar100_test.cifar100_test_utils import *

from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

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
        hidden_size=64, 
        hidden_s=32, 
        hidden_c=128, 
        num_layers=1, 
        num_classes=100, 
        drop_p=0.).to(args.device)

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
        trainer = Trainer(ann_model, args)
        trainer.fit(train_dl, test_dl)
        torch.save(ann_model.state_dict(), f"../saved_models/{experiment_name}.state_dict")

    return ann_model
    


def test_cifar100():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    args = Args()
    args.batch_size = 128
    args.epochs = 1
    args.device = torch.device("cuda")

    ann_model = train_model(args)
    #ann_model = get_ann_model(args)

    print("---")
    generated_files_path = "../generated/cifar100/"

    print("Forward pass against test input...")

    input_count = 1
    test_input = torch.randn(input_count, 3, 32, 32)
    out_size = 100
    test_loader = [(test_input, torch.ones(1,out_size))]
    ann_model(test_input.to(device))

    print("Mapping trained model to chip...")
    chip = polat_mlp_mixer_to_chip(ann_model, leak=0, quantize=False, weight_type=float)

    print("Saving trained weights and chip generation code...")
    save_inputs_to_files(generated_files_path, test_loader, input_count)
    save_weights_and_chip_code(chip, generated_files_path)

    print("Generating main function code...")
    generate_main_function_for_real_valued_exec(generated_files_path)

    print("Compiling nevresim for mapped chip...")
    simulator_filename = \
        compile_simulator(generated_files_path, "../nevresim/")
    print("Compilation outcome:", simulator_filename)

    print("Executing simulator...")
    start_time = time.time()
    chip_output = execute_simulator(simulator_filename, input_count, num_proc=50)
    end_time = time.time()
    print("Simulation time:", end_time - start_time)

    print("chipsum", sum(chip_output))
    tot = 0
    inc = 0
    layer_debug_tensor = ann_model.debug.detach().cpu().numpy().flatten()
    for i in range(len(chip_output)):
        if(not almost_equal(chip_output[i], layer_debug_tensor[i])):
            inc +=1
        tot += 1
    

    print(chip_output)
    print(layer_debug_tensor)
    
    print("inc", inc)
    print("tot", tot)
    print("cifar100 test done.")