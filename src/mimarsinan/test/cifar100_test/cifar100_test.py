from mimarsinan.test.cifar100_test.cifar100_test_utils import *

from mimarsinan.mapping.simple_mlp_mapper import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

from mimarsinan.models.ensemble_mlp_mixer import *

import torch
import json

def test_cifar100():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    cifar100_h = 32
    cifar100_w = 32
    cifar100_c = 3
    cifar100_output_size = 100

    parameters_json = """
{
    "patch_size0": 2,
    "hidden_size0": 128,
    "hidden_c0": 256,
    "hidden_s0": 64,
    "num_layers0": 8,
    "patch_size1": 4,
    "hidden_size1": 128,
    "hidden_c1": 256,
    "hidden_s1": 64,
    "num_layers1": 8
}
    """

    print("Training model...")
    args = Args()
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
    
    for _, v in json.loads(parameters_json).items():
        experiment_name += f"_{v}"

    wandb.login()
    with wandb.init(project='mlp_mixer_test_run', config=args, name=experiment_name):
        train_dl, test_dl = get_dataloaders(args)
        ann_model = EnsembleMLPMixer(
            get_parameter_dict_list(
                json.loads(parameters_json), 2), 32, 32, 3, 100).to(args.device)
        trainer = Trainer(ann_model, args)
        trainer.fit(train_dl, test_dl)
    
    torch.save(ann_model.state_dict(), f"../saved_models/{experiment_name}")

    print("cifar100 test done.")