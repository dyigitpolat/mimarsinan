import sys
sys.path.append(r'./')

from mimarsinan.models.omihub_mlp_mixer import *
from mimarsinan.models.ensemble_mlp_mixer import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.cifar100_test.cifar100_test_utils import *
from mimarsinan.model_evaluation.te_nas_utils import *

import torch
import nni

def get_number_of_mlp_mixers():
    return 2

def get_ntk(model, input_loader):
    return get_ntk_n(input_loader, [model], num_batch=1)[0]
    
def cifar100_nni_worker():
    cifar100_h = 32
    cifar100_w = 32
    cifar100_c = 3
    cifar100_output_size = 100
    
    parameters = nni.get_next_parameter()
    #parameters = {'patch_size0': 2, 'hidden_size0': 192, 'hidden_c0': 256, 'hidden_s0': 16, 'num_layers0': 32, 'patch_size1': 16, 'hidden_size1': 192, 'hidden_c1': 192, 'hidden_s1': 64, 'num_layers1': 8}
    print(parameters)
    ann_model = EnsembleMLPMixer(
        get_parameter_dict_list(parameters, get_number_of_mlp_mixers()), 
        cifar100_h, cifar100_w, cifar100_c, cifar100_output_size
    )

    print("Evaluating model...")
    train_loader, _ = get_dataloaders(Args())
    ntk = get_ntk(ann_model, train_loader)
    print(f"NTK:{ntk}")

    nni.report_final_result(ntk)

if __name__ == "__main__":
    cifar100_nni_worker()