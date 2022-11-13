import sys
sys.path.append(r'./')

from mimarsinan.models.simple_mlp_mixer import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.cifar10_test.cifar10_test_utils import *
from mimarsinan.model_evaluation.te_nas_utils import *

import torch
import nni

def get_ntk(model, input_loader):
    return get_ntk_n(input_loader, [model], num_batch=1)[0]
    
def cifar10_nni_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cifar10_h = 32
    cifar10_w = 32
    cifar10_c = 3
    cifar10_output_size = 10
    
    parameters = nni.get_next_parameter()

    region_borders_x = get_region_borders(
        int(parameters['patch_cols']), 
        int(parameters['patch_center_x']), 
        int(parameters['patch_lensing_exp_x']))

    region_borders_y = get_region_borders(
        int(parameters['patch_rows']), 
        int(parameters['patch_center_y']), 
        int(parameters['patch_lensing_exp_y']))
        
    ann_model = SimpleMLPMixer(
        int(parameters['patch_rows']), int(parameters['patch_cols']),
        int(parameters['patch_features']),
        int(parameters['patch_channels']),
        int(parameters['mixer_features']),
        int(parameters['inner_mlp_width']),
        int(parameters['inner_mlp_count']),
        region_borders_x,
        region_borders_y,
        cifar10_h,cifar10_w,cifar10_c, 
        cifar10_output_size)

    print("Evaluating model...")
    train_loader, _ = get_cifar10_data(200)
    ntk = get_ntk(ann_model, train_loader)
    print(f"NTK:{ntk}")

    nni.report_final_result(ntk)

if __name__ == "__main__":
    cifar10_nni_worker()