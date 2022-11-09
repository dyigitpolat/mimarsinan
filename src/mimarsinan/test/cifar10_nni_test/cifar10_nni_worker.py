import sys
sys.path.append(r'./')

from mimarsinan.models.simple_mlp import *
from mimarsinan.test.test_utils import *
from mimarsinan.test.cifar10_test.cifar10_test_utils import *

import torch
import nni

def cifar10_nni_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cifar10_input_size = 32*32*3
    cifar10_output_size = 10

    parameters = nni.get_next_parameter()
    inner_mlp_width = int(parameters['inner_mlp_width'])
    inner_mlp_count = int(parameters['inner_mlp_count'])
    epochs = int((10*inner_mlp_count*(inner_mlp_width//64))**0.5)

    ann_model = SimpleMLP(
        inner_mlp_width, 
        inner_mlp_count, 
        cifar10_input_size, 
        cifar10_output_size)

    print("Training model...")
    train_on_cifar10(ann_model, device, epochs)
    correct, total = test_on_cifar10(ann_model, device)

    nni.report_final_result(correct / total)

if __name__ == "__main__":
    cifar10_nni_worker()