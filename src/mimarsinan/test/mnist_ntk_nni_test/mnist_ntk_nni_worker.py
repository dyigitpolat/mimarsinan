import sys
sys.path.append(r'./')

from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.test.test_utils import *
from mimarsinan.model_evaluation.te_nas_utils import *

import torch
import nni

def get_ntk(model, input_loader):
    return get_ntk_n(input_loader, [model], num_batch=1)[0]


def mnist_ntk_nni_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    mnist_input_size = 28*28
    mnist_output_size = 10

    parameters = nni.get_next_parameter()
    inner_mlp_width = int(parameters['inner_mlp_width'])
    inner_mlp_count = int(parameters['inner_mlp_count'])

    ann_model = SimpleMLP(
        inner_mlp_width, 
        inner_mlp_count, 
        mnist_input_size, 
        mnist_output_size)

    train_loader, _ = get_mnist_data(200)
    ntk = get_ntk(ann_model, train_loader)
    print(f"NTK:{ntk}")

    nni.report_final_result(ntk)

if __name__ == "__main__":
    mnist_ntk_nni_worker()