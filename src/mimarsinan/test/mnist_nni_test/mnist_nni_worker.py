import sys
sys.path.append(r'./')

from mimarsinan.models.simple_mlp import *
from mimarsinan.test.mnist_test.mnist_test_utils import *
from mimarsinan.test.test_utils import *

import torch
import nni

def mnist_nni_worker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mnist_input_size = 28*28
    mnist_output_size = 10

    parameters = nni.get_next_parameter()
    inner_mlp_width = int(parameters['inner_mlp_width'])
    inner_mlp_count = int(parameters['inner_mlp_count'])
    epochs = int((10*inner_mlp_count*(inner_mlp_width//64))**0.5)

    ann_model = SimpleMLP(
        inner_mlp_width, 
        inner_mlp_count, 
        mnist_input_size, 
        mnist_output_size)

    print("Training model...")
    train_on_mnist(ann_model, device, epochs)
    correct, total = test_on_mnist(ann_model, device)

    nni.report_final_result(correct / total)

if __name__ == "__main__":
    mnist_nni_worker()