from mimarsinan.test.cifar10_test.cifar10_test_utils import *

from mimarsinan.models.omihub_mlp_mixer import *
from mimarsinan.chip_simulation.compile_nevresim import *
from mimarsinan.chip_simulation.execute_nevresim import *
from mimarsinan.code_generation.generate_main import *
from mimarsinan.test.test_utils import *

import torch
import json

def test_cifar10():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = 40
    ann_model = MLPMixer()

    print("Training model...")
    train_on_cifar10(ann_model, device, epochs)
    test_on_cifar10(ann_model, device, epochs)

    print("CIFAR10 test done.")