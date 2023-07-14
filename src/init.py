from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver

import torch

import importlib.util

def import_class_from_path(path_to_module, class_name):
    spec = importlib.util.spec_from_file_location(class_name, path_to_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class_ = getattr(module, class_name)
    return class_

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def init():
    force_cudnn_initialization()
    NevresimDriver.nevresim_path = "./nevresim/"
    DataProvider.datasets_path = "./datasets/"
