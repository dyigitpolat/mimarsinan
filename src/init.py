from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver

import torch

import torch.multiprocessing as mp

def configure_multiprocessing():
    mp.set_start_method('spawn')

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def init():
    NevresimDriver.nevresim_path = "./nevresim/"
    DataProvider.datasets_path = "./datasets/"

    force_cudnn_initialization()
    configure_multiprocessing()