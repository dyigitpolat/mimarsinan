from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver

import os

import torch

import torch.multiprocessing as mp


def configure_multiprocessing():
    """Pick a multiprocessing start method that works for the whole pipeline.

    The previous behaviour was an unconditional ``mp.set_start_method('spawn')``
    for CUDA-fork-safety on PyTorch DataLoader workers. That collides with
    Lava: ``lava.magma.runtime.message_infrastructure.multiprocessing``
    calls ``mp.set_start_method('fork')`` at import, and Lava's
    SharedMemoryManager expects ``fork`` semantics — under ``spawn`` the
    Loihi step's first ``lif.run(...)`` deadlocks the runtime
    (``SharedMemoryManager.shutdown`` was never bound).

    Default to ``fork`` (the system default on Linux) so Lava just works.
    Set ``MIMARSINAN_MP_START_METHOD=spawn`` if a CUDA-fork hazard surfaces
    in a specific workload — that's an opt-in escape hatch, not the
    project default.
    """
    method = os.environ.get("MIMARSINAN_MP_START_METHOD")
    if method is None:
        return  # leave the system default (fork on Linux)
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method(method)


def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def init():
    NevresimDriver.nevresim_path = "./nevresim/"

    force_cudnn_initialization()
    configure_multiprocessing()