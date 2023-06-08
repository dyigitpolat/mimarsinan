from mimarsinan.visualization.activation_function_visualization import *
from mimarsinan.data_handling.data_provider import DataProvider
from mimarsinan.chip_simulation.nevresim_driver import NevresimDriver
from mimarsinan.models.layers import *

import torch

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))

def init():
    force_cudnn_initialization()
    NevresimDriver.nevresim_path = "../nevresim/"
    DataProvider.datasets_path = "../datasets/"


    ActivationFunctionVisualizer(CQ_Activation_Soft(Tq=5, alpha = 0.1)).plot("../generated/_soft_cq_tq_5_a_0.1.png")
    ActivationFunctionVisualizer(CQ_Activation_Soft(Tq=5, alpha = 0.5)).plot("../generated/_soft_cq_tq_5_a_0.5.png")
    ActivationFunctionVisualizer(CQ_Activation_Soft(Tq=5, alpha = 1.5)).plot("../generated/_soft_cq_tq_5_a_1.5.png")
    ActivationFunctionVisualizer(CQ_Activation_Soft(Tq=5, alpha = 2.5)).plot("../generated/_soft_cq_tq_5_a_2.5.png")
    ActivationFunctionVisualizer(CQ_Activation_Soft(Tq=5, alpha = 10)).plot("../generated/_soft_cq_tq_5_a_10.png")

    ActivationFunctionVisualizer(CQ_Activation_Parametric(Tq=5, rate = 0.1)).plot("../generated/_p_cq_tq_5_a_0.1.png")
    ActivationFunctionVisualizer(CQ_Activation_Parametric(Tq=5, rate = 0.5)).plot("../generated/_p_cq_tq_5_a_0.5.png")
    ActivationFunctionVisualizer(CQ_Activation_Parametric(Tq=5, rate = 0.9)).plot("../generated/_p_cq_tq_5_a_0.9.png")
    ActivationFunctionVisualizer(CQ_Activation_Parametric(Tq=5, rate = 1.0)).plot("../generated/_p_cq_tq_5_a_1.0.png")

    ActivationFunctionVisualizer(CQ_Activation(Tq=5)).plot("../generated/_cq_tq_5.png")
    ActivationFunctionVisualizer(ClampedReLU()).plot("../generated/_clampedrelu.png")
    ActivationFunctionVisualizer(ShiftedActivation(ClampedReLU(), 1/5)).plot("../generated/_clampedrelu_shift1.0.png")
    ActivationFunctionVisualizer(ShiftedActivation(ClampedReLU(), 1/5)).plot("../generated/_clampedrelu_shift0.5.png")
    ActivationFunctionVisualizer(ShiftedActivation(ClampedReLU(), 1/5)).plot("../generated/_clampedrelu_shift0.5_2.png")
