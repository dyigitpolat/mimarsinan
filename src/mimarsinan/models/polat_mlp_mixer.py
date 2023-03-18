import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import warmup_scheduler
import numpy as np

import torchvision
import torchvision.transforms as transforms

from mimarsinan.models.layers import * 

class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        self.weight.data = weight.data
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
class SoftQuantize(nn.Module):
    def __init__(self, Tq):
        super(SoftQuantize, self).__init__()
        self.Tq = Tq
    
    def forward(self, x, alpha=4.5):
        h = 1.0 / self.Tq
        w = 1.0 / self.Tq
        a = torch.tensor(alpha)

        output = x
        output = torch.clamp(output, 0.0, 1.0)
        output = h * (
            0.5 * (1.0/torch.tanh(a/2)) * 
            torch.tanh(a * ((output/w-torch.floor(output/w))-0.5)) + 
            0.5 + torch.floor(output/w))
        
        #output = torch.clamp(output, 0.0, 1.0)
        return output
class PolatMLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256, hidden_c=2048, num_layers=8, num_classes=10, drop_p=0.):
        super(PolatMLPMixer, self).__init__()
        
        num_patches = img_size // patch_size * img_size // patch_size
        self.num_patches = num_patches
        self.img_dim_h = img_size
        self.img_dim_w = img_size
        self.img_dim_c = in_channels
        self.num_classes = num_classes

        self.patch_emb = nn.Sequential(
            #Conv2d_WS(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size, bias=True),
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size, bias=True),
            Rearrange('b d h w -> b (h w) d'), 
            nn.LeakyReLU()
        )

        self.num_layers = num_layers
        self.hidden_s = hidden_s
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(self.num_patches, hidden_size, hidden_s, hidden_c, drop_p) 
            for _ in range(num_layers)
            ]
        )
        self.clf = nn.Linear(hidden_size, num_classes, bias=True)
        self.debug = None

    def forward(self, x):
        out = self.patch_emb(x)

        out = self.mixer_layers(out)
        
        out = out.mean(dim=1)
        out = nn.LeakyReLU()(out)

        out = self.clf(out)
        out = nn.LeakyReLU()(out)

        self.debug = out

        return out

class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)

        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p):
        super(MLP1, self).__init__()
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1, bias=True)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1, bias=True)
        self.ln = nn.Identity() #nn.BatchNorm1d(num_patches)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = nn.LeakyReLU()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.do1(out)

        out = self.fc2(out)
        out = self.ln(out)
        out = self.act(out)
        out = self.do2(out)
        
        return out

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p):
        super(MLP2, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_c, bias=True)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size, bias=True)
        self.ln = WokeBatchNorm1d(hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.do1(out)

        out = self.fc2(out)
        out = self.ln(out)
        out = self.act(out)
        out = self.do2(out)

        return out