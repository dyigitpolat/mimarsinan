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
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b d h w -> b (h w) d'), 
            nn.ReLU()
        )

        self.num_layers = num_layers
        self.hidden_s = hidden_s
        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(self.num_patches, hidden_size, hidden_s, hidden_c, drop_p) 
            for _ in range(num_layers)
            ]
        )
        self.clf = nn.Linear(hidden_size, num_classes, bias=False)
        self.norm = Normalizer()
        self.debug = None

    def forward(self, x):
        out = self.patch_emb(x)
        out = self.mixer_layers(out)
        out = out.mean(dim=1)
        out = self.clf(out)
        out = nn.ReLU()(out)
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
        self.ln = Normalizer()
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1, bias=False)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1, bias=False)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = nn.ReLU()
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
        self.ln = Normalizer()
        self.fc1 = nn.Linear(hidden_size, hidden_c, bias=False)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size, bias=False)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.do1(out)

        out = self.fc2(out)
        out = self.ln(out)
        out = self.act(out)
        out = self.do2(out)

        return out