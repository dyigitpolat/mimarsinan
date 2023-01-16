import torch
import torch.nn as nn
from mimarsinan.search.patch_borders import *

def calculate_patch_size(i, j, divs_h, divs_w, c):
    patch_h = divs_h[i+1] - divs_h[i]
    patch_w = divs_w[j+1] - divs_w[j]

    return round(patch_h * patch_w * c)

class GetPatch(nn.Module):
    def __init__(self, i, j, divs_h, divs_w):
        super(GetPatch, self).__init__()
        self.i_begin = divs_h[i]
        self.i_end = divs_h[i+1]
        self.j_begin = divs_w[j]
        self.j_end = divs_w[j+1]
    
    def forward(self, x):
        return x[:,:,
            self.i_begin:self.i_end,
            self.j_begin:self.j_end].reshape(x.size(0), -1)

class SimpleMLPMixer(nn.Module):
    def __init__(
        self, 
        patch_dim_h, patch_dim_w,
        features_per_patch,
        mixer_channels,
        mixer_features,
        inner_mlp_width,
        inner_mlp_count, 
        divs_w, divs_h,
        h, w, c,
        output_size):
        super(SimpleMLPMixer, self).__init__()

        self.patch_dim_h = patch_dim_h
        self.patch_dim_w = patch_dim_w
        self.mixer_channels = mixer_channels
        self.divs_h = divs_h
        self.divs_w = divs_w
        self.h = h
        self.w = w
        self.c = c

        patch_count = patch_dim_h * patch_dim_w
        self.patch_count = patch_count
        self.features_per_patch = features_per_patch
        self.patch_features = features_per_patch

        self.input_layer_norm = nn.LayerNorm([c,h,w])

        self.patch_mlps = nn.ModuleList([])
        for i in range(self.patch_dim_h):
            for j in range(self.patch_dim_w):
                patch_size = calculate_patch_size(i,j,divs_h,divs_w,c)

                self.patch_mlps.append(
                    nn.Sequential(
                        GetPatch(i,j,divs_h, divs_w),
                        nn.Linear(
                            in_features=patch_size, 
                            out_features=self.patch_features),
                        nn.ReLU(),
                        nn.Linear(
                            in_features=self.patch_features, 
                            out_features=self.patch_features)))

        self.mixer_mlps = nn.ModuleList([])
        mixer_input_size = (features_per_patch//mixer_channels)*patch_count
        for i in range(self.mixer_channels):
            self.mixer_mlps.append(
                nn.Sequential(
                    nn.LayerNorm([mixer_input_size]),
                    nn.Linear(
                        in_features=mixer_input_size, 
                        out_features=mixer_features),
                    nn.ReLU(),
                    nn.Linear(
                        in_features=mixer_features, 
                        out_features=mixer_features)))

        mixers_out_size = mixer_features*mixer_channels
        patches_out_size = self.patch_features*patch_count
        combiner_in_size = mixers_out_size + patches_out_size
        self.combiner = nn.Sequential(
            nn.LayerNorm([combiner_in_size]),
            nn.Linear(
                in_features=combiner_in_size,
                out_features=inner_mlp_width),
            nn.ReLU(),
            nn.Linear(
                in_features=inner_mlp_width,
                out_features=inner_mlp_width))
        
        self.inner_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    in_features=inner_mlp_width, 
                    out_features=inner_mlp_width),
                nn.ReLU(),
                nn.Linear(
                    in_features=inner_mlp_width, 
                    out_features=inner_mlp_width),
                nn.LayerNorm([inner_mlp_width])
            )] * inner_mlp_count)

        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=inner_mlp_width, 
                out_features=output_size)
        )
            
        
    def forward(self, x : torch.Tensor):
        xnorm = self.input_layer_norm(x)

        patches = []
        for patch_mlp in self.patch_mlps:
            patches.append(patch_mlp(xnorm))

        mixers = []
        mixer_channel_size = self.features_per_patch//self.mixer_channels
        for i, mixer in enumerate(self.mixer_mlps):
            begin = i*mixer_channel_size
            end = (i + 1)*mixer_channel_size
            mixers.append(mixer(
                torch.cat([p[:,begin:end] for p in patches], -1)))
        
        y = self.combiner(torch.cat([*mixers, *patches], -1))
        y = nn.Dropout(p = 0.1)(y)

        for mlp in self.inner_mlps:
            y = mlp(y)
        
        return self.classifier(y)
        

def get_mlp_mixer_model(parameters, h, w, c, output_size):
    region_borders_x = get_region_borders(
        int(parameters['patch_cols']), 
        float(parameters['patch_center_x']), 
        float(parameters['patch_lensing_exp_x']),
        w)

    region_borders_y = get_region_borders(
        int(parameters['patch_rows']), 
        float(parameters['patch_center_y']), 
        float(parameters['patch_lensing_exp_y']),
        h)
        
    return SimpleMLPMixer(
        int(parameters['patch_rows']), int(parameters['patch_cols']),
        int(parameters['features_per_patch']),
        int(parameters['mixer_channels']),
        int(parameters['mixer_features']),
        int(parameters['inner_mlp_width']),
        int(parameters['inner_mlp_count']),
        region_borders_x,
        region_borders_y,
        h,w,c, 
        output_size)

def get_mlpmixer_search_space():
    return {
        'patch_cols': {'_type': 'quniform', '_value': [1, 8, 1]},
        'patch_rows': {'_type': 'quniform', '_value': [1, 8, 1]},
        'features_per_patch': {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128]},
        'mixer_channels': {'_type': 'quniform', '_value': [1, 16, 1]},
        'mixer_features': {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'inner_mlp_count': {'_type': 'quniform', '_value': [1, 5, 1]},
        'inner_mlp_width': {'_type': 'choice', '_value': [
            16, 32, 48, 64, 96, 128, 192, 256]},
        'patch_center_x': {'_type': 'uniform', '_value': [-0.15, 0.15]},
        'patch_center_y': {'_type': 'uniform', '_value': [-0.15, 0.15]},
        'patch_lensing_exp_x': {'_type': 'uniform', '_value': [0.5, 2.0]},
        'patch_lensing_exp_y': {'_type': 'uniform', '_value': [0.5, 2.0]}
    }


        
        
            
        

        


                

        
        
