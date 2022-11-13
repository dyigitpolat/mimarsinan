import torch
import torch.nn as nn

def calculate_patch_size(i, j, divs_h, divs_w, h, w, c):

    divs_ht = [round(v * h) for v in divs_h]
    divs_wt = [round(v * w) for v in divs_w]
    patch_h = divs_ht[i+1] - divs_ht[i]
    patch_w = divs_wt[j+1] - divs_wt[j]

    return round(patch_h * patch_w * c)

class SimpleMLPMixer(nn.Module):
    def __init__(
        self, 
        patch_dim_h, patch_dim_w,
        patch_features,
        patch_channels,
        mixer_features,
        inner_mlp_width,
        inner_mlp_count, 
        divs_w, divs_h,
        h, w, c,
        output_size):
        super(SimpleMLPMixer, self).__init__()

        self.patch_dim_h = patch_dim_h
        self.patch_dim_w = patch_dim_w
        self.patch_features = patch_features
        self.patch_channels = patch_channels
        self.divs_h = divs_h
        self.divs_w = divs_w
        self.h = h
        self.w = w
        self.c = c

        patch_count = patch_dim_h * patch_dim_w

        self.input_layer_norm = nn.LayerNorm([c,h,w])

        self.patch_mlps = nn.ModuleList([])

        for i in range(self.patch_dim_h):
            for j in range(self.patch_dim_w):
                self.patch_mlps.append(
                    nn.Sequential(
                        nn.Linear(
                            in_features=calculate_patch_size(i,j,divs_h,divs_w,h,w,c), 
                            out_features=patch_features),
                        nn.ReLU()))

        self.mixer_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                in_features=(patch_features//patch_channels)*patch_count, 
                out_features=mixer_features),
                nn.ReLU(),
                nn.Linear(
                in_features=mixer_features, 
                out_features=mixer_features),
                nn.LayerNorm([mixer_features])
            )] * patch_channels)

        self.combiner = nn.Sequential(
            nn.Linear(
            in_features=mixer_features*patch_channels,
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

        self.classifier = nn.Linear(
            in_features=inner_mlp_width, 
            out_features=output_size)
            
        
    def forward(self, x : torch.Tensor):
        x = self.input_layer_norm(x)

        patches = []
        divs_h = [round(v * self.h) for v in self.divs_h]
        divs_w = [round(v * self.w) for v in self.divs_w]
        for i in range(self.patch_dim_h):
            for j in range(self.patch_dim_w):
                patches.append(
                    self.patch_mlps[i*self.patch_dim_w + j](
                        x[:,:,
                            divs_h[i]:divs_h[i+1],
                            divs_w[j]:divs_w[j+1]].reshape(x.size(0), -1)))
        
        mixers = []
        mixer_channel_size = self.patch_features // self.patch_channels
        for i in range(self.patch_channels):
            begin = i*mixer_channel_size
            end = (i + 1)*mixer_channel_size

            mixers.append(self.mixer_mlps[i](
                torch.cat(
                    [p[:,begin:end] for p in patches], -1)))

        y = self.combiner(torch.cat(mixers, -1))

        for mlp in self.inner_mlps:
            y = mlp(y)
        
        return self.classifier(y)
        


        
        
            
        

        


                

        
        
