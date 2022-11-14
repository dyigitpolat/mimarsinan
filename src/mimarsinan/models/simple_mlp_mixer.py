import torch
import torch.nn as nn

def calculate_patch_size(i, j, divs_h, divs_w, c):
    patch_h = divs_h[i+1] - divs_h[i]
    patch_w = divs_w[j+1] - divs_w[j]

    return round(patch_h * patch_w * c)

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
                        nn.Linear(
                            in_features=patch_size, 
                            out_features=self.patch_features),
                        nn.GELU(),
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
                    nn.GELU(),
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
            nn.GELU(),
            nn.Linear(
                in_features=inner_mlp_width,
                out_features=inner_mlp_width))
        
        self.inner_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    in_features=inner_mlp_width, 
                    out_features=inner_mlp_width),
                nn.GELU(),
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
        divs_h = self.divs_h
        divs_w = self.divs_w
        for i in range(self.patch_dim_h):
            for j in range(self.patch_dim_w):
                patches.append(
                    self.patch_mlps[i*self.patch_dim_w + j](
                        xnorm[:,:,
                            divs_h[i]:divs_h[i+1],
                            divs_w[j]:divs_w[j+1]].reshape(xnorm.size(0), -1)))
        
        mixers = []
        mixer_channel_size = self.features_per_patch//self.mixer_channels
        for i in range(self.mixer_channels):
            begin = i*mixer_channel_size
            end = (i + 1)*mixer_channel_size

            mixers.append(self.mixer_mlps[i](
                torch.cat([p[:,begin:end] for p in patches], -1)))
                
        y = self.combiner(torch.cat([*mixers, *patches], -1))
        y = nn.Dropout()(y)

        for mlp in self.inner_mlps:
            y = mlp(y)
        
        return self.classifier(y)
        


        
        
            
        

        


                

        
        
