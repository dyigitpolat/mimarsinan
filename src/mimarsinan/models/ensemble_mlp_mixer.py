from mimarsinan.models.simple_mlp_mixer import *

import torch
import torch.nn as nn

class EnsembleMLPMixer(nn.Module):
    def __init__(self, parameter_dict_list, h, w, c, num_classes):
        super(EnsembleMLPMixer, self).__init__()
        self.models = nn.ModuleList([])

        encoding_width = 128
        for param in parameter_dict_list:
            self.models.append(
                nn.Sequential(
                    get_mlp_mixer_model(param, h, w, c, encoding_width),
                    nn.LayerNorm([encoding_width])))
        
        classifier_input_size = encoding_width*len(self.models)
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=classifier_input_size,
                out_features=classifier_input_size),
            nn.GELU(),
            nn.Linear(
                in_features=classifier_input_size,
                out_features=num_classes)
            )


    def forward(self, x):
        ys = [m(x) for m in self.models]
        
        return self.classifier(torch.cat(ys, -1))

