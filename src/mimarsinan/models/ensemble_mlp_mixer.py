from mimarsinan.models.omihub_mlp_mixer import *

import torch
import torch.nn as nn

class EnsembleMLPMixer(nn.Module):
    def __init__(self, parameter_dict_list, h, w, c, num_classes):
        super(EnsembleMLPMixer, self).__init__()
        self.models = nn.ModuleList([])

        encoding_width = 256 // len(parameter_dict_list)
        for param in parameter_dict_list:
            self.models.append(
                nn.Sequential(
                    get_custom_omihub_mlp_mixer(
                        param['patch_size'], 
                        param['hidden_size'], 
                        param['hidden_c'], 
                        param['hidden_s'], 
                        param['num_layers'], h, w, c, encoding_width),
                    nn.LayerNorm([encoding_width])))
        
        classifier_input_size = encoding_width*len(self.models)
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=classifier_input_size,
                out_features=num_classes)
            )

    def forward(self, x):
        ys = [m(x) for m in self.models]
        return self.classifier(torch.cat(ys, -1))


def get_parameter_dict_list(augmented_search_space, number_of_mlpmixers):
    parameter_dicts = []
    for i in range(number_of_mlpmixers):
        parameter_dicts.append({})
        for k in get_omihub_mlp_mixer_search_space():
            parameter_dicts[i][k] = augmented_search_space[k + str(i)]
    
    return parameter_dicts

