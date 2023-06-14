from mimarsinan.search.basic_configuration_sampler import BasicConfigurationSampler

class MLP_Mixer_ConfigurationSampler(BasicConfigurationSampler):
    def __init__(self):
        super().__init__()

    def _get_configuration_space(self):
        return {
            "input_patch_division": [1, 2, 3, 4, 5],
            #"mixer_fc_width": [16, 32, 48, 64, 96, 128, 192, 256, 384, 512],
            "fc_count": [1, 2, 3, 4, 5]
            #"fc_width": [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
        }
