from mimarsinan.search.basic_configuration_sampler import BasicConfigurationSampler

class MLP_Mixer_ConfigurationSampler(BasicConfigurationSampler):
    def __init__(self):
        super().__init__()

    def _get_configuration_space(self):
        return {
            "patch_n_1": [1, 2, 3, 4, 5, 6, 7, 8],
            "patch_m_1": [1, 2, 3, 4, 5, 6, 7, 8],
            "patch_c_1": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
            "fc_k_1": [1, 2, 3, 4, 5, 6, 7, 8],
            "fc_w_1": [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
            "patch_n_2": [1, 2, 3, 4, 5, 6, 7, 8],
            "patch_c_2": [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048],
            "fc_k_2": [1, 2, 3, 4, 5, 6, 7, 8],
            "fc_w_2": [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
        }
