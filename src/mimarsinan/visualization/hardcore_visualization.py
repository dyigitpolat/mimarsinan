import matplotlib.pyplot as plt

import numpy as np
import math

class HardCoreMappingVisualizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def visualize(self, filename):
        rows = int(math.sqrt(len(self.mapping.cores)))
        cols = int(math.ceil(len(self.mapping.cores) / rows))
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 10))

        for i_y in range(rows):
            for i_x in range(cols):
                axs[i_y][i_x].axis('off')

        for i, core in enumerate(self.mapping.cores):
            i_y = i // cols
            i_x = i % cols

            axs[i_y][i_x].imshow(np.abs(core.core_matrix), cmap='YlOrRd')
            axs[i_y][i_x].set_title(f'Core {i+1}, L={core.latency}', fontsize=12)

        fig.tight_layout()

        plt.savefig(filename)

    