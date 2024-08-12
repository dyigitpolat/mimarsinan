import matplotlib.pyplot as plt
import numpy as np
import math

class HardCoreMappingVisualizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def visualize(self, filename):
        num_cores = len(self.mapping.cores)
        cols = int(math.ceil(math.sqrt(num_cores)))
        rows = int(math.ceil(num_cores / cols))

        # Calculate the figure size based on the number of cores
        fig_width = cols * 4  # 4 inches per core width
        fig_height = rows * 4  # 4 inches per core height
        fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_width, fig_height))

        if num_cores == 1:
            axs = np.array([[axs]])  # Convert to a 2D array of Axes objects
        elif rows == 1:
            axs = axs.reshape(1, -1)  # Ensure axs is always a 2D array

        for i, core in enumerate(self.mapping.cores):
            i_y = i // cols
            i_x = i % cols

            axs[i_y][i_x].imshow(np.abs(core.core_matrix), cmap='YlOrRd')
            axs[i_y][i_x].set_title(f'Core {i+1}, L={core.latency}', fontsize=12)
            axs[i_y][i_x].axis('off')

        # Hide any unused subplots
        for i in range(num_cores, rows * cols):
            i_y = i // cols
            i_x = i % cols
            axs[i_y][i_x].axis('off')

        fig.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()