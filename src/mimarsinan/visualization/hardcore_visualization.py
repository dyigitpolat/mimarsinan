import matplotlib.pyplot as plt
import numpy as np
import math

# Target size in inches for the longest side of a single core
_CORE_LONG_INCH = 4.0


class HardCoreMappingVisualizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def visualize(self, filename):
        num_cores = len(self.mapping.cores)
        if num_cores == 0:
            return
        cols = int(math.ceil(math.sqrt(num_cores)))
        rows = int(math.ceil(num_cores / cols))

        cores = list(self.mapping.cores)
        max_axons = max(c.axons_per_core for c in cores)
        max_neurons = max(c.neurons_per_core for c in cores)
        M = max(max_axons, max_neurons, 1)
        scale = _CORE_LONG_INCH / M

        def core_width(c):
            return scale * c.neurons_per_core

        def core_height(c):
            return scale * c.axons_per_core

        # Per-row heights and per-row column positions so each cell has correct aspect
        row_heights = []
        for r in range(rows):
            row_h = 0.0
            for c in range(cols):
                idx = r * cols + c
                if idx < num_cores:
                    row_h = max(row_h, core_height(cores[idx]))
            row_heights.append(row_h)

        row_widths = []
        for r in range(rows):
            w = 0.0
            for c in range(cols):
                idx = r * cols + c
                if idx < num_cores:
                    w += core_width(cores[idx])
            row_widths.append(w)

        total_height = sum(row_heights)
        total_width = max(row_widths) if row_widths else 1.0

        fig, _ = plt.subplots(figsize=(total_width, total_height))
        fig.clear()

        for i, core in enumerate(cores):
            i_y = i // cols
            i_x = i % cols
            cw = core_width(core)
            ch = core_height(core)
            # Left: sum of widths of previous cells in this row
            left_in_row = sum(
                core_width(cores[i_y * cols + j])
                for j in range(i_x)
                if i_y * cols + j < num_cores
            )
            bottom = sum(row_heights[k] for k in range(i_y))
            # Normalized coordinates for add_axes
            left_n = left_in_row / total_width
            bottom_n = bottom / total_height
            w_n = cw / total_width
            h_n = ch / total_height
            ax = fig.add_axes([left_n, bottom_n, w_n, h_n])
            ax.imshow(np.abs(core.core_matrix), cmap="YlOrRd", aspect="auto")
            ax.set_title(f"Core {i+1}, L={core.latency}", fontsize=12)
            ax.axis("off")

        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()