import matplotlib.pyplot as plt

class HistogramVisualizer:
    def __init__(self, hist, bin_edges, min_x = -1.5, max_x = 1.5, v_line = None, device = 'cpu'):
        self.hist = hist
        self.bin_edges = bin_edges
        self.max_x = bin_edges[-1]
        self.min_x = bin_edges[0]
        self.v_line = v_line

    def plot(self, filename):
        plt.grid()
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        plt.ylim(max(0, min(self.hist) - 0.1), max(self.hist) + 0.1)

        plt.bar(self.bin_edges[:-1], self.hist, width=self.bin_edges[1] - self.bin_edges[0], align="edge")
        if self.v_line is not None:
            plt.axvline(x=self.v_line, color='r')
            
        plt.xlabel('Activation Value')
        plt.ylabel('Count')
        plt.title('Activation Histogram')

        plt.savefig(filename)
        plt.clf()
