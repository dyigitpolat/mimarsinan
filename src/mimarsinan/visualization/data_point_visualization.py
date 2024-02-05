import matplotlib.pyplot as plt
import math

class DataPointVisualizer:
    def __init__(self, dataset_x, dataset_y, count):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.count = count

    def visualize_images(self, filename):
        fig = plt.figure(figsize=(8, 8))
        columns = int(math.sqrt(self.count))
        rows = math.ceil(self.count / columns)

        for i in range(1, self.count + 1):
            img = self.dataset_x[i-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f"Class: {self.dataset_y[i-1]}")
            plt.axis('off')

        plt.savefig(filename)
    
    def visualize_curves(self, filename):
        fig = plt.figure(figsize=(8, 8))
        columns = int(math.sqrt(self.count))
        rows = math.ceil(self.count / columns)

        for i in range(1, self.count + 1):
            img = self.dataset_x[i-1]
            fig.add_subplot(rows, columns, i)
            plt.plot(img.squeeze())
            plt.title(f"Class: {self.dataset_y[i-1]}")

        plt.savefig(filename)
        plt.close()
        