import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    def __init__(self, nrows=1, ncols=1, figsize=(5, 4)):
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        self.dimensionality = (nrows, ncols) if nrows > 1 or ncols > 1 else (1, 1)
        self.ax = np.array(self.ax).reshape(self.dimensionality)  # Ensure ax is always a 2D array for consistent indexing

    def draw_image(self, image, cmap = None, norm = None, title=None, xlabel=None, ylabel=None, row=0, col=0):
        if row >= self.dimensionality[0] or col >= self.dimensionality[1]:
            raise ValueError(
                f"({row},{col}) out of bounds for grid {self.dimensionality}"
            )
        
        ax = self.ax[row, col]
        ax.clear()
        ax.imshow(image, cmap=cmap, norm=norm)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        plt.draw()
        plt.pause(0.001)  # Pause to update the plot

    def save(self, filename):
        self.fig.savefig(filename)
