"""Shared visualization helpers."""
import math
import matplotlib.pyplot as plt
import numpy as np


def show_image_grid(images, titles, suptitle=None, cols=4, cell_size=3.2,
                    title_fontsize=8.5, save_path=None):
    """Display a list of images in a grid with titles.

    Args:
        images: list of numpy arrays (H, W, 3)
        titles: list of strings, one per image
        suptitle: optional overall title
        cols: number of columns
        cell_size: size of each subplot
        title_fontsize: font size for subplot titles
        save_path: if provided, save to file instead of showing
    """
    rows = math.ceil(len(images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell_size, rows * cell_size))
    axes = np.array(axes).flatten()
    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(title, fontsize=title_fontsize)
    for j in range(len(images), len(axes)):
        axes[j].axis('off')
    if suptitle:
        plt.suptitle(suptitle, fontsize=12, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
