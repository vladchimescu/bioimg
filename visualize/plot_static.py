#!/usr/env/bin python
"""
Functions and classes for static plots
"""
import matplotlib.pyplot as plt


def plot_gallery(images, titles, h, w, c, n_row=3, n_col=4):
    """Helper function to plot a gallery of image instances"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w, c)))
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
