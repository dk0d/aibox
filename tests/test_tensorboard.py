import torch
from aibox.torch.tensorboard import make_image_grid_figure, figure_to_image, make_image_figure

import matplotlib.pyplot as plt


def test_make_image_grid():
    plt.close()
    images = [torch.rand((1, 128, 128)) * i for i in range(4)]
    figure = make_image_grid_figure(images, "title", nrow=2, cmap="afmhot")
    # plt.show()
    image = figure_to_image(figure)
    plt.close()
    figure = make_image_figure(image, "title")
    # plt.show()
