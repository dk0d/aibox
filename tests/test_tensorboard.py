import torch
from aibox.torch.tensorboard import make_image_grid_figure, figure_to_image, make_image_figure
# import matplotlib.pyplot as plt



def test_make_image_grid():
    images = [torch.rand((1, 128, 128)) * i for i in range(4)]
    figure = make_image_grid_figure(images, 'title', nrow=2)
    image = figure_to_image(figure)
    figure = make_image_figure(image, 'title')
    # plt.show()

    
