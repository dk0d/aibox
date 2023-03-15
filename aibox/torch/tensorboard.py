try:
    import torch.nn as nn
    import torchvision
    from torch.utils.tensorboard.writer import SummaryWriter
    from torchvision.io.image import decode_image
except ImportError:
    print("pytorch required for these utilities")
    exit(1)

import io

import matplotlib.pyplot as plt


def figure_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = decode_image(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = image.unsqueeze(0)
    return image


def make_image_grid_figure(images, title, figsize=(10, 10), **kwargs):
    """
    Return a figure from the images as a matplotlib figure.
    """

    # create an image grid
    grid = torchvision.utils.make_grid(images, **kwargs)

    # Create a figure to contain the plot.
    figure, ax = plt.subplots(1, 1, figsize=figsize)

    # set title
    ax.set_title(title)
    ax.set_axis_off()

    # show image grid
    ax.imshow(grid, cmap=plt.cm.binary)

    return figure


def weight_histograms(writer: SummaryWriter, step: int, model: nn.Module, prefix="", per_kernel=True):
    print("Visualizing model weights...")

    histParams = dict(global_step=step, bins="tensorflow")
    # Iterate over all model parameters
    for name, param in model.named_parameters(prefix=prefix):
        if per_kernel and len(param.shape) >= 4:
            # if len(param.shape) == 4: # Conv2d or kernel
            #     num_kernels, in_channels = param.shape[:2]
            # if len(param.shape) == 5: # Conv3d or kernel
            #     num_kernels, in_channels = param.shape[:2]
            num_kernels, in_channels = param.shape[:2]
            for k in range(num_kernels):
                writer.add_histogram(f"{name}.k{k}".replace(".", "/"), param[k].flatten(), **histParams)
        else:
            writer.add_histogram(name.replace(".", "/"), param.flatten(), **histParams)


try:
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

    class TBLogger(TensorBoardLogger):
        @property
        def writer(self) -> SummaryWriter:
            return self.experiment

        def log_named_parameters(self, model: nn.Module, global_step, prefix=""):
            weight_histograms(self.writer, step=global_step, model=model, prefix=prefix)

except ImportError:
    print("Unable to use Pytorch Lightning TensorboardLogger ")
