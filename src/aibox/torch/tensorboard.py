try:
    import torch
    import torch.nn as nn
    import torchvision
    from torch.utils.tensorboard._convert_np import make_np
    from torch.utils.tensorboard._utils import convert_to_HWC
    from torch.utils.tensorboard.writer import SummaryWriter
    from torchvision.io.image import ImageReadMode, decode_image
except ImportError:
    import sys

    print("pytorch required for these utilities")
    sys.exit(1)

import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import FigureBase
from PIL import Image

from aibox.torch.transforms import ToPILImage


def figure_to_tensor(figure, add_batch_dim=False) -> torch.Tensor:
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

    # Set buffer to begining
    buf.seek(0)

    # Convert PNG buffer to Tensor image
    tensor = torch.frombuffer(buf.getbuffer(), dtype=torch.uint8)
    image = decode_image(tensor, mode=ImageReadMode.RGB_ALPHA)

    if add_batch_dim:
        # Add the batch dimension
        image = image.unsqueeze(0)

    return image


def tensors_to_images(tensors: torch.Tensor, **make_grid_kwargs) -> Image.Image:
    """Return a grid of images from the tensors as a PIL image.

    Args:
        tensors: A 4D mini-batch tensor of shape (B, C, H, W).
        make_grid_kwargs: Keyword arguments to pass to torchvision.utils.make_grid.

    Returns:
        A PIL image.
    """
    grid = torchvision.utils.make_grid(tensors, **make_grid_kwargs)
    return tensor_to_image(grid)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Converts a torch tensor to a PIL image.

    Args:
        tensor: A 3D or 4D torch tensor of shape (C, H, W) or (B, C, H, W).

    Returns:
        A PIL image.
    """
    image = ToPILImage()(tensor)
    return image


def figure_to_image(figure) -> Image.Image:
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    tensor = figure_to_tensor(figure, add_batch_dim=False)
    image = tensor_to_image(tensor)
    return image


def tensor_to_figure(
    tensor: np.ndarray | torch.Tensor,
    title=None,
    figsize=(10, 10),
    cmap=None,
    **fig_kwargs,
) -> plt.FigureBase:
    image = ToPILImage()(tensor)

    # Create a figure to contain the plot.
    figure = plt.figure(figsize=figsize, **fig_kwargs)

    assert isinstance(figure, FigureBase)

    if cmap is not None:
        image = image.convert("L")

    # show image grid
    plt.imshow(image, cmap=cmap)

    # set title
    if title is not None:
        plt.title(title)

    plt.axis("off")

    return figure


def make_image_grid_figure(
    images,
    title=None,
    figsize=(10, 10),
    cmap=None,  # "afmhot",
    **grid_kwargs,
) -> plt.FigureBase:
    """
    Return a figure from the images as a matplotlib figure.
    """

    # create an image grid
    grid = torchvision.utils.make_grid(images, **grid_kwargs)
    return tensor_to_figure(grid, title, figsize=figsize, cmap=cmap)


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


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def make_confusion_matrix_image(predictions, labels, class_names):
    import sklearn

    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(labels, predictions)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names)
    cm_image = figure_to_image(figure)
    # # Log the confusion matrix as an image summary.
    # with file_writer_cm.as_default():
    #     tf.summary.image("epoch_confusion_matrix", cm_image, step=epoch)
    return cm_image


try:
    from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

    class TBLogger(TensorBoardLogger):
        @property
        def writer(self) -> SummaryWriter:
            return self.experiment

        def log_named_parameters(self, model: nn.Module, global_step, prefix=""):
            weight_histograms(self.writer, step=global_step, model=model, prefix=prefix)

except ImportError:
    print("Unable to use Pytorch Lightning TensorboardLogger ")
