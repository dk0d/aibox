try:
    import torch
    from torchvision.transforms import ToPILImage, ToTensor
    from torchvision.utils import make_grid
except ImportError:
    print("pytorch required for these utilities")
    exit(1)

from typing import Tuple, TypeGuard

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from skimage.util import compare_images

from ..utils import is_list_of
from .transforms import ToNumpyImage


def is_image_list(images: list):
    # return all(isinstance(image, PILImage.Image) for image in images)
    return is_list_of(images, PILImage.Image)


def is_tensor_list(images: list) -> TypeGuard[list[torch.Tensor]]:
    return all(isinstance(image, PILImage.Image) for image in images)


def display_images(
    images: list[PILImage.Image] | list[torch.Tensor] | torch.Tensor,
    n_columns=1,
    figsize=(12, 12),
    normalize=False,
):
    if isinstance(images, (list, tuple)):
        if is_image_list(images):
            tensors = [ToTensor()(s) for s in images]
        elif is_tensor_list(images):
            tensors = images
        else:
            raise ValueError("images must be a list of PIL Images or Tensors")
        tensors = torch.cat(tensors, dim=0)
    else:
        tensors = images
    image = ToPILImage()(make_grid(tensors, nrow=n_columns, padding=1, normalize=normalize))
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def n_channels_to_pil_mode(n_channels: int | str) -> str:
    """map number of image channels to PIL Image mode

    supports 'mask', 'gray', 1, 3, 4

    PIL Modes = ["1", "CMYK", "F", "HSV", "I", "L", "LAB", "P", "RGB", "RGBA", "RGBX", "YCbCr"]

    Returns:
        "RGB" if not one of above
    """
    match n_channels:
        case "mask":
            return "1"
        case 1, "gray":
            return "L"
        case 3:
            return "RGB"
        case 4:
            return "RGBA"
    return "RGB"


def tensor_to_rgb(x):
    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)


def show_tensor_image(image: torch.Tensor):
    reverseTransforms = ToNumpyImage()
    if len(image.shape) == 4:  # if batched image, show the first one
        image = image[0, :, :, :]
    plt.imshow(reverseTransforms(image))


def image_diff(image1: torch.Tensor | np.ndarray, image2: torch.Tensor | np.ndarray, return_tensor=True):
    to_numpy = ToNumpyImage()
    i1 = to_numpy(image1) if isinstance(image1, torch.Tensor) else image1
    i2 = to_numpy(image2) if isinstance(image2, torch.Tensor) else image2
    comp = compare_images(i1, i2, method="diff")
    if return_tensor:
        return ToTensor()(comp)
    return comp


def calc_shape_2d_conv(
    H_in,
    W_in,
    stride=(2, 2),
    padding=(1, 1),
    dilation=(1, 1),
    kernel_size=(3, 3),
):
    H_out = np.floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (stride[0]) + 1)
    W_out = np.floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (stride[1]) + 1)
    return H_out, W_out


def calc_shape_2d_transpose(
    H_in,
    W_in,
    stride=(2, 2),
    padding=(1, 1),
    dilation=(1, 1),
    kernel_size=(3, 3),
    output_padding=(1, 1),
):
    H_out = (H_in - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[0] + 1
    W_out = (W_in - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[1] + 1
    return H_out, W_out


def calc_ae_conv_2d(hIn, wIn, numLayers):
    shape = (hIn, wIn)
    print(0, shape)
    for i in range(1, numLayers + 1):
        shape = calc_shape_2d_conv(*shape)
        print(i, shape)
    return shape


def calc_ae_conv_transpose_2d(hIn, wIn, numLayers):
    shape = (hIn, wIn)
    for i in range(numLayers, 0, -1):
        shape = calc_shape_2d_transpose(*shape)
        print(i - 1, shape)
    return shape


def calc_conv_shapes(imageShape: Tuple):
    calc_ae_conv_transpose_2d(*calc_ae_conv_2d(*imageShape, 5), numLayers=5)
