try:
    from torchvision.transforms import ToPILImage, ToTensor
    from torchvision.utils import make_grid

    import torch
except ImportError:
    import sys

    print("pytorch required for these utilities")
    sys.exit(1)

from pathlib import Path
from typing import TypeGuard

import matplotlib.pyplot as plt
import numpy as np
from aibox.torch.transforms import ToNumpyImage
from aibox.utils import is_list_of
from PIL import Image as PILImage
from skimage.util import compare_images


def is_image_list(images: list):
    # return all(isinstance(image, PILImage.Image) for image in images)
    return is_list_of(images, PILImage.Image)


def is_tensor_list(images: list) -> TypeGuard[list[torch.Tensor]]:
    return all(isinstance(image, torch.Tensor) for image in images)


def interlace_images(images: list[torch.Tensor], maxImages: int = 8) -> torch.Tensor:
    """
    assumes image tensors are of shape (batch, channels, height, width)

    takes list of images and interlaces them into a single tensor of images of size
    (batch * len(images), channels, height, width)

    """
    if len(images) == 1:
        return images[0]

    numImages = min(images[0].shape[0], maxImages)
    # logIms = [torch.stack(row, dim=0) for row in zip(*[im[:numImages] for im in images])]
    # return torch.cat(logIms, dim=0).detach().cpu()
    logIms = torch.hstack([im[:numImages, None] for im in images]).flatten(0, 1)
    return logIms


def imsave(
    image: np.ndarray,
    path: Path,
):
    with path.open("wb") as fp:
        PILImage.Image.save(image, fp)


def imshow(
    image: np.ndarray,
    figsize=(12, 12),
):
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(image)
    plt.axis("off")
    plt.show()


def _images_to_tensor(
    images: list[np.ndarray] | list[PILImage.Image] | list[torch.Tensor] | torch.Tensor,
    interlace=False,
) -> torch.Tensor:
    if isinstance(images, (list, tuple)):
        if is_image_list(images):
            tensors = [ToTensor()(s).unsqueeze(0) for s in images]
        elif is_tensor_list(images):
            tensors = images
        else:
            raise ValueError("images must be a list of PIL Images or Tensors")
        if interlace:
            tensors = interlace_images(tensors).detach().cpu()
        else:
            tensors = torch.cat(tensors, dim=0)
    else:
        tensors = images

    return tensors


def save_images(
    images: list[torch.Tensor] | torch.Tensor,
    names: list[str],
    save_dir: Path,
    ext: str = "png",
):
    """save batch of images

    if a list of torch.Tensor passed, assumes the batch should be interlaced

    Args:
        images (list[torch.Tensor] | torch.Tensor): _description_
        names (list[str]): _description_
        save_dir (Path): _description_
        ext (str, optional): _description_. Defaults to "png".
    """
    ext = ext.lstrip(".")
    save_dir.mkdir(parents=True, exist_ok=True)
    transform = ToPILImage()
    if isinstance(images, (list, tuple)):
        for i, batch in enumerate(zip(*images, strict=True)):
            for n, b in zip(names, batch, strict=True):
                save_path = save_dir / f"{i}" / f"{n}.{ext}"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                imsave(image=transform(b), path=save_path)
    else:
        for image, name in zip(images, names, strict=True):
            save_path = save_dir / f"{name}.{ext}"
            imsave(image=transform(image), path=save_path)


def display_images(
    images: list[np.ndarray] | list[PILImage.Image] | list[torch.Tensor] | torch.Tensor,
    n_columns=1,
    figsize=(12, 12),
    normalize=False,
    interlace=False,
    padding=1,
    save_path: Path | None = None,
):
    tensors = _images_to_tensor(images, interlace=interlace)
    image = ToPILImage()(
        make_grid(tensors, nrow=n_columns, padding=padding, normalize=normalize)
    )
    if save_path is None:
        imshow(
            image=image,
            figsize=figsize,
        )
    else:
        imsave(
            image=image,
            path=save_path,
        )
        plt.close()


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


def tensor_to_rgb(x, mean_shift=1.0, std_shift=2.0):
    return torch.clamp((x + mean_shift) / std_shift, min=0.0, max=1.0)


def show_tensor_image(image: torch.Tensor):
    reverseTransforms = ToNumpyImage()
    if len(image.shape) == 4:  # if batched image, show the first one
        image = image[0, :, :, :]
    plt.imshow(reverseTransforms(image))


def image_diff(
    image1: torch.Tensor | np.ndarray,
    image2: torch.Tensor | np.ndarray,
    return_tensor=True,
):
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
    H_out = np.floor(
        (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / (stride[0])
        + 1
    )
    W_out = np.floor(
        (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / (stride[1])
        + 1
    )
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
    H_out = (
        (H_in - 1) * stride[0]
        - 2 * padding[0]
        + dilation[0] * (kernel_size[0] - 1)
        + output_padding[0]
        + 1
    )
    W_out = (
        (W_in - 1) * stride[1]
        - 2 * padding[1]
        + dilation[1] * (kernel_size[1] - 1)
        + output_padding[1]
        + 1
    )
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


def calc_conv_shapes(imageShape: tuple):
    calc_ae_conv_transpose_2d(*calc_ae_conv_2d(*imageShape, 5), numLayers=5)
