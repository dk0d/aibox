try:
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import Compose, Lambda, ToPILImage
except ImportError:
    print("pytorch required for these utilities")
    exit(1)

from enum import Enum

import numpy as np
from PIL import Image as PILImage


class ToNumpyImage:
    def __init__(self, mode=None):
        """Transforms a CHW tensor to numpy image.

        Args:
            denorm (bool, optional): Flag to denormalize image. Assumes _. Defaults to True.
        """
        self.transform = Compose([ToPILImage(mode=mode), Lambda(lambda img: np.array(img))])

    def __call__(self, image: torch.Tensor):
        out = self.transform(image)
        return out


class BlendMode(Enum):
    linear = "linear"
    composite = "composite"


# AggFn = 'occlusion' | 'occlude' | '*' | 'multiply' | 'prod' | 'sum'


class ImageBlend:
    def __init__(
        self,
        blend_mode: BlendMode | str = BlendMode.composite,
        weight_range=(1, 0.1),  # (top layer, deepest layer)
        agg_fn="occlusion",
        prefilter_fn=None,
        clip=True,
        out_mode="RGB",
    ):
        self.blend_mode = blend_mode if isinstance(blend_mode, BlendMode) else BlendMode(blend_mode)
        self.weight_range = weight_range
        self.clip = clip
        self.agg_fn = agg_fn
        self.out_mode = out_mode
        if prefilter_fn is None:

            def _pf(img, **kwargs):  # passthrough
                return img

            self.prefilter_fn = _pf
        else:
            self.prefilter_fn = prefilter_fn

    def make_weights(self, layers: int):
        match self.blend_mode:
            case BlendMode.composite:
                weights = F.softmax(torch.linspace(*self.weight_range, steps=layers), dim=0).numpy()
            case BlendMode.linear:
                weights = np.linspace(*self.weight_range, num=layers)
        return weights

    def __call__(self, images: list[PILImage.Image] | list[np.ndarray]) -> PILImage.Image:
        if isinstance(images[0], PILImage.Image):
            images = [np.array(img) for img in images]

        weights = self.make_weights(len(images))

        print(weights)
        weights[0] = 1.0

        filtered = np.stack([self.prefilter_fn(img.astype(np.float32)) * weights[i] for i, img in enumerate(images)])

        print(filtered.dtype, filtered.shape, filtered.min(), filtered.max())
        match self.agg_fn:
            case "*" | "multiply" | "prod":
                out = np.prod(filtered, axis=0)
            case "occlusion" | "occlude":
                out = None

                for i, img in enumerate(images):
                    if out is None:
                        out = np.zeros_like(img)
                    print(np.any(img > 0), np.any(out == 0))
                    idxs = np.logical_and(img > 0, out == 0)
                    out[idxs] = filtered[i][idxs]

            case _:
                out = np.sum(filtered, axis=0)

        if self.clip:
            out = PILImage.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        else:
            PILImage.fromarray(out)

        return out.convert(self.out_mode)
