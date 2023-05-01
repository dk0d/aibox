try:
    import torch
    from torchvision.transforms import ToPILImage, Lambda, Compose
except ImportError:
    print("pytorch required for these utilities")
    exit(1)

import numpy as np


class ToNumpyImage:
    def __init__(self, mode="RGB"):
        """Transforms a CHW tensor to numpy image.

        Args:
            denorm (bool, optional): Flag to denormalize image. Assumes _. Defaults to True.
        """
        self.transform = Compose([ToPILImage(mode=mode), Lambda(lambda img: np.array(img))])

    def __call__(self, image: torch.Tensor):
        out = self.transform(image)
        return out
