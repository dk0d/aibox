try:
    import torch
    from torchvision.transforms import Lambda, Compose
except ImportError:
    print("pytorch required for these utilities")
    exit(1)

import numpy as np


class TensorImageToNumpy:
    def __init__(self, denorm=True):
        """Transforms a CHW tensor to numpy image.

        Args:
            denorm (bool, optional): Flag to denormalize image. Assumes _. Defaults to True.
        """
        if denorm:
            transforms = [Lambda(lambda t: (t + 1) / 2)]
        else:
            transforms = []

        transforms.extend(
            [
                Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
                Lambda(lambda t: t * 255.0),
                Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
            ]
        )
        self.transform = Compose(transforms)

    def __call__(self, image: torch.Tensor):
        return self.transform(image)
