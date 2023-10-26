from typing import Any

import numpy as np
import torch
from PIL.Image import Image
from aibox.logger import get_logger

LOGGER = get_logger(__name__)
try:
    from ffcv.fields import (
        Field,
        RGBImageField,
        BytesField,
        IntField,
        FloatField,
        NDArrayField,
        JSONField,
        TorchTensorField,
    )

    from ffcv.loader import OrderOption
    from ffcv.reader import Reader

    def field_to_str(f: Field) -> str:
        mapping = {
            RGBImageField: "image",
            BytesField: "bytes",
            IntField: "int",
            FloatField: "float",
            NDArrayField: "array",
            JSONField: "json",
            TorchTensorField: "tensor",
        }
        return mapping[f]

    def obj_to_field(obj: Any) -> Field:
        if isinstance(obj, Image):
            return RGBImageField(write_mode="jpg")

        elif isinstance(obj, int):
            return IntField()

        elif isinstance(obj, float):
            return FloatField()

        elif isinstance(obj, np.ndarray) and not isinstance(obj[0], int):
            return NDArrayField(obj.dtype, obj.shape)

        elif isinstance(obj, np.ndarray) and isinstance(obj[0], int):
            return BytesField()

        elif isinstance(obj, dict):
            return JSONField()

        elif isinstance(obj, torch.Tensor):
            return TorchTensorField(obj.dtype, obj.shape)

        else:
            raise AttributeError(f"FFCV dataset can not manage {type(obj)} objects")

except ImportError:
    LOGGER.error("FFCV or FFCV Dependencies are not installed. Please see https://ffcv.io for more.")
    exit(0)
