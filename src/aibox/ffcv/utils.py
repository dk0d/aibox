from typing import Any

import numpy as np
import torch
from aibox.logger import get_logger
from PIL.Image import Image

LOGGER = get_logger(__name__)
try:
    from ffcv.fields import (
        BytesField,
        Field,
        FloatField,
        IntField,
        JSONField,
        NDArrayField,
        RGBImageField,
        TorchTensorField,
    )

    def field_to_str(f: Field) -> str:
        if isinstance(f, RGBImageField):
            return "image"
        elif isinstance(f, BytesField):
            return "bytes"
        elif isinstance(f, IntField):
            return "int"
        elif isinstance(f, FloatField):
            return "float"
        elif isinstance(f, NDArrayField):
            return "array"
        elif isinstance(f, JSONField):
            return "json"
        elif isinstance(f, TorchTensorField):
            return "tensor"
        else:
            raise AttributeError(f"FFCV dataset can not manage {type(f)} fields")

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
    import sys

    LOGGER.error("FFCV or FFCV Dependencies are not installed. Please see https://ffcv.io for more.")
    sys.exit(0)
