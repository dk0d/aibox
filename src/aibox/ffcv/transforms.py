from collections.abc import Callable
from dataclasses import replace
from typing import Optional

import torch

from aibox.logger import get_logger

LOGGER = get_logger(__name__)

try:
    from ffcv.pipeline.allocation_query import AllocationQuery
    from ffcv.pipeline.operation import Operation
    from ffcv.pipeline.state import State

    class DivideImage255(Operation):
        def __init__(self, dtype: torch.dtype):
            """
            Divide input torch tensor by 255. Intended to get images in range [0, 1]
            :param dtype: type of returned images. Accepts torch.float16, torch.float32, torch.float64
            """
            super().__init__()

            assert (
                dtype == torch.float16 or dtype == torch.float32 or dtype == torch.float64
            ), f"wrong dtype passed: {dtype}"

            self.dtype = dtype

        def generate_code(self) -> Callable:
            def divide(image, dst):
                dst = image.to(self.dtype) / 255.0
                return dst

            divide.is_parallel = True

            return divide

        def declare_state_and_memory(self, previous_state: State) -> tuple[State, Optional[AllocationQuery]]:
            return replace(previous_state, dtype=self.dtype), None

except ImportError:
    import sys

    LOGGER.error("FFCV or FFCV Dependencies are not installed. Please see https://ffcv.io for more.")
    sys.exit(0)
