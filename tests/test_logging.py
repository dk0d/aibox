from aibox.torch.logging import CombinedLogger
import pytest

DEFAULT_CONFIG_DIR = "tests/resources/configs"


def test_cli_no_args():
    logger = CombinedLogger("./logs")
