from aibox.config import init_from_cfg
import copy
import pytest


def test_init_styles():
    configs = [
        {
            "class_path": "aibox.torch.callbacks.LogImagesCallback",
            "args": {"batch_frequency": 2},
            "other": {"something_that_should_not_be_touched": 10000},
        },
        {
            "__class__": "aibox.torch.callbacks.LogImagesCallback",
            "__args__": {"batch_frequency": 2},
            "other": {"something_that_should_not_be_touched": 10000},
        },
        {
            "__classpath__": "aibox.torch.callbacks.LogImagesCallback",
            "__args__": {"batch_frequency": 2},
            "other": {"something_that_should_not_be_touched": 10000},
        },
        {
            "__classpath__": "aibox.torch.callbacks.LogImagesCallback",
            "__args__": {"batch_frequency": 2},
            "other": {"something_that_should_not_be_touched": 10000},
        },
        {
            "target": "aibox.torch.callbacks.LogImagesCallback",
            "kwargs": {"batch_frequency": 2},
            "other": {"something_that_should_not_be_touched": 10000},
        },
        # Flattened args
        {
            "__classpath__": "aibox.torch.callbacks.LogImagesCallback",
            "batch_frequency": 2,
        },
        {
            "__class_path__": "aibox.torch.callbacks.LogImagesCallback",
            "batch_frequency": 2,
        },
        {
            "__class__": "aibox.torch.callbacks.LogImagesCallback",
            "batch_frequency": 2,
        },
    ]

    for config in configs:
        config_copy = copy.deepcopy(config)
        callback = init_from_cfg(config)
        assert callback.batch_frequency == 2
        assert config_copy == config
        if "other" in config:
            assert "other" in config
            assert config["other"]["something_that_should_not_be_touched"] == 10000


test_init_styles()
