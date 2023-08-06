from aibox.config import init_from_cfg, SUPPORTED_INIT_ARGS_KEYS, SUPPORTED_INIT_TARGET_KEYS
import copy
import pytest


def test_init_styles():
    specs = []
    for target_key in SUPPORTED_INIT_TARGET_KEYS:
        specs.append(
            {
                "name": f"targetonly-{target_key}",
                "config": {
                    target_key: "aibox.torch.callbacks.LogImagesCallback",
                    "batch_frequency": 2,
                },
            }
        )
        for arg_key in SUPPORTED_INIT_ARGS_KEYS:
            specs.append(
                {
                    "name": f"paired-{target_key}-{arg_key}",
                    "config": {
                        target_key: "aibox.torch.callbacks.LogImagesCallback",
                        arg_key: {"batch_frequency": 2},
                        "other": {"something_that_should_not_be_touched": 10000},
                    },
                }
            )

    # Deprecated structure
    specs.extend(
        [
            {
                "name": "deprecated-1",
                "config": {
                    "target": "aibox.torch.callbacks.LogImagesCallback",
                    "kwargs": {"batch_frequency": 2},
                    "other": {"something_that_should_not_be_touched": 10000},
                },
            },
            {
                "name": "deprecated-2",
                "config": {
                    "class_path": "aibox.torch.callbacks.LogImagesCallback",
                    "args": {"batch_frequency": 2},
                    "other": {"something_that_should_not_be_touched": 10000},
                },
            },
        ]
    )
    for spec in specs:
        name = spec["name"]
        config = spec["config"]
        config_copy = copy.deepcopy(config)
        try:
            callback = init_from_cfg(config)
            assert callback.batch_frequency == 2
            assert config_copy == config
            if "other" in config:
                assert (
                    config["other"]["something_that_should_not_be_touched"] == 10000
                ), f"Other keys should not be touched: {name}"
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {name}: {e}  {config}")


test_init_styles()
