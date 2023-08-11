import copy

import aibox
import pytest
from aibox.config import (
    SUPPORTED_INIT_ARGS_KEYS,
    SUPPORTED_INIT_TARGET_KEYS,
    config_from_dict,
    init_from_cfg,
    ConfigDict,
)
from aibox.utils import as_path
from omegaconf import DictConfig


def make_config_variations():
    specs = []
    for target_key in SUPPORTED_INIT_TARGET_KEYS:
        # No args
        specs.append(
            (
                f"targetonly-noargs-{target_key}",
                {
                    target_key: "aibox.config.ConfigDict",
                },
            )
        )
        # With args
        specs.append(
            (
                f"targetonly-args-{target_key}",
                {
                    target_key: "aibox.config.ConfigDict",
                    "a": 2,
                },
            )
        )
        # With args and other keys
        for arg_key in SUPPORTED_INIT_ARGS_KEYS:
            specs.append(
                (
                    f"paired-{target_key}-{arg_key}",
                    {
                        target_key: "aibox.config.ConfigDict",
                        arg_key: {"a": 2},
                        "other": {"something_that_should_not_be_touched": 10000},
                    },
                )
            )

    # Deprecated structure
    specs.extend(
        [
            (
                "deprecated-1",
                {
                    "target": "aibox.config.ConfigDict",
                    "kwargs": {"a": 2},
                    "other": {"something_that_should_not_be_touched": 10000},
                },
            ),
            (
                "deprecated-2",
                {
                    "class_path": "aibox.config.ConfigDict",
                    "args": {"a": 2},
                    "other": {"something_that_should_not_be_touched": 10000},
                },
            ),
        ]
    )
    return specs


@pytest.mark.parametrize("name,config", make_config_variations())
def test_config_parse(name, config):
    # TODO: test cases that should fail well
    config_copy = copy.deepcopy(config)
    callback = init_from_cfg(config)
    if "-args-" in name:
        assert callback.a == 2
    assert config_copy == config
    if "other" in config:
        assert (
            config["other"]["something_that_should_not_be_touched"] == 10000
        ), f"Other keys should not be touched: {name}"


def test_config_dict():
    config = ConfigDict()
    config.__classpath__ = "aibox.config.ConfigDict"
    config.a = 2
    callback = init_from_cfg(config)
    assert isinstance(callback, ConfigDict), f"Expected ConfigDict got {type(callback)}"
    assert callback.a == 2, f"Expected a=2, got {callback.a}"


def test_config_conversion():
    config_dict = ConfigDict()
    config_dict.__classpath__ = "aibox.config.ConfigDict"
    config_dict.a = 2
    config = config_from_dict(config_dict)
    assert isinstance(config, DictConfig), f"Expected DictConfig, got {type(config)}"


def test_as_path_resolver():
    user_dir = as_path("~")
    config = ConfigDict()
    config.root_dir = "${as_path:'~'}"
    config = config_from_dict(config)
    assert str(config.root_dir) == str(user_dir), f"Expected {user_dir}, got {config.root_dir}"
