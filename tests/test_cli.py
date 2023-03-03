from aibox.cli import cli_main, CLIException
import pytest

DEFAULT_CONFIG_PATH = "tests/resources/configs/default.toml"


def test_cli_no_args():
    cli_main(args=[])


def test_cli_args():
    config = cli_main(["-c", DEFAULT_CONFIG_PATH, "--model.args.name", "TESTMODEL"])
    assert config.model.args.name == "TESTMODEL"


def test_cli_none():
    config = cli_main(["-c", DEFAULT_CONFIG_PATH])
    assert config.exp_name is None


def test_cli_args_dotlist():
    config = cli_main(["-c", DEFAULT_CONFIG_PATH, "--model.args.name=TESTMODEL"])
    assert config.model.args.name == "TESTMODEL"


def test_cli_bad_args():
    with pytest.raises(CLIException):
        cli_main(["-c", DEFAULT_CONFIG_PATH, "--model.args.name"])
