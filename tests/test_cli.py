from aibox.cli import cli_main, CLIException
import pytest

DEFAULT_CONFIG_DIR = "tests/resources/configs"


def test_cli_no_args():
    cli_main(args=[])


def test_cli_args():
    config = cli_main(["-e", "debug", "--model.args.name", "TESTMODEL", "-cd", DEFAULT_CONFIG_DIR])
    assert hasattr(config, "model")
    assert config.model.args.name == "TESTMODEL"


def test_cli_folder_model():
    config = cli_main(["-e", "debug", "-m", "test", "-cd", DEFAULT_CONFIG_DIR])
    assert config.model.class_path == "ae.models.DFCVAE"
    assert config.model.args.name == "non-default"
    assert config.trainer.accelerator == "ddp"


def test_cli_args_dotlist():
    config = cli_main(["-cd", DEFAULT_CONFIG_DIR, "--model.args.name=TESTMODEL"])
    assert config.model.args.name == "TESTMODEL"


def test_cli_bad_args():
    with pytest.raises(CLIException):
        cli_main(["-cd", DEFAULT_CONFIG_DIR, "--model.args.name"])
