from aibox.cli import cli_main, CLIException
import pytest


def test_cli_no_args():
    cli_main(args=[])

def test_cli_args():
    cli_main(["-c", "tests/resources/config.toml", "--model.args.name", "TESTMODEL"])

def test_cli_bad_args():
    with pytest.raises(CLIException):
        cli_main(["-c", "tests/resources/config.toml", "--model.args.name"])

    