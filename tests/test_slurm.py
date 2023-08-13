# %%
import pytest
import argparse
from aibox.cli import AIBoxCLI
from aibox.slurm.cli import main as submit_main
from aibox.utils import print


DEFAULT_CONFIG_DIR = "tests/resources/configs"


@pytest.mark.parametrize(
    "args",
    [
        [
            "train",
            "-e",
            "debug",
            "-cd",
            DEFAULT_CONFIG_DIR,
            "--model.name=TESTMODEL",
            "--slurm.env_name=testenv",
            "--debug",
        ],
        [
            "train",
            "-e",
            "debug-sl-modules",
            "-cd",
            DEFAULT_CONFIG_DIR,
            "--model.name=TESTMODEL",
            "--slurm.env_name=testenv",
            "--debug",
        ],
    ],
)
def test_slurm_cli(args):
    submit_main(args)
