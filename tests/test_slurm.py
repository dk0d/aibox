# %%
import argparse
from aibox.cli import AIBoxCLI
from aibox.slurm.cli import main as submit_main
from aibox.utils import print


DEFAULT_CONFIG_DIR = "tests/resources/configs"


def test_slurm_cli():
    args = [
        "train",
        "-e",
        "debug",
        "-cd",
        DEFAULT_CONFIG_DIR,
        "--model.name=TESTMODEL",
        "--slurm.env_name=testenv",
        "--debug",
    ]
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--debug",
    #     "-d",
    #     action="store_true",
    #     default=False,
    # )
    # args, unknown = parser.parse_known_args(args)
    # if len(unknown) % 2 != 0:
    #     command, unknown = unknown[0], unknown[1:]
    #     print(command, unknown)
    # else:
    #     command = None
    #
    # config = AIBoxCLI().parse_args(args=unknown)
    # print(config)
    submit_main(args)


test_slurm_cli()
