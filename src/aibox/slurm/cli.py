import argparse
from aibox.cli import AIBoxCLI
from aibox.slurm import SlurmConfig, submit_slurm_script

import shutil

# from aibox.config import config_merge
from aibox.utils import as_path, print


def sbatch_available() -> bool:
    return shutil.which("sbatch") is not None


#
# class SlurmParser(AIBoxCLI):
#     def __init__(self) -> None:
#         super().__init__()
#         self.parser.add_argument(
#             "--slurm.env_name",
#             "-slurm.env_name",
#             help="name of conda environment to run in configs/environments",
#             required=True,
#         )
#         self.parser.add_argument(
#             "--slurm.env_dir",
#             "-slurm.env_dir",
#             help="path to the conda environments directory",
#             action=PathAction,
#             default=as_path("~/.conda/envs/"),
#         )
#         self.parser.add_argument(
#             "--slurm.python_file",
#             "-slurm.python_file",
#             required=True,
#             action=PathAction,
#         )
#
#         self.parser.add_argument(
#             "--slurm.script_dir",
#             "-slurm.script_dir",
#             default="slurm-scripts",
#             action=PathAction,
#         )
#
#     def parse_args(self, args=None):
#         cli_config = self._parse_cli_args(args)
#
#         _, default_config = self._load_config(
#             root=cli_config.exp_dir,
#             name="default",
#         )
#
#        _, experiment_config = self._resolve_config_from_root_name(root=cli_config.exp_dir, name=cli_config.exp_name)
#
#         # = Path("configs/experiments") / f"{cli_config.exp_name}.toml"
#         config = config_merge(default_config, experiment_config, cli_config)
#         config = self._resolve_links(config)
#         return config


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
    )
    args, unknown = parser.parse_known_args(args)
    if len(unknown) % 2 != 0:
        # assume the first argument is a subcommand
        config_args = unknown[1:]
    else:
        config_args = unknown

    config = AIBoxCLI().parse_args(args=config_args)

    if args.debug:
        config.debug = True

    if not sbatch_available() and not args.debug:
        print("sbatch not available, exiting")
        return

    # try:
    submit_slurm_script(
        name=config.name,
        env_name=config.slurm.env_name,
        py_file_path=config.slurm.python_file,
        py_file_args=unknown,
        scripts_dir=config.slurm.script_dir,
        log_dir=as_path(config.slurm.script_dir) / "logs",
        cudaVersion=config.slurm.cuda,
        slurm_cfg=SlurmConfig(**config.slurm),
        conda_envs_dir=config.slurm.env_dir,
        debug=args.debug,
    )
    # except Exception as err:
    #     print(err)
