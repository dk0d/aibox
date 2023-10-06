import argparse
import shutil

from aibox.cli import AIBoxCLI
from aibox.slurm import SlurmDirectives, submit_slurm_script
from aibox.utils import as_path, print


def sbatch_available() -> bool:
    return shutil.which("sbatch") is not None


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        default=False,
    )
    args, file_args = parser.parse_known_args(args)
    commands = []
    # remove any leading commands
    while len(file_args) > 0:
        if "-" not in file_args[0]:
            commands.append(file_args.pop(0))
        else:
            break

    config = AIBoxCLI().parse_args(args=file_args)

    if not sbatch_available() and not args.debug:
        print("sbatch not available, running in debug mode")
        args.debug = True

    if args.debug:
        config.debug = True

    submit_slurm_script(
        name=config.name,
        env_name=config.slurm.env_name,
        py_file_path=config.slurm.python_file,
        py_file_args=commands + file_args,  # forward the original commands and args
        scripts_dir=config.slurm.script_dir,
        log_dir=as_path(config.slurm.script_dir) / "logs",
        cudaVersion=config.slurm.cuda,
        modules=config.slurm.get("modules", None),
        slurm_cfg=SlurmDirectives(**config.slurm),
        conda_envs_dir=config.slurm.env_dir,
        debug=args.debug,
    )
