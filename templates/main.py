# Really basic main file to get you started.
# not guaranteed to work, but should be a good starting point.

import argparse
from aibox.slurm.cli import main as slurm_main
from aibox.torch.training import main as train_main
from aibox.torch.evaluate import main_cli as eval_main


try:
    from aibox.torch.tuning import main as tune_main

    def tune(args):
        tune_main(args)
except ImportError:
    tune_main = None


def train(args):
    train_main(args)


def evaluate(args):
    eval_main(args)


def submit(args):
    import os
    from pathlib import Path

    file_path = as_path(__file__)
    conda_prefix = as_path(os.environ.get("CONDA_PREFIX", "~/.conda/envs/"))
    env_dir = conda_prefix.parent.as_posix()
    env_name = conda_prefix.name

    if args is None:
        args = []

    args += [
        f"--slurm.python_file={file_path.as_posix()}",
        f"--slurm.env_name={env_name}",
        f"--slurm.env_dir={env_dir}",
        f"--slurm.script_dir={(Path.cwd() / 'slurm-scripts').as_posix()}",
    ]
    if "tune" in args:
        args += ["--slurm.nodes=4"]
        args += ["--slurm.ray_tune=true"]
    slurm_main(args)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.set_defaults(func=train)

    submit_parser = subparsers.add_parser("submit", help="Submit this file to slurm with any args that follow")
    submit_parser.set_defaults(func=submit)

    if tune_main is not None:
        tune_parser = subparsers.add_parser(
            "tune", help="Tune training using Ray Tune. Uses slurm if 'sbatch' detected"
        )
        tune_parser.set_defaults(func=tune)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation on model for test and/or predict splits")
    eval_parser.set_defaults(func=evaluate)

    args, func_args = parser.parse_known_args()
    args.func(func_args)


if __name__ == "__main__":
    main()
