import argparse
from pathlib import Path

from aibox.utils import print

try:
    from aibox.mlflow import fix_mlflow_artifact_paths
except ImportError:
    fix_mlflow_artifact_paths = None


def run(args, install):
    import subprocess

    if not install:
        print(f"[blue bold]\n\\[Dry Run][/blue bold]\n{' '.join(args)}\n")
    else:
        print(f'[green bold]\n\\[Running][/green bold]\n{" ".join(args)}\n')
        subprocess.run(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix-mlflow-path",
        "-mlflow",
        help="Path to root of mlruns folder. Rewrites all metadata for artifacts based on given path to mlruns folder",
        default=None,
    )
    parser.add_argument(
        "--ffcv",
        "-ffcv",
        action="store_true",
        default=False,
        help="Install FFCV and dependencies. Requires conda env.",
    )
    args = parser.parse_args()

    if args.fix_mlflow_path is not None:
        if fix_mlflow_artifact_paths is not None:
            fix_mlflow_artifact_paths(Path(args.fix_mlflow_path))
        else:
            print("Unable to import mlflow package")

    if args.ffcv:
        import shutil

        if shutil.which("mamba") is not None:
            command = "mamba"
        else:
            command = "conda"

        command = [
            command,
            "install",
            "cupy",
            "pkg-config",
            "libjpeg-turbo",
            "opencv",
            "numba",
            "-c",
            "conda-forge",
        ]
        run(command, install=True)

        pip_cmd = [
            "pip",
            "install",
            "ffcv",
        ]
        run(pip_cmd, install=True)

        # print("[green bold] FFCV installed successfully")


if __name__ == "__main__":
    main()
