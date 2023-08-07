import argparse
from pathlib import Path

from aibox.utils import print

try:
    from aibox.mlflow import fix_mlflow_artifact_paths
except ImportError:
    fix_mlflow_artifact_paths = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix-mlflow-path",
        "-mlflow",
        help="Path to root of mlruns folder. Rewrites all metadata for artifacts based on given path to mlruns folder",
        default=None,
    )
    args = parser.parse_args()

    if args.fix_mlflow_path is not None:
        if fix_mlflow_artifact_paths is not None:
            fix_mlflow_artifact_paths(Path(args.fix_mlflow_path))
        else:
            print("Unable to import mlflow package")


if __name__ == "__main__":
    main()
