import argparse
from .utils import fix_mlflow_artifact_paths
from pathlib import Path


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
        fix_mlflow_artifact_paths(Path(args.fix_mlflow_path))


if __name__ == "__main__":
    main()
