from pathlib import Path
from typing import Annotated

import typer

from aibox.mlflow import fix_mlflow_artifact_uri_sqlite
from aibox.utils import Err, Ok, get_dirs, print
from returns.result import safe


app = typer.Typer()


def run(args, install):
    import subprocess

    if not install:
        print(f"[blue bold]\n\\[Dry Run][/blue bold]\n{' '.join(args)}\n")
    else:
        print(f"[green bold]\n\\[Running][/green bold]\n{' '.join(args)}\n")
        subprocess.run(args, check=False)


@safe
def find_mlruns_folders(root: Path, recurse: bool) -> list[Path]:
    if recurse:
        paths = [Path(p) for p in get_dirs(root, filter="^mlruns$", desc="Searching for mlruns folders")]
        return paths

    if (root / "mlruns").exists():
        return [root / "mlruns"]
    elif root.name == "mlruns":
        return [root]

    raise ValueError(f"Given path does not contain an mlruns folder: {root}")


@app.command(
    help="Fix MLflow artifact paths in mlruns folder. Rewrites all metadata for artifacts based on given path to mlruns folder. "
    + "Can optionally recurse into subfolders looking for all mlruns folders.",
)
def mlflow(
    root: Annotated[Path, typer.Argument(..., help="Path to directory that contains the mlruns folder")] = Path.cwd(),
    db: Annotated[
        Path | None,
        typer.Option(..., help="Path to sqlite DB to update its artifact_uri to where the root param specifies"),
    ] = None,
    recurse: Annotated[
        bool, typer.Option("--recurse", "-r", help="Recurse into subfolders looking for mlruns folders")
    ] = False,
):
    try:
        from aibox.mlflow import fix_mlflow_artifact_paths
    except ImportError:
        raise ImportError("mlflow package is not installed. Please install it with `pip install mlflow`")

    if db is not None:
        fix_mlflow_artifact_uri_sqlite(root, db)
        return

    res = find_mlruns_folders(root, recurse)

    match res:
        case Ok(paths):
            print(f"[blue bold]Found {len(paths)} mlruns folders to fix:")
            print("\n".join([f"- {p}" for p in paths]))
            if fix_mlflow_artifact_paths is not None:
                for path in paths:
                    fix_mlflow_artifact_paths(Path(path))
            else:
                print("Unable to import mlflow package")

        case Err(err):
            print(f"[red bold]Error finding mlruns folders: {err}")


@app.command(
    help="Install FFCV and dependencies. Requires conda env.",
)
def ffcv():
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

    print("[green bold] FFCV installed successfully")


def main():
    app()


if __name__ == "__main__":
    main()
