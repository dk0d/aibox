import os
from collections.abc import Sequence
from pathlib import Path
from pprint import pformat
from typing import TypeGuard, TypeVar

import tqdm
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as rprint

from aibox.logger import get_logger

# import numpy as np

T = TypeVar("T")

LOGGER = get_logger(__name__)


def is_list(x) -> bool:
    return OmegaConf.is_list(x) or isinstance(x, list)


def is_dict(x) -> bool:
    return OmegaConf.is_dict(x) or isinstance(x, dict)


def get_files(folder: Path | str, allowed_exts: list[str] | None, desc=None):
    """Recursively gets_files in directory. faster than Path.glob, walk, etc."""

    if not Path(folder).exists():
        LOGGER.warn(f'Folder does not exist: {folder}')
        return

    check_ext = allowed_exts is not None and len(allowed_exts) > 0
    if check_ext:
        # remove period at beginning if present
        allowed_exts = [a[1:] if a.startswith(".") else a for a in allowed_exts]

    if desc is None:
        progress = None
    else:
        progress = tqdm.tqdm(desc=desc)

    def _get_files(_folder):
        with os.scandir(_folder) as scan:
            for item in scan:
                if item.is_dir():
                    for p in _get_files(item):
                        yield p
                    continue
                p = None
                if check_ext:
                    if item.name.split(".")[-1] in allowed_exts:
                        p = item.path
                else:
                    p = item.path
                if p is not None and progress is not None:
                    progress.update()
                if p is not None:
                    yield p

    for p in _get_files(folder):
        yield p

    if progress is not None:
        progress.close()


def nearest_square_grid(num: int) -> tuple[int, int]:
    """
    Returns the nearest square number to the given number
    as a tuple of (nrows, ncols)

    assumes num < 50

    """

    primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    if num in primes:
        num = num + 1

    factors = [i for i in range(1, num + 1) if num % i == 0]
    nrows, ncols = factors[len(factors) // 2], factors[len(factors) // 2 - 1]
    if nrows * nrows == num:
        return nrows, nrows
    return nrows, ncols


def is_list_of(obj: Sequence, T) -> TypeGuard[Sequence]:
    return all(isinstance(el, T) for el in obj)


def as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def print(*args, **kwargs):
    def _config_format(arg):
        if isinstance(arg, (DictConfig, ListConfig)):
            arg = OmegaConf.to_container(arg, resolve=True)
        return pformat(arg)

    args = [a if isinstance(a, str) else _config_format(a) for a in args]
    rprint(*args, **kwargs)


def as_uri(path: str | Path | None) -> str | None:
    if path is None:
        return None

    import re

    if re.search(r"^[\w]+://", str(path)) is not None:
        # already a URI
        return str(path)

    path = as_path(path)

    if path is not None:
        return path.as_uri()

    return str(path)


def chunk(iterable, n):
    """
    Yield successive n-sized chunks from iterable.

    Args:
        iterable: the iterable to chunk
        n: the size of each chunk

    Returns:
        a generator that yields chunks of size n from iterable
    """
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def get_config_root(config):
    paths = []
    for k, v in config.items():
        if ("root" in k or "dir" in k) and isinstance(v, str):
            paths.append(as_path(v))
        elif isinstance(v, dict):
            paths.extend(get_config_root(v))
    return paths


def _resolve_paths(config, old_root, new_root=Path.cwd(), verbose=False):
    for k, v in config.items():
        if ("root" in k or "dir" in k or "path" in k) and (isinstance(v, str) or isinstance(v, Path)):
            try:
                p = as_path(v)
                if p is not None and p.is_relative_to(old_root):
                    if verbose:
                        LOGGER.info(f"Replacing key {k}\n{p}\nwith\n{new_root / p.relative_to(old_root)}")
                    config[k] = new_root / p.relative_to(old_root)
            except Exception as e:
                if verbose:
                    LOGGER.error(f"Could not resolve path for key {k}\n{v}\n{e}")
                pass
        elif is_dict(v):
            _resolve_paths(v, old_root, new_root=new_root, verbose=verbose)


def resolve_paths(config, new_root=Path.cwd(), verbose=False):
    """Resolves the most likely root path in a config based on all keys that contain
    `root` or `dir` in their key name and replaces it with new_root

    The old root is determined by finding the longest common path between all paths

    Args:
        config: the config to resolve paths in
        new_root: the new root path to replace the old root path with.
            Defaults to the current working directory

    """
    from functools import reduce

    def _path_intersection(p1, p2):
        p1, p2 = as_path(p1), as_path(p2)
        same = []
        if p1 is not None and p2 is not None:
            for c1, c2 in zip(p1.parts, p2.parts):
                if c1 != c2:
                    break
                same.append(c1)
        return as_path("/".join(same))

    paths: list[Path] = get_config_root(config)
    old_root = None
    if len(paths) > 0:
        old_root = reduce(lambda x, y: _path_intersection(x, y), paths)
    if old_root is not None:
        _resolve_paths(config, old_root=old_root, new_root=new_root, verbose=verbose)
    return config
