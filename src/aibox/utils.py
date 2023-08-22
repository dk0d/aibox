from collections.abc import Sequence
from pathlib import Path
from pprint import pformat
from typing import TypeGuard, TypeVar
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as rprint

# import numpy as np

T = TypeVar("T")


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


def as_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
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


def resolve_paths(config, new_root=Path.cwd()):
    """Resolves the most likely root path in a config based on all keys that contain
    `root` or `dir` in their key name and replaces it with new_root

    The old root is determined by finding the longest common path between all paths

    Args:
        config: the config to resolve paths in
        new_root: the new root path to replace the old root path with.
            Defaults to the current working directory

    """
    from functools import reduce

    def _resolve_paths(config, old_root, new_root=Path.cwd()):
        for k, v in config.items():
            if ("root" in k or "dir" in k) and isinstance(v, str):
                try:
                    p = as_path(v)
                    if p is not None and p.is_relative_to(old_root):
                        config[k] = new_root / p.relative_to(old_root)
                except Exception:
                    pass
            elif isinstance(v, dict):
                _resolve_paths(v, old_root, new_root=new_root)

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
        _resolve_paths(config, old_root=old_root, new_root=new_root)
    return config
