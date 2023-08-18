from collections.abc import Sequence
from pathlib import Path
from pprint import pformat
from typing import TypeGuard, TypeVar

from omegaconf import DictConfig, ListConfig, OmegaConf
from rich import print as rprint

T = TypeVar("T")


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


def as_uri(path: str | Path) -> str:
    import re

    if re.search(r"^[\w]+://", str(path)) is not None:
        # already a URI
        return str(path)
    return as_path(path).as_uri()


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
                    if p.is_relative_to(old_root):
                        config[k] = new_root / p.relative_to(old_root)
                except Exception:
                    pass
            elif isinstance(v, dict):
                _resolve_paths(v, old_root, new_root=new_root)

    def _path_intersection(p1, p2):
        p1, p2 = as_path(p1), as_path(p2)
        same = []
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
