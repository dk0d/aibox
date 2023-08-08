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
