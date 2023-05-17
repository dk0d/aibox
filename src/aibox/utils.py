from pprint import pformat
from rich import print as rprint


def print(*args, **kwargs):
    args = [a if isinstance(a, str) else pformat(a) for a in args]
    rprint(*args, **kwargs)
