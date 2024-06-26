import importlib
from pathlib import Path
from typing import TypeAlias

from omegaconf import DictConfig, OmegaConf

from aibox.logger import get_logger
from aibox.utils import as_path, as_uri


def basename(path: str) -> str:
    import re

    return re.split(r"[\./]", path)[-1]


OmegaConf.register_new_resolver("as_path", as_path)
OmegaConf.register_new_resolver("as_uri", as_uri)
OmegaConf.register_new_resolver("increment", lambda x: x + 1)
OmegaConf.register_new_resolver("decrement", lambda x: x - 1)
OmegaConf.register_new_resolver("add", lambda *nums: sum(nums))
OmegaConf.register_new_resolver("basename", basename)

Config: TypeAlias = DictConfig

SUPPORTED_INIT_TARGET_KEYS = [
    "__classpath__",
    "__class_path__",
    "__target__",
    "__init_target__",
]
SUPPORTED_INIT_ARGS_KEYS = [
    "__args__",
    "__kwargs__",
    "__init_args__",
    "__init_kwargs__",
]

LOGGER = get_logger(__name__)


def print_config(config):
    import json

    import rich

    rich.print_json(json.dumps(OmegaConf.to_container(config)))


# def rewrite_config_paths(config, orig_root, new_root, verbose=False):
#     new_config = config.copy()
#     dot_list = config_to_dotlist(config)
#     for k, v in dot_list.items():
#         if verbose:
#             print(k, v)
#         if isinstance(v, str):
#             if as_path(v).is_relative_to(orig_root):

#                 config_update(new_config, k, str(new_root / as_path(v).relative_to(orig_root)))
#     return new_config


def derive_classpath_deprecated(config: Config) -> str:
    """Derives the classpath from the given config"""
    conf = config_to_dict(config)
    if "class_path" in conf:
        class_string = conf["class_path"]
    elif "target" in conf:
        class_string = conf["target"]
    else:
        raise KeyError("Expected one of `class_path` or `target` as module path to instantiate object")
    return class_string


def derive_classpath(config: Config) -> str:
    """Derives the classpath from the given config"""
    resolved = []
    conf = config_to_dict(config)
    for skey in SUPPORTED_INIT_TARGET_KEYS:
        if skey not in conf:
            continue
        value = conf.pop(skey, None)
        if value is not None and len(resolved) == 0:
            resolved.append((skey, value))
        elif value is not None:
            raise ValueError(
                f"Multiple init targets specified in config: {skey} and {resolved[-1][0]}" f"are both present in {conf}"
            )
    class_string = resolved[-1][1] if len(resolved) > 0 else None

    if class_string is None:
        raise KeyError(f"None of the supported init target keys found in config keys: {list(config.keys())}")

    return class_string


def derive_args_deprecated(config: Config, **kwargs):
    conf = config_to_dict(config)
    params = conf.pop("args", dict())
    for key in ["args", "kwds", "kwargs", "init_args", "params"]:
        params.update(**conf.pop(key, dict()))

    params.update(**kwargs)
    return params


def derive_args(config: Config, **kwargs):
    params = {}
    conf = config_to_dict(config)
    args_specified = False
    for skey in SUPPORTED_INIT_ARGS_KEYS:
        args_specified = skey in conf if not args_specified else args_specified
        params.update(**conf.pop(skey, dict()))

    if not args_specified:
        for tkey in SUPPORTED_INIT_TARGET_KEYS:
            _ = conf.pop(tkey, None)
        params = dict(**conf)

    params.update(**kwargs)
    return params


# Config reading (ported from ldm module)
def class_from_string(string: str, reload=False):
    if "." in string:
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imported = importlib.import_module(module)
            importlib.reload(module_imported)
    else:
        module, cls = "__main__", string
    return getattr(importlib.import_module(module, package=None), cls)


def config_from_dict(d: dict) -> DictConfig:
    if isinstance(d, ConfigDict):
        d = d.to_dict()

    c = OmegaConf.create(d)
    return c


def config_from_namespace(namespace) -> Config:
    import argparse

    new = {}
    for k, v in vars(namespace).items():
        if isinstance(v, argparse.Namespace):
            new[k] = config_from_namespace(v)
        else:
            new[k] = v
    return config_from_dict(new)


def config_from_dotlist(dotlist: list[str]):
    return OmegaConf.from_dotlist(dotlist)


def config_to_dict(config: Config) -> dict:
    if isinstance(config, dict):
        return dict(**config)
    container = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    assert isinstance(container, dict)
    return container


def config_select(config, key, **kwargs):
    return OmegaConf.select(config, key, **kwargs)


def config_update(config, key, value, **kwargs):
    OmegaConf.update(config, key, value, **kwargs)
    return config


def config_get(config, key, default=None):
    """
    Gets a value from the config. Does not throw if the string interpolation fails or if the key is missing.
    Returns the default value in those cases.

    Returns:
        value if key exists, default otherwise
    """
    return config_select(
        config,
        key=key,
        default=default,
        throw_on_resolution_failure=False,
        throw_on_missing=False,
    )


def init_from_cfg(config: Config, *args, **kwargs):
    """Builds an object from the given configuration

    Args:
        config (OmegaConf | dict): configuration to use for instantiation
        *args: positional args passed to object's init
        **kwargs: keyword args passed to object's init

    Returns:
        _type_: Object
    """
    try:
        class_string = derive_classpath(config)
        params = derive_args(config, **kwargs)
    except KeyError:
        class_string = derive_classpath_deprecated(config)
        params = derive_args_deprecated(config, **kwargs)

    Class = class_from_string(class_string)
    return Class(*args, **params)


def config_from_toml_stream(stream, **kwargs):
    from tomlkit import loads

    conf = OmegaConf.create(loads(stream).unwrap())
    conf = config_merge(conf, dict(**kwargs))
    return conf


def config_from_toml(path: Path | str) -> Config:
    from tomlkit import load as tomlload

    with Path(path).open("r") as fp:
        cfg = OmegaConf.create(tomlload(fp).unwrap())
    return cfg


def config_merge(*args, **kwargs):
    return OmegaConf.merge(*args, **kwargs)


def config_from_path(path: Path | str) -> Config:
    path = Path(path)
    if path.suffix in [".toml"]:
        return config_from_toml(path)
    import yaml

    with path.open("r") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    return OmegaConf.create(config)


def flatten_dict(_dict: dict | list, keys_only=False, delimiter=".") -> dict | list:
    _new_dict = {}

    if isinstance(_dict, list):
        entries = enumerate(_dict)
    else:
        entries = _dict.items()

    for k, v in entries:
        if isinstance(v, (dict, list)):
            _d: dict = flatten_dict(v)
            for _k, _v in _d.items():
                _new_dict[f"{k}{delimiter}{_k}"] = _v
        else:
            _new_dict[k] = v

    if keys_only:
        ks = list(_new_dict.keys())
        ks.sort()
        return ks

    return _new_dict


def config_to_dotlist(config: Config, delimiter="."):
    """
    Flattens a config to a dictionary with dot-separated keys
    """
    if isinstance(config, dict):
        config = config_from_dict(config)
    return flatten_dict(
        dict(**OmegaConf.to_container(config, resolve=True, enum_to_str=True)),
        delimiter=delimiter,
    )


def config_dump(config: Config, path: Path):
    from tomlkit import dump as tomldump

    with path.open("w") as fp:
        c_dict = config_to_dict(config)
        c_dict = manage_invalid_values(c_dict)
        tomldump(c_dict, fp)


def manage_invalid_values(config: Config) -> dict:
    """
    Recursively removes any None values from the config while keeping structure
    and converts Path objects to strings

    Args:
        config (Config): Config to process

    Returns:
        Config: Processed dict

    """

    def _convert_types(value):
        if isinstance(value, Path):
            return str(value)
        return value

    if isinstance(config, Config):
        config = config_to_dict(config)
    out = {
        k: (manage_invalid_values(v) if isinstance(v, Config) else _convert_types(v))
        for k, v in config.items()
        if v is not None  # Remove None Values
    }

    return out


def _configs_to_toml(configs, source_dir: Path, out_dir: Path, name_fn=None):
    from tomlkit import dump as tomldump

    out_dir.mkdir(parents=True, exist_ok=True)
    for p, y in configs:
        relPath = p.relative_to(source_dir)
        if name_fn is not None:
            relPath = name_fn(relPath)
        outPath = (out_dir / relPath).with_suffix(".toml")
        outPath.parent.mkdir(parents=True, exist_ok=True)
        with outPath.open("w") as fp:
            tomldump(manage_invalid_values(y), fp)


def json_to_toml(source_dir: Path, out_dir: Path, name_fn=None):
    try:
        import json
        from pprint import pprint
    except ImportError:
        print("yaml, tomlkit required")
        return

    json_configs = [(p, json.load(p.open("r"))) for p in source_dir.rglob("**/*.json")]
    pprint(f"Found {len(json_configs)} JSON Files")
    _configs_to_toml(json_configs, source_dir, out_dir, name_fn)


def yaml_to_toml(source_dir: Path, out_dir: Path, name_fn=None):
    try:
        from pprint import pprint

        import yaml
    except ImportError:
        print("yaml, tomlkit required")
        return

    yamlConfigs = [(p, yaml.load(p.open("r"), yaml.Loader)) for p in source_dir.rglob("**/*.yaml")]
    pprint(f"Found {len(yamlConfigs)} YAML Files")
    _configs_to_toml(yamlConfigs, source_dir, out_dir, name_fn)


class ConfigDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    @classmethod
    def __dict_to_configdict__(cls, d: dict):
        d = ConfigDict(d)
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = cls.__dict_to_configdict__(v)
        return d

    def __setattr__(self, name: str, value):
        if isinstance(value, dict):
            value = self.__dict_to_configdict__(value)
        self[name] = value

    def __delattr__(self, name: str):
        del self[name]

    def to_dict(self) -> dict:
        """Convert to plain dict."""
        out = dict(**self)

        for k, v in out.items():
            # TODO: why doesn't this work?
            if isinstance(v, ConfigDict):
                out[k] = v.to_dict()
        return out

    def to_config(self):
        """Convert to OmegaConf config."""
        return config_from_dict(self.to_dict())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            if isinstance(v, dict):
                self[k] = self.__dict_to_configdict__(v)


def get_config_root(config):
    paths = []
    for k, v in config.items():
        if ("root" in k or "dir" in k) and isinstance(v, str):
            paths.append(as_path(v))
        elif isinstance(v, dict):
            paths.extend(get_config_root(v))
    return paths


def _resolve_paths(
    config,
    old_root,
    new_root=Path.cwd(),
    path_keys: list[str] | None = None,
    verbose=False,
):
    if path_keys is None:
        path_keys = []

    for full_key, v in config_to_dotlist(config).items():
        k = full_key.split(".")[-1]
        if (full_key in path_keys or "root" in k or "dir" in k or "path" in k or "file" in k) and (
            isinstance(v, Path | str)
        ):
            try:
                p = as_path(v)
                if p is not None and p.is_relative_to(old_root):
                    if verbose:
                        LOGGER.info(f"Replacing key {k}\n{p}\nwith\n{new_root / p.relative_to(old_root)}")
                    config_update(config, full_key, new_root / p.relative_to(old_root))
                    # config[k] = new_root / p.relative_to(old_root)
            except Exception as e:
                if verbose:
                    LOGGER.error(f"Could not resolve path for key {k}\n{v}\n{e}")


def resolve_paths(
    config,
    new_root=Path.cwd(),
    path_keys: list[str] | None = None,
    verbose=False,
):
    """Resolves the most likely root path in a config based on all keys that contain
    `root` or `dir` in their key name and replaces it with new_root

    The old root is determined by finding the longest common path between all paths

    Args:
        config: the config to resolve paths in
        new_root: the new root path to replace the old root path with.
            Defaults to the current working directory
        path_keys: explicitly set any extra keys to keys to resolve paths for.
            e.g. ['model.checkpoint']
            Defaults to None

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
        _resolve_paths(
            config,
            old_root=old_root,
            new_root=new_root,
            path_keys=path_keys,
            verbose=verbose,
        )
    return config
