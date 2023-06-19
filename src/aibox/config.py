import importlib
from pathlib import Path
from omegaconf import OmegaConf
from omegaconf import DictConfig, ListConfig

KeyConfig = DictConfig | dict
Config = DictConfig | ListConfig | dict


def print_config(config):
    import rich
    import json

    rich.print_json(json.dumps(OmegaConf.to_container(config)))


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


def init_from_cfg(config: Config, *args, **kwargs):
    """Builds an object from the given configuration

    Args:
        config (OmegaConf | dict): configuration to use for instantiation
        *args: positional args passed to object's init
        **kwargs: keyword args passed to object's init

    Returns:
        _type_: Object
    """

    if "class_path" not in config:
        raise KeyError("Expected key `class_path` to instantiate object")

    Class = class_from_string(config["class_path"])
    params = config.get("args", dict())
    for key in ["kwds", "kwargs", "init_args"]:
        params.update(**config.get(key, dict()))
    params.update(**kwargs)
    return Class(*args, **params)


def config_from_toml_stream(stream):
    from tomlkit import loads

    return OmegaConf.create(loads(stream).unwrap())


def config_from_toml(path: Path | str) -> Config:
    from tomlkit import load as tomlload

    with Path(path).open("r") as fp:
        cfg = OmegaConf.create(tomlload(fp).unwrap())
    return cfg


def config_from_path(path: Path | str) -> Config:
    path = Path(path)
    if path.suffix in [".toml"]:
        return config_from_toml(path)
    import yaml

    with path.open("r") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    return OmegaConf.create(config)


def flatten_dict(_dict: dict | list, keys_only=False, delimiter=".") -> dict:
    _new_dict = {}

    if isinstance(_dict, list):
        entries = enumerate(_dict)
    else:
        entries = _dict.items()

    for k, v in entries:
        if isinstance(v, (dict, list)):
            _d = flatten_dict(v)
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
    return flatten_dict(dict(**OmegaConf.to_container(config, resolve=True, enum_to_str=True)), delimiter=delimiter)


def config_dump(config: OmegaConf, path: Path):
    from tomlkit import dump as tomldump

    with path.open("w") as fp:
        c = OmegaConf.to_container(config, resolve=True)
        c = dict(OmegaConf.to_object(c))
        tomldump(c, fp)


def remove_any_none_values(config: Config):
    """
    Recursively removes any None values from the config while keeping structure
    """
    if isinstance(config, Config):
        config = OmegaConf.to_object(config)
    out = {k: (remove_any_none_values(v) if isinstance(v, Config) else v) for k, v in config.items() if v is not None}
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
            tomldump(remove_any_none_values(y), fp)


def json_to_toml(source_dir: Path, out_dir: Path, name_fn=None):
    try:
        from pprint import pprint

        import json
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