import importlib
from pathlib import Path
from omegaconf import OmegaConf


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


def build_from_cfg(config: OmegaConf | dict, *args, **kwargs):
    """Builds an object from the given configuration

    Args:
        config (OmegaConf | dict): configuration to use for instantiation
        *args: positional args passed to object's init
        **kwargs: keyword args passed to object's init

    Returns:
        _type_: Object
    """
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate object")
    Class = class_from_string(config["class_path"])
    params = config.get("args", dict())
    params.update(**kwargs)
    return Class(*args, **params)


def config_from_toml_stream(stream):
    from tomlkit import loads

    return OmegaConf.create(loads(stream).unwrap())


def config_from_toml(path: Path | str) -> OmegaConf:
    from tomlkit import load as tomlload

    with Path(path).open("r") as fp:
        cfg = OmegaConf.create(tomlload(fp).unwrap())
    return cfg


def config_from_path(path: Path | str) -> OmegaConf:
    if path.suffix in [".toml"]:
        return config_from_toml(path)
    import yaml

    with path.open("r") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)
    return OmegaConf.create(config)


def config_dump(config: OmegaConf, path: Path):
    from tomlkit import dump as tomldump

    with path.open("w") as fp:
        c = OmegaConf.to_container(config, resolve=True)
        tomldump(c, fp)


def yaml_to_toml(source_dir: Path, out_dir: Path, name_fn=None):

    try:
        from pprint import pprint

        import yaml
        from tomlkit import dump as tomldump
    except ImportError:
        print("yaml, tomlkit required")
        return

    yamlConfigs = [
        (p, yaml.load(p.open("r"), yaml.Loader)) for p in source_dir.rglob("**/*.yaml")
    ]
    pprint(f"Found {len(yamlConfigs)} YAML Files")

    out_dir.mkdir(parents=True, exist_ok=True)
    for p, y in yamlConfigs:
        relPath = p.relative_to(source_dir)
        if name_fn is not None:
            relPath = name_fn(relPath)
        outPath = (out_dir / relPath).with_suffix(".toml")
        outPath.parent.mkdir(parents=True, exist_ok=True)
        with outPath.open("w") as fp:
            tomldump(y, fp)
