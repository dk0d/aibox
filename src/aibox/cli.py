import datetime
from argparse import Action, ArgumentParser, Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import typer


from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf

from aibox.config import config_from_path, config_get, config_merge, config_update

from aibox.logger import get_logger
from aibox.utils import chunk

# import hydra
from aibox.utils import as_path, as_uri, basename

import functools

LOGGER = get_logger(__name__)


class CLIException(Exception):
    pass


def ExpandedPathType(s) -> Path | None:
    if s is not None:
        return Path(s).expanduser().resolve()
    return None


def cwd(path: str | None) -> Path | None:
    if path is not None:
        return ExpandedPathType(Path.cwd() / path)
    return None


class PathAction(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, ExpandedPathType(values))


def _resolve_config_path(root: Path | str, name: str):
    for ext in [".toml", ".yaml", ".yml"]:
        path = (Path(root) / name).with_suffix(ext)
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find YAML or TOML {name} in {root}")


class AIBoxCLI:
    def __init__(self, root_config: str, setup_default_args=False) -> None:
        """
        Arguments:
            root_config: relative path to the root configuration to load
            setup_default_args: create default args for the parser
                --configs_dir
                --logs_dir
                --debug
                --dry_run
        """
        self.parser = ArgumentParser()
        if setup_default_args:
            self.setup_default_args()
        self.root_config = root_config
        self.linked = []
        self.config_dirs = []

    def setup_default_args(self):
        self.parser.add_argument(
            "-cd",
            "--configs_dir",
            action=PathAction,
            default=cwd("configs"),
        )
        self.parser.add_argument(
            "-l",
            "--logs_dir",
            action=PathAction,
            default=cwd("logs"),
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            default=False,
        )
        self.parser.add_argument(
            "--dry_run",
            action="store_true",
            default=False,
        )

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def add_linked_properties(self, source: str, other: str, default: Any):
        """links two config entries
        assumes the first is the priority. if no value is set, default is used. if default is set to None,
        nothing will be set when finding p2 doesn't exist or is already set to None

        Args:
            source (_type_): dot key
            other (_type_): dot key
            default (_type_): default value
        """
        self.linked.append((source, other, default))

    def resolve_linked_props(self, config):
        for src, other, default in self.linked:
            srcVal = OmegaConf.select(config, src, default=None)
            otherVal = OmegaConf.select(config, other, default=None)

            if srcVal is None and default is None:
                continue
            elif srcVal is None:
                OmegaConf.update(config, src, default, force_add=True)

            if otherVal is None or srcVal != otherVal:
                OmegaConf.update(config, other, OmegaConf.select(config, src), force_add=True)

    def _args_to_config(self, args: Namespace | list):
        if isinstance(args, list):
            _unk = [u.split("=") for u in args]
            unk = []
            for u in _unk:
                unk.extend(u)
            if len(unk) % 2 != 0:
                LOGGER.error("[bold red]Only key-value pair arguments are supported for OmegaConf")
                raise CLIException("Only key-value pair arguments are supported for OmegaConf")
            args_dict: dict = {k: v for k, v in chunk(unk, 2)}
        else:
            args_dict: dict = vars(args)
        dotlist = [f"{k.lstrip('-')}={v}" for k, v in args_dict.items()]
        conf = OmegaConf.from_dotlist(dotlist)
        # Ensure None values instead of a string 'None'
        for k, v in args_dict.items():
            if v is not None:
                continue
            OmegaConf.update(conf, k, None, force_add=True)
        return conf

    def _load_config(self, root: Path | str, name: str, custom_msg=None, verbose=False):
        try:
            path = _resolve_config_path(root, name)
            config = config_from_path(path)
            return path, config
        except Exception as e:
            if verbose:
                errormsg = (
                    f"[bold red]Error loading {custom_msg} {name}: {e}"
                    if custom_msg is not None
                    else f"[bold red]Error loading {name}: {e}"
                )
                LOGGER.error(errormsg)
                LOGGER.exception(e)
        return None, OmegaConf.from_dotlist([])

    def _resolve_links(self, config) -> DictConfig:
        config = OmegaConf.create(OmegaConf.to_object(config))
        assert isinstance(config, DictConfig), f"Expected DictConfig, got {type(config)}"
        self.resolve_linked_props(config)
        config.created = datetime.datetime.now().isoformat()
        return config

    def _parse_cli_args(self, args=None):
        args, unk = self.parser.parse_known_args(args)

        cli_config = self._args_to_config(args)

        if len(unk) > 0:
            cli_config = OmegaConf.merge(cli_config, self._args_to_config(unk))

        return cli_config

    def resolve_config_from_root_name(self, root: Path | str, name: str, verbose=False):
        root = Path(root)

        # Look for root/name directory first
        if name is not None and (root / name).exists() and (root / name).is_dir():
            root = root / name

        # Load config, if present
        config_path, config = self._load_config(root=root, name=name, custom_msg=name, verbose=verbose)

        # Load defaults, if present
        defaults_root = config_path.parent if config_path is not None else root
        _, defaults = self._load_config(
            root=defaults_root, name="defaults", custom_msg=f"{name} defaults", verbose=False
        )

        # Merge defaults and config
        config = OmegaConf.merge(defaults, config)

        return config_path, config

    def parse_args(self, args=None) -> DictConfig:
        cli_config = self._parse_cli_args(args)

        # Load global default config
        _, global_defaults = self._load_config(
            root=cli_config.config_dir,
            name=self.root_config,
            custom_msg="root configuration",
            verbose=True,
        )

        # Load overriding config if given
        _config = [OmegaConf.from_dotlist([])]
        if config_get(cli_config, "config", None) is not None:
            try:
                _config = [config_from_path(_resolve_config_path(cli_config.config_dir, c)) for c in cli_config.config]
            except Exception as e:
                LOGGER.error(f"[bold red]Error loading config {cli_config.config}")
                LOGGER.exception(e)

            # remove the passed configs from the config to clean things up
            del cli_config.config

        config = OmegaConf.merge(
            global_defaults,
            *_config,
            cli_config,
        )
        return self._resolve_links(config)


# def cli_main(args=None):
#     cli = AIBoxCLI()
#     # First key (source) takes priority
#     # cli.add_linked_properties("model.args.image_size", "data.args.image_size", default=64)
#     # cli.add_linked_properties("model.args.image_channels", "data.args.image_channels", default=1)
#     config = cli.parse_args(args=args)
#     return config

MainFn = Callable[[Any], Any]


def process_cfg(configs_dir: Path, config):
    config_root_keys = [p.name for p in configs_dir.glob("*") if p.is_dir()]
    cfg = config
    for key in config_root_keys:
        if hasattr(cfg, key) and isinstance(
            config_get(cfg, key, None), str
        ):  # interpret value as path to another config
            other_cfg = config_from_path(_resolve_config_path(configs_dir / key, config_get(cfg, key)))
            if hasattr(other_cfg, key):  # if same key exists, merge
                cfg: DictConfig = config_merge(cfg, other_cfg)
                # pass
            else:  # assume its a flattened config that only contains the key config
                config_update(cfg, key, other_cfg)
        else:  # can be a dict or object leave as is
            pass
    return cfg


def main(
    configs_dir: str | Path = "./configs",
    root_config: str = "defaults",
) -> Callable[[MainFn], Any]:
    """
    configs_dir: path to config root
    entry_config: the default root config - if not provided, looks for a `defaults.{toml,yaml}` file in the root config directory
    """
    configs_dir = as_path(configs_dir) if configs_dir is not None else as_path("./configs")

    def main_wrapper(fn: MainFn) -> Callable[[], None]:
        @functools.wraps(fn)
        def wrapped_main(config: DictConfig | None = None) -> Any:
            def resolve_cfg(path: str, key=None):
                """
                if key is not None - will return the provided key from the referenced config
                this is defined here to capture the value of the `configs_dir`
                """
                fpath = _resolve_config_path(configs_dir, path)
                _conf = config_from_path(fpath)
                if key is None:
                    return _conf
                config_get(_conf, key, None)

            OmegaConf.register_new_resolver("cfg", resolve_cfg)
            OmegaConf.register_new_resolver("as_path", as_path)
            OmegaConf.register_new_resolver("as_uri", as_uri)
            OmegaConf.register_new_resolver("add", lambda *nums: sum(nums))
            OmegaConf.register_new_resolver("basename", basename)

            if isinstance(config, DictConfig):
                # if a config is already given, just forward it to the function
                return fn(config)
            else:
                # if no cfg is given, assume need to parse CLI args
                parser = AIBoxCLI(root_config=root_config)
                parser.add_argument("-cd", "--config_dir", action=PathAction, default=as_path(configs_dir))
                parser.add_argument("-c", "--config", default=None, nargs="+")

                config = parser.parse_args(config)
                config = process_cfg(configs_dir, config)
                return fn(config)

        return wrapped_main

    return main_wrapper


# if __name__ == "__main__":
#     cli_main()
