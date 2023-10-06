import datetime
from argparse import Action, ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Sequence

import rich
from omegaconf import DictConfig, OmegaConf

from aibox.config import config_from_path
from aibox.logger import get_logger
from aibox.utils import chunk

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


class AIBoxCLI:
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.setup_default_args()
        self.linked = []
        self.config_dirs = []

    def setup_default_args(self):
        self.parser.add_argument(
            "-e",  # expe
            "-n",
            "--name",
            dest="name",
            type=str,
            help="Name of the experiment, overall name of the configuration",
        )
        self.parser.add_argument(
            "-m",
            "--model_name",
            type=str,
            help="Name of the model, used to load model config",
        )
        self.parser.add_argument(
            "-c",
            "--config",
            action=PathAction,
            default=None,
            help="Path to any config file to override experiment config",
        )
        self.parser.add_argument(
            "-cd",
            "--config_dir",
            action=PathAction,
            default=cwd("configs"),
        )
        self.parser.add_argument(
            "-l",
            "--log_dir",
            action=PathAction,
            default=cwd("logs"),
        )
        self.parser.add_argument(
            "-ed",
            "--exp_dir",
            action=PathAction,
            default=None,
            # default=cwd("configs/experiments"),
        )
        self.parser.add_argument(
            "-md",
            "--models_dir",
            action=PathAction,
            # default=cwd("configs/models"),
            default=None,
        )
        self.parser.add_argument(
            "--debug",
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

    def _resolve_config_path(self, root: Path | str, name: str):
        for ext in [".toml", ".yaml", ".yml"]:
            path = (Path(root) / name).with_suffix(ext)
            if path.exists():
                return path
        raise FileNotFoundError(f"Could not find YAML or TOML {name} in {root}")

    def _load_config(self, root: Path | str, name: str, custom_msg=None, verbose=False):
        try:
            path = self._resolve_config_path(root, name)
            config = config_from_path(path)
            return path, config
        except Exception as e:
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

        try:
            if args.exp_dir is None:
                args.exp_dir = args.config_dir / "experiments"

            if args.models_dir is None:
                args.models_dir = args.config_dir / "models"
        except Exception as e:
            pass

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
            root=defaults_root, name="default", custom_msg=f"{name} defaults", verbose=False
        )

        # Merge defaults and config
        config = OmegaConf.merge(defaults, config)

        return config_path, config

    def parse_args(self, args=None) -> DictConfig:
        cli_config = self._parse_cli_args(args)

        # model_path, model_config = Path(cli_config.models_dir) / f"{cli_config.model_name}.toml", None
        # model_defaults_path, model_defaults_config = model_path.parent / "default.toml", None
        # exp_path, exp_config = Path(cli_config.exp_dir) / f"{cli_config.name}.toml", None
        # exp_defaults_path, exp_defaults_config = exp_path.parent / "default.toml", None

        # Load global default config
        _, global_defaults = self._load_config(
            root=cli_config.config_dir,
            name="default",
            custom_msg="global defaults",
            verbose=True,
        )

        # Load Model Config
        _, model_config = self.resolve_config_from_root_name(
            root=cli_config.models_dir,
            name=cli_config.model_name,
            verbose=cli_config.model_name is not None,
        )

        # Load Experiment Config
        _, exp_config = self.resolve_config_from_root_name(
            root=cli_config.exp_dir,
            name=cli_config.name,
            verbose=cli_config.name is not None,
        )

        # Load overriding config if given
        _config = OmegaConf.from_dotlist([])
        if cli_config.config is not None:
            try:
                _config = config_from_path(cli_config.config)
            except Exception as e:
                LOGGER.error(f"[bold red]Error loading config {cli_config.config}")
                LOGGER.exception(e)

        config = OmegaConf.merge(
            global_defaults,
            exp_config,
            model_config,
            _config,
            cli_config,
        )
        return self._resolve_links(config)


def cli_main(args=None):
    cli = AIBoxCLI()

    # First key (source) takes priority
    cli.add_linked_properties("model.args.image_size", "data.args.image_size", default=64)
    cli.add_linked_properties("model.args.image_channels", "data.args.image_channels", default=1)

    config = cli.parse_args(args=args)

    return config


if __name__ == "__main__":
    cli_main()
