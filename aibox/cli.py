from argparse import ArgumentParser, Action, Namespace
from pathlib import Path
from omegaconf import OmegaConf
from typing import Any, Sequence
import rich
import datetime


from .config import config_from_toml


class CLIException(Exception):
    pass


def ExpandedPathType(s):
    return Path(s).expanduser().resolve()


def cwd(path: str):
    return ExpandedPathType(Path.cwd() / path)


class PathAction(Action):
    def __call__(
        self,
        parser: ArgumentParser,
        namespace: Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        setattr(namespace, self.dest, ExpandedPathType(values))


def _chunk(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


class AIBoxCLI:
    def __init__(self) -> None:
        self.parser = ArgumentParser()
        self.setup_default_args()
        self.linked = []

    def setup_default_args(self):
        self.parser.add_argument("-e", "--exp_name", type=str)
        self.parser.add_argument("-m", "--model_name", type=str)
        self.parser.add_argument("-c", "--config", type=str)
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
            default=cwd("configs/experiments"),
        )
        self.parser.add_argument(
            "-md",
            "--models_dir",
            action=PathAction,
            default=cwd("configs/models"),
        )
        self.parser.add_argument("--debug", action="store_true", default=False)

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
                rich.print("[bold red]Only key-value pair arguments are supported for OmegaConf")
                raise CLIException("Only key-value pair arguments are supported for OmegaConf")
            args = {k: v for k, v in _chunk(unk, 2)}
        else:
            args = vars(args)
        dotlist = [f"{k.lstrip('-')}={v}" for k, v in args.items()]
        conf = OmegaConf.from_dotlist(dotlist)
        # Ensure None values instead of a string 'None'
        for k, v in args.items():
            if v is not None:
                continue
            OmegaConf.update(conf, k, None, force_add=True)
        return conf

    def parse_args(self, args=None) -> OmegaConf:
        args, unk = self.parser.parse_known_args(args)
        cli_config = self._args_to_config(args)
        if len(unk) > 0:
            cli_config = OmegaConf.merge(cli_config, self._args_to_config(unk))

        try:
            model_config = config_from_toml(args.models_dir / f"{args.model_name}.toml")
        except Exception as e:
            if args.model_name is not None:
                rich.print(f"[bold red]Error loading model {args.model_name}: {e}")
            model_config = OmegaConf.from_dotlist([])

        try:
            modelPath = args.models_dir / f"{args.model_name}.toml"
            model_defaults_config = config_from_toml(modelPath.parent / "default.toml")
        except Exception as e:
            if args.model_name is not None:
                rich.print(f"[bold red]Error loading model defaults {args.model_name}: {e}")
            model_defaults_config = OmegaConf.from_dotlist([])

        try:
            expPath = args.exp_dir / f"{args.exp_name}.toml"
            exp_defaults_config = config_from_toml(expPath.parent / "default.toml")
        except Exception as e:
            if args.model_name is not None:
                rich.print(f"[bold red]Error loading experiment defaults {args.exp_name}: {e}")
            exp_defaults_config = OmegaConf.from_dotlist([])

        try:
            experiment_config = config_from_toml(args.exp_dir / f"{args.exp_name}.toml")
        except Exception as e:
            if args.exp_name is not None:
                rich.print(f"[bold red]Error loading experiment {args.exp_name}: {e}")
            experiment_config = OmegaConf.from_dotlist([])

        try:
            _config = config_from_toml(args.config)
        except Exception as e:
            if args.config is not None:
                rich.print(f"[bold red]Error loading config {args.config}: {e}")
            _config = OmegaConf.from_dotlist([])

        config = OmegaConf.merge(
            _config,
            model_defaults_config,
            model_config,
            exp_defaults_config,
            experiment_config,
            cli_config,
        )
        config = OmegaConf.create(OmegaConf.to_object(config))
        self.resolve_linked_props(config)

        config.created = datetime.datetime.now().isoformat()

        return config


def cli_main(args=None):
    cli = AIBoxCLI()

    # First key (source) takes priority
    cli.add_linked_properties("model.args.image_size", "data.args.image_size", default=64)
    cli.add_linked_properties("model.args.image_channels", "data.args.image_channels", default=1)

    config = cli.parse_args(args=args)

    return config


if __name__ == "__main__":
    cli_main()
