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

    def _load_config(self, path: Path, custom_msg=None, verbose=False):
        try:
            config = config_from_toml(path)
            return config
        except Exception as e:
            if verbose:
                errormsg = (
                    f"[bold red]Error loading {custom_msg} {path.name}: {e}"
                    if custom_msg is None
                    else f"[bold red]Error loading {path.name}: {e}"
                )
                rich.print(errormsg)
        return OmegaConf.from_dotlist([])

    def _resolve(self, config):
        config = OmegaConf.create(OmegaConf.to_object(config))
        self.resolve_linked_props(config)
        config.created = datetime.datetime.now().isoformat()
        return config

    def _get_cli_config(self, args=None):
        args, unk = self.parser.parse_known_args(args)
        cli_config = self._args_to_config(args)
        if len(unk) > 0:
            cli_config = OmegaConf.merge(cli_config, self._args_to_config(unk))
        return cli_config

    def parse_args(self, args=None) -> OmegaConf:
        cli_config = self._get_cli_config(args)

        model_path, model_config = Path(cli_config.models_dir) / f"{cli_config.model_name}.toml", None
        model_defaults_path, model_defaults_config = model_path.parent / "default.toml", None
        exp_path, exp_config = Path(cli_config.exp_dir) / f"{cli_config.exp_name}.toml", None
        exp_defaults_path, exp_defaults_config = exp_path.parent / "default.toml", None

        # Load default model config if present
        model_defaults_config = self._load_config(model_defaults_path, custom_msg="model defaults")

        # Load model config
        model_config = self._load_config(model_path, custom_msg="model", verbose=cli_config.model_name is not None)

        # Load default experiment config if present
        exp_defaults_config = self._load_config(exp_defaults_path, custom_msg="experiment defaults")

        # Load experiment config
        exp_config = self._load_config(exp_path, custom_msg="experiment", verbose=cli_config.exp_name is not None)

        # Load overriding config if given
        _config = self._load_config(cli_config.config, custom_msg="config", verbose=cli_config.config is not None)

        config = OmegaConf.merge(
            _config,
            model_defaults_config,
            model_config,
            exp_defaults_config,
            exp_config,
            cli_config,
        )
        return self._resolve(config)


def cli_main(args=None):
    cli = AIBoxCLI()

    # First key (source) takes priority
    cli.add_linked_properties("model.args.image_size", "data.args.image_size", default=64)
    cli.add_linked_properties("model.args.image_channels", "data.args.image_channels", default=1)

    config = cli.parse_args(args=args)

    return config


if __name__ == "__main__":
    cli_main()
