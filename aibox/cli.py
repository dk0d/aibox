from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from .config import config_from_toml


def ExpandedPathType(s):
    return Path(s).expanduser().resolve()


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
            "-cd", "--config_dir", type=ExpandedPathType, default=Path.cwd() / "configs"
        )
        self.parser.add_argument(
            "-l", "--log_dir", type=ExpandedPathType, default=Path.cwd() / "logs"
        )
        self.parser.add_argument(
            "-d",
            "--defaults",
            type=ExpandedPathType,
            default=None,
        )
        self.parser.add_argument(
            "-ed",
            "--exp_dir",
            type=ExpandedPathType,
            default=None,
        )
        self.parser.add_argument(
            "-md",
            "--models_dir",
            type=ExpandedPathType,
            default=None,
        )

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
                OmegaConf.update(
                    config, other, OmegaConf.select(config, src), force_add=True
                )
                

    def parse_args(self, args=None) -> OmegaConf:
        args, unk = self.parser.parse_known_args(args)
        cli_config = OmegaConf.from_cli([f.lstrip("-") for f in unk])
        config_dir = args.config_dir
        defaults_path = (
            config_dir / "default.toml" if args.defaults is None else args.defaults
        )
        exp_dir = config_dir / "experiments" if args.exp_dir is None else args.exp_dir
        models_dir = (
            config_dir / "models" if args.models_dir is None else args.models_dir
        )
        try:
            defaults_config = config_from_toml(defaults_path)
        except:
            defaults_config = OmegaConf.from_dotlist([])

        try:
            model_config = config_from_toml(models_dir / f"{args.model_name}.toml")
        except:
            model_config = OmegaConf.from_dotlist([])

        try:
            experiment_config = config_from_toml(exp_dir / f"{args.exp_name}.toml")
        except:
            experiment_config = OmegaConf.from_dotlist([])

        try:
            _config = config_from_toml(args.config)
        except:
            _config = OmegaConf.from_dotlist([])

        config = OmegaConf.merge(
            defaults_config, _config, model_config, experiment_config, cli_config
        )
        config = OmegaConf.create(OmegaConf.to_object(config))
        self.resolve_linked_props(config)
        return config


def cli_main():
    cli = AIBoxCLI()

    # Second key takes priority
    cli.add_linked_properties(
        "model.args.image_size", "data.args.image_size", default=64
    )
    cli.add_linked_properties(
        "model.args.image_channels", "data.args.image_channels", default=1
    )

    config = cli.parse_args()

    return config


if __name__ == "__main__":
    cli_main()
