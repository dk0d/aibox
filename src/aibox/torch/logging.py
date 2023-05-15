from argparse import Namespace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from aibox.config import config_to_dotlist

try:
    import mlflow

    from mlflow.client import MlflowClient
    from pytorch_lightning.loggers import MLFlowLogger
    from tensorboardX import SummaryWriter
except ImportError:
    print("MLFlow not installed, CombinedLogger will not be available.")
    mlflow = None
    MLFlowLogger = None

if mlflow is not None:

    class CombinedLogger(MLFlowLogger):
        """
        A logger that logs to both MLFlow and Tensorboard

        Tensorboard logs are stored in the `logs` folder
        and sent to MLFlow as an artifact.

        """

        @property
        def experiment_name(self) -> str:
            return self._experiment_name

        @property
        def mlflow_client(self) -> MlflowClient:
            return self.experiment

        @property
        def tb_writer(self) -> SummaryWriter:
            return self.writer

        def __init__(self, tensorboard_logdir=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tensorboard_logdir = tensorboard_logdir or Path("logs")
            self.writer = SummaryWriter(
                logdir=(tensorboard_logdir / self.experiment_id / self.run_id).expanduser().resolve().as_posix()
            )

        def log_hyperparams(self, params: dict[str, Any] | Namespace | DictConfig) -> None:
            if isinstance(params, DictConfig):
                container = dict(**OmegaConf.to_container(params, resolve=True))
                # log config as an artifact for easier instantiation of model from the config
                self.experiment.log_dict(self.run_id, container, "config.yml")
                params = dict(**config_to_dotlist(params))
            super().log_hyperparams(params)
