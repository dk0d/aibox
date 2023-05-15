from argparse import Namespace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from aibox.config import config_to_dotlist

try:
    import mlflow

    from mlflow.client import MlflowClient
    from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
    from torch.utils.tensorboard.writer import SummaryWriter
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
            return self._tb_logger.experiment

        @property
        def log_dir(self) -> str:
            return self.experiment.tracking_uri

        def __init__(self, root_log_dir=None, **kwargs):
            self.root_log_dir = Path(root_log_dir or "logs")
            tensorboard_logdir = self.root_log_dir / "tbruns"
            if "tracking_uri" not in kwargs:
                mlflow_logdir = f"file:{self.root_log_dir / 'mlruns'}"
                kwargs.update(tracking_uri=mlflow_logdir)

            super().__init__(**kwargs)

            self._tb_logger = TensorBoardLogger(
                tensorboard_logdir.expanduser().resolve().as_posix(),
                name=self.experiment_id,
                version=self.run_id,
                log_graph=True,
            )

            # self.writer = SummaryWriter(
            #     logdir=(tensorboard_logdir / self.experiment_id / self.run_id).expanduser().resolve().as_posix(),
            #     log_graph=True,
            # )

        def log_hyperparams(self, params: dict[str, Any] | Namespace | DictConfig) -> None:
            if isinstance(params, DictConfig):
                container = dict(**OmegaConf.to_container(params, resolve=True))
                # log config as an artifact for easier instantiation of model from the config
                self.experiment.log_dict(self.run_id, container, "config.yml")
                params = dict(**config_to_dotlist(params))
            super().log_hyperparams(params)

        def log_graph(self, *args, **kwargs):
            return self._tb_logger.log_graph(*args, **kwargs)
