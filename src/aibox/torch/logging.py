from argparse import Namespace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# from lightning_fabric.fabric import rank_zero_experiment

from aibox.config import config_to_dotlist

try:
    from mlflow.client import MlflowClient
    from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
    from pytorch_lightning.loggers.mlflow import LOCAL_FILE_URI_PREFIX
    from lightning_fabric.loggers.logger import rank_zero_experiment, rank_zero_only
    from torch.utils.tensorboard.writer import SummaryWriter

    class CombinedLogger(MLFlowLogger):
        """
        A logger that logs to both MLFlow and Tensorboard

        Tensorboard logs are stored in the `logs` folder
        and sent to MLFlow as an artifact.

        """

        @property
        @rank_zero_experiment
        def experiment_name(self) -> str:
            return self._experiment_name

        @property
        @rank_zero_experiment
        def mlflow_client(self) -> MlflowClient:
            return self.experiment

        @property
        @rank_zero_experiment
        def tb_writer(self) -> SummaryWriter:
            return self._tb_logger.experiment

        @property
        @rank_zero_experiment
        def log_dir(self) -> str:
            return self.experiment.tracking_uri

        @property
        def save_dir(self) -> str | None:
            """The root file directory in which MLflow experiments are saved.
            fixes error in MLFlowLogger that uses lstrip and removes extra characters

            Return:
                Local path to the root experiment directory if the tracking uri is local.
                Otherwise returns `None`.
            """
            if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
                return self._tracking_uri.removeprefix(LOCAL_FILE_URI_PREFIX)
            return None

        def __init__(self, root_log_dir=None, tb_log_graph=True, **kwargs):
            self.root_log_dir = Path(root_log_dir or "logs")
            self._tensorboard_logdir = (self.root_log_dir / "tbruns").expanduser().resolve()
            if "tracking_uri" not in kwargs:
                mlflow_logdir = f"file:{self.root_log_dir / 'mlruns'}"
                kwargs.update(tracking_uri=mlflow_logdir)

            super().__init__(**kwargs)

            self._tb_logger = TensorBoardLogger(
                save_dir=self._tensorboard_logdir.as_posix(),
                name=self.experiment_id,
                version=self.run_id,
                log_graph=tb_log_graph,
            )

        @rank_zero_only
        def log_hyperparams(self, params: dict[str, Any] | Namespace | DictConfig) -> None:
            if isinstance(params, DictConfig):
                container = dict(**OmegaConf.to_container(params, resolve=True))
                # log config as an artifact for easier instantiation of model from the config
                self.experiment.log_dict(self.run_id, container, "config.yml")
                params = dict(**config_to_dotlist(params))
            super().log_hyperparams(params)

        @rank_zero_only
        def log_graph(self, *args, **kwargs):
            return self._tb_logger.log_graph(*args, **kwargs)

        @rank_zero_only
        def finalize(self, status: str = "success") -> None:
            super().finalize(status)
            self._tb_logger.finalize(status)
            if Path(self._tb_logger.log_dir).exists():
                self.experiment.log_artifacts(self.run_id, self._tb_logger.log_dir, "tensorboard_events")

except ImportError:
    print("MLFlow not installed, CombinedLogger will not be available.")
    mlflow = None
    MLFlowLogger = None
