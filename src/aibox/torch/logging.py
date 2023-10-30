from argparse import Namespace
from typing import Any

from omegaconf import DictConfig, OmegaConf

from aibox.config import config_to_dotlist
from aibox.utils import as_path

try:
    import ray.train as ray_train
except ImportError:
    ray_train = None

try:
    import shutil

    import numpy as np
    import torch
    import torchvision
    from lightning.fabric.loggers.logger import rank_zero_experiment
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
    from lightning.pytorch.loggers.mlflow import LOCAL_FILE_URI_PREFIX
    from lightning.pytorch.loggers.utilities import _scan_checkpoints
    from lightning_utilities.core.rank_zero import rank_zero_only
    from mlflow.client import MlflowClient
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
        @rank_zero_only
        def has_tb_logger(self) -> bool:
            return self._tb_logger is not None

        @property
        @rank_zero_experiment
        def tb_writer(self) -> SummaryWriter | None:
            if self._tb_logger is not None:
                return self._tb_logger.experiment
            return None

        @property
        @rank_zero_experiment
        def log_dir(self) -> str:
            return self.experiment.tracking_uri

        @property
        @rank_zero_experiment
        def run_name(self) -> str:
            return self.experiment.get_run(self.run_id).info.run_name

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

        def __init__(
            self,
            local_log_root=None,
            tb_log_graph=True,
            enable_tb_logging=False,
            **kwargs,
        ):
            self.local_log_root = as_path(local_log_root or "logs")
            self.enable_tb_logging = enable_tb_logging
            self._tensorboard_logdir = (self.local_log_root / "tbruns").expanduser().resolve()
            if "tracking_uri" not in kwargs:
                mlflow_logdir = f"file:{self.local_log_root / 'mlruns'}"
                kwargs.update(tracking_uri=mlflow_logdir)

            if ray_train is not None:
                if kwargs.get("run_name", None) is None:
                    trial_name = ray_train.get_context().get_trial_name()
                    if trial_name is not None:
                        kwargs.update(run_name=trial_name)

                    try:
                        run_id = (as_path(ray_train.get_context().get_trial_dir()) / "run_id.txt").read_text()
                        kwargs.update(run_id=run_id)
                    except Exception:
                        pass

            super().__init__(**kwargs)

            if self.enable_tb_logging:
                self._tb_logger = TensorBoardLogger(
                    save_dir=self._tensorboard_logdir.as_posix(),
                    name=self.experiment_id,
                    version=self.run_id,
                    log_graph=tb_log_graph,
                )
            else:
                self._tensorboard_logdir = None
                self._tb_logger = None

        @rank_zero_only
        def log_hyperparams(self, params: dict[str, Any] | Namespace | DictConfig) -> None:
            if isinstance(params, DictConfig):
                container = dict(**OmegaConf.to_container(params, resolve=True))  # type: ignore
                # log config as an artifact for easier instantiation of model from the config
                self.experiment.log_dict(self.run_id, container, "config.yml")
                params = dict(**config_to_dotlist(params))  # type: ignore
            super().log_hyperparams(params)

        @rank_zero_only
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            super().log_metrics(metrics, step)
            if self.enable_tb_logging and self._tb_logger is not None:
                self._tb_logger.log_metrics(metrics, step)

        @rank_zero_only
        def log_artifact(
            self,
            local_path: str,
            artifact_path=None,
        ):
            self.experiment.log_artifact(self.run_id, local_path, artifact_path)

        @rank_zero_only
        def log_artifacts(
            self,
            local_path: str,
            artifact_path=None,
        ):
            self.experiment.log_artifacts(self.run_id, local_path, artifact_path)

        @rank_zero_only
        def log_graph(self, *args, **kwargs):
            if self.enable_tb_logging and self._tb_logger is not None:
                return self._tb_logger.log_graph(*args, **kwargs)

        @rank_zero_only
        def log_image(
            self,
            tag: str,
            image: np.ndarray | torch.Tensor,
            global_step=None,
            dataformats="CHW",
        ):
            if self.enable_tb_logging and self._tb_logger is not None:
                self.tb_writer.add_image(tag, image, global_step, dataformats)

            if isinstance(image, torch.Tensor):
                image = torchvision.transforms.ToPILImage()(image.detach().cpu())

            # run_id: str, image: Union["numpy.ndarray", "PIL.Image.Image"], artifact_file: str
            self.mlflow_client.log_image(
                self.run_id,
                image,
                artifact_file=f"{tag}.png",
            )

        @rank_zero_only
        def log_tags(self, tags: dict[str, Any]):
            for tag, value in tags.items():
                self.experiment.set_tag(self.run_id, tag, value)

        @rank_zero_only
        def finalize(self, status: str = "success") -> None:
            super().finalize(status)
            if self.enable_tb_logging and self._tb_logger is not None:
                self._tb_logger.finalize(status)
                if as_path(self._tb_logger.log_dir).exists():
                    self.experiment.log_artifacts(
                        self.run_id,
                        as_path(self._tb_logger.log_dir).as_posix(),
                        "tensorboard_events",
                    )

        def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
            # get checkpoints to be saved with associated score
            super()._scan_and_log_checkpoints(checkpoint_callback)
            try:
                checkpoints = _scan_checkpoints(checkpoint_callback, {})
                if self.save_dir is not None:
                    for _, p, _, _ in checkpoints:
                        p = as_path(p)
                        if p.is_relative_to(self.save_dir) and p in self._logged_model_time:
                            p.unlink()
                    parent = list(set(as_path(p).parent for _, p, _, _ in checkpoints))[0]
                    if parent.is_dir() and not list(parent.iterdir()):
                        shutil.rmtree(parent)
            except Exception:
                pass

except ImportError as e:
    print(f"Logging import error: {e}")
