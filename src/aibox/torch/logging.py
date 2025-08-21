from argparse import Namespace
import logging
import os
from pathlib import Path
from typing import Any, Literal, Mapping
import uuid

from omegaconf import DictConfig, OmegaConf

from ueid.utils.config import config_to_dotlist
from ueid.utils import as_path, as_uri


try:
    import ray.train as ray_train  # pyright: ignore
except ImportError:
    ray_train = None


class BestMonitor:
    def __init__(self, name: str, op: Literal["min", "max"]):
        monitor_ops = {"min": torch.lt, "max": torch.gt}
        self.op = monitor_ops[op]
        self.best = None

    def __call__(self, value):
        """
        Returns true if the value is changed
        """
        if value is None:
            return False
        if self.best is None or self.op(value if torch.is_tensor(value) else torch.tensor(value), self.best):
            self.best = value
            return True
        return False


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
        def tracking_uri(self) -> str:
            return self._tracking_uri

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
        @rank_zero_experiment
        def run(self) -> str:
            return self.experiment.get_run(self.run_id)

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
            local_log_uri,
            artifact_location,
            tb_log_graph=True,
            enable_tb_logging=False,
            enable_ray_train=False,
            track_best_of_metrics: dict[str, Literal["min", "max"]] = {},
            **kwargs,
        ):
            self.local_log_uri = as_uri(local_log_uri)
            self.artifact_location = artifact_location
            self.enable_tb_logging = enable_tb_logging
            self.track_best_of_metrics = track_best_of_metrics
            self.monitors: dict[str, BestMonitor] = {}

            self._tensorboard_logdir = (
                as_path(self.local_log_uri) / "tbruns"
                if self.local_log_uri is not None and self.enable_tb_logging
                else as_path("./logs/tbruns")
            )
            if "tracking_uri" not in kwargs:
                kwargs.update(
                    tracking_uri=self.local_log_uri,
                )

            if self.local_log_uri[:4] != "file":
                artifact_location = as_path(artifact_location)
                if "mlruns" not in set(artifact_location.parts):
                    artifact_location = artifact_location / "mlruns"
                kwargs.update(artifact_location=artifact_location.as_posix())

            if enable_ray_train and ray_train is not None:
                if kwargs.get("run_name", None) is None:
                    try:
                        trial_name = ray_train.get_context().get_trial_name()
                        if trial_name is not None:
                            kwargs.update(run_name=trial_name)

                        try:
                            run_id = (as_path(ray_train.get_context().get_trial_dir()) / "run_id.txt").read_text()
                            kwargs.update(run_id=run_id)
                        except Exception:
                            pass
                    except Exception:
                        pass
            elif os.environ.get("SLURM_JOB_NAME") is not None:
                kwargs.update(run_name=f"{os.environ.get('SLURM_JOB_NAME')}-{str(uuid.uuid4())[:4]}")

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
        def _track_best(self, metrics: Mapping[str, float]):
            changed = {}
            for k, op in self.track_best_of_metrics.items():
                for m, val in metrics.items():
                    if k in m:
                        if m not in self.monitors:
                            self.monitors[m] = BestMonitor(m, op)
                        if self.monitors[m](val):
                            changed[f"{m}_{op}"] = self.monitors[m].best
            if len(changed) > 0:
                super().log_metrics(changed)
                if self.enable_tb_logging and self._tb_logger is not None:
                    self._tb_logger.log_metrics(changed)

        @rank_zero_only
        def log_metrics(
            self,
            metrics: Mapping[str, float],
            step: int | None = None,
        ) -> None:
            super().log_metrics(metrics, step)
            if self.enable_tb_logging and self._tb_logger is not None:
                self._tb_logger.log_metrics(metrics, step)
            self._track_best(metrics)

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
            image: np.ndarray | torch.Tensor,
            *,
            tag: str | None = None,
            artifact_file: str | None = None,
            global_step: int | None = None,
            dataformats="CHW",
        ):
            """
            Args:
                tag: relative to where image will be saved as jpeg
            """
            if tag is None and artifact_file is None:
                logging.warning("One of tag or artifact file must be provided in order to log image")
                return

            if self.enable_tb_logging and self._tb_logger is not None:
                self.tb_writer.add_image(
                    # tb_tag if tb_tag is not None else Path(artifact_file).stem,
                    tag,
                    image,
                    global_step=global_step,
                    dataformats=dataformats,
                )

            if isinstance(image, torch.Tensor):
                image = torchvision.transforms.ToPILImage()(image.detach().cpu())

            if artifact_file is not None:
                self.mlflow_client.log_image(
                    self.run_id,
                    image,
                    artifact_file=str(Path("images") / artifact_file),
                )
                return

            if tag is not None:
                self.mlflow_client.log_image(
                    self.run_id,
                    image,
                    step=global_step,
                    key=f"{tag}",
                    synchronous=False,  # TODO: double check this
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
