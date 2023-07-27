from argparse import Namespace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# from lightning_fabric.fabric import rank_zero_experiment
from aibox.config import config_to_dotlist

try:
    import shutil

    import numpy as np
    import torch
    import torchvision
    from lightning_fabric.loggers.logger import rank_zero_experiment, rank_zero_only
    from mlflow.client import MlflowClient
    from PIL import Image
    from pytorch_lightning.callbacks import Callback
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
    from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger
    from pytorch_lightning.loggers.mlflow import LOCAL_FILE_URI_PREFIX
    from pytorch_lightning.loggers.utilities import _scan_checkpoints
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
            root_log_dir=None,
            tb_log_graph=True,
            enable_tb_logging=False,
            **kwargs,
        ):
            self.root_log_dir = Path(root_log_dir or "logs")
            self.enable_tb_logging = enable_tb_logging
            self._tensorboard_logdir = (self.root_log_dir / "tbruns").expanduser().resolve()
            if "tracking_uri" not in kwargs:
                mlflow_logdir = f"file:{self.root_log_dir / 'mlruns'}"
                kwargs.update(tracking_uri=mlflow_logdir)

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
                container = dict(**OmegaConf.to_container(params, resolve=True))
                # log config as an artifact for easier instantiation of model from the config
                self.experiment.log_dict(self.run_id, container, "config.yml")
                params = dict(**config_to_dotlist(params))
            super().log_hyperparams(params)

        @rank_zero_only
        def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
            super().log_metrics(metrics, step)
            if self.enable_tb_logging and self._tb_logger is not None:
                self._tb_logger.log_metrics(metrics, step)

        @rank_zero_only
        def log_graph(self, *args, **kwargs):
            if self.enable_tb_logging and self._tb_logger is not None:
                return self._tb_logger.log_graph(*args, **kwargs)

        @rank_zero_only
        def log_image(
            self,
            tag: str,
            images: np.ndarray | torch.Tensor,
            global_step=None,
            dataformats="CHW",
        ):
            if self.enable_tb_logging and self._tb_logger is not None:
                self.tb_writer.add_image(tag, images, global_step, dataformats)

            # run_id: str, image: Union["numpy.ndarray", "PIL.Image.Image"], artifact_file: str
            self.mlflow_client.log_image(
                self.run_id,
                images,
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
                if Path(self._tb_logger.log_dir).exists():
                    self.experiment.log_artifacts(self.run_id, self._tb_logger.log_dir, "tensorboard_events")

        def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
            # get checkpoints to be saved with associated score
            super()._scan_and_log_checkpoints(checkpoint_callback)
            try:
                checkpoints = _scan_checkpoints(checkpoint_callback, {})
                if self.save_dir is not None:
                    for _, p, _, _ in checkpoints:
                        if Path(p).is_relative_to(self.save_dir) and p in self._logged_model_time:
                            Path(p).unlink()
                    parent = list(set(Path(p).parent for _, p, _, _ in checkpoints))[0]
                    if parent.is_dir() and not list(parent.iterdir()):
                        shutil.rmtree(parent)
            except Exception:
                pass

    class LogImagesCallback(Callback):
        """
        Callback that logs images to Tensorboard or MLFlow
        """

        def __init__(
            self,
            batch_frequency,
            frequency_base=2,
            nrow=8,
            max_images=8,
            clamp=True,
            rescale=True,
            increase_log_steps=True,
            log_on_batch_idx=False,
            log_first_step=False,
            log_images_kwargs=None,
            disabled=False,
        ):
            super().__init__()
            self.batch_freq = batch_frequency
            self.max_images = max_images
            self.nrow = nrow
            self.log_on_batch_idx = log_on_batch_idx
            self.log_first_step = log_first_step
            self.rescale = rescale
            self.disabled = disabled

            self.logger_log_images = {
                TensorBoardLogger: self._tensorboard,
                MLFlowLogger: self._mlflow,
            }
            self.log_steps = [
                frequency_base**n for n in range(int(np.log(self.batch_freq) / np.log(frequency_base)) + 1)
            ]
            if not increase_log_steps:
                self.log_steps = [self.batch_freq]
            self.clamp = clamp
            self.log_images_kwargs = log_images_kwargs if log_images_kwargs is not None else {}

        @rank_zero_only
        def _tensorboard(self, pl_module, images, split, batch_idx=None):
            writer: SummaryWriter = pl_module.logger.experiment
            if not isinstance(writer, SummaryWriter):
                return
            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)
                grid = (grid + 1.0) / 2.0
                label = f"{split}/{k}" if batch_idx is None else f"{split}/{k}_{batch_idx}"
                writer.add_image(label, grid.detach().cpu(), pl_module.global_step)

        @rank_zero_only
        def _mlflow(self, pl_module, images, split, batch_idx=None):
            # TODO:
            logger: CombinedLogger = pl_module.logger
            if not isinstance(logger, CombinedLogger):
                return
            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)
                grid = (grid + 1.0) / 2.0
                label = f"{split}/{k}" if batch_idx is None else f"{split}/{k}_{batch_idx}"
                logger.log_image(label, grid.detach().cpu(), pl_module.global_step)

        @rank_zero_only
        def _wandb(self, pl_module, images, batch_idx, split):
            raise ValueError("No way wandb")
            # grids = dict()
            # for k in images:
            #     grid = torchvision.utils.make_grid(images[k])
            #     grids[f"{split}/{k}"] = wandb.Image(grid)
            # pl_module.logger.experiment.log(grids)

        @rank_zero_only
        def _testtube(self, pl_module, images, batch_idx, split):
            for k in images:
                grid = torchvision.utils.make_grid(images[k])
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

                tag = f"{split}/{k}"
                pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

        @rank_zero_only
        def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
            root = Path(save_dir) / "images" / split
            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)

                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy().astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
                path = root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(grid).save(path)

        def log_img(self, pl_module, batch, batch_idx, split="train"):
            check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
            if (
                self.check_frequency(check_idx)
                and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
                and callable(pl_module.log_images)
                and self.max_images > 0
            ):
                logger = type(pl_module.logger)

                is_train = pl_module.training
                if is_train:
                    pl_module.eval()

                with torch.no_grad():
                    images: list[torch.Tensor] = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

                for k in images:  # allows for multiple images per batch
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1.0, 1.0)

                # self.log_local(
                #     pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx
                # )
                logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                logger_log_images(pl_module, images, pl_module.global_step, split)

                if is_train:
                    pl_module.train()

        @rank_zero_only
        def log_histogram(self, pl_module, histogram, batch_idx, split, factor_name):
            pl_module.logger.experiment.add_histogram(
                f"{split}/{factor_name}", histogram.detach().cpu(), pl_module.global_step
            )

        def check_frequency(self, iter_idx):
            if (iter_idx % self.batch_freq) == 0 or iter_idx in self.log_steps:
                try:
                    self.log_steps.pop(0)
                except IndexError:
                    pass
                return True
            return False

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            if self.disabled:
                return

            if pl_module.global_step > 0 or self.log_first_step:
                self.log_img(pl_module, batch, trainer.global_step, split="train")

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            if self.disabled:
                return

            if pl_module.global_step > 0:
                self.log_img(pl_module, batch, batch_idx, split="val")

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            if self.disabled:
                return

            if pl_module.global_step > 0 or self.log_first_step:
                self.log_img(pl_module, batch, batch_idx, split="test")

except ImportError:
    print("MLFlow not installed, CombinedLogger will not be available.")
