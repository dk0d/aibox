from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchvision
from lightning.pytorch.loggers import MLFlowLogger, TensorBoardLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from mlflow.client import MlflowClient
from PIL import Image
from torch.utils.tensorboard.writer import SummaryWriter

from .logging import CombinedLogger


class InputMonitor(L.Callback):
    """
    This callback logs a histogram of the input to the model every `log_every_n_steps` steps.
    """

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx: int) -> None:
        if trainer.logger is None:
            return

        if (batch_idx + 1) % trainer.log_every_n_steps == 0:
            x, *_ = batch
            if isinstance(trainer.logger.experiment, SummaryWriter):
                experiment = trainer.logger.experiment
            elif hasattr(trainer.logger, "tb_writer"):
                experiment = trainer.logger.tb_writer
            else:
                return

            if experiment is None:
                return

            experiment.add_histogram("input", x, global_step=trainer.global_step)


class CheckBatchGradient(L.Callback):
    """
    Gradient checking callback for PyTorch Lightning.

    Validates that the model does not mix data across the batch dimension.
    """

    def on_train_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """
        Called when training begins
        """
        n = 0

        if module.example_input_array is None:
            return

        inputs = list(module.example_input_array)
        x = inputs[n]
        for i, x in enumerate(inputs):
            x = x.to(module.device)
            x.requires_grad = True
            inputs[i] = x

        module.zero_grad()
        output = module(*inputs)
        output[n].abs().sum().backward()

        zero_grad_inds = list(range(x.size(0)))
        zero_grad_inds.pop(n)

        if x.grad is None:
            return

        if x.grad[zero_grad_inds].abs().sum().item() > 0:
            raise RuntimeError("Your model mixes data across the batch dimension!")


class LogImagesCallback(L.Callback):
    """
    Callback that logs images to lightning loggers.

    Supports TensorBoardLogger and CombinedLogger, and MLFlowLogger.

    """

    def __init__(
        self,
        batch_frequency=100000,
        epoch_frequency=10,
        frequency_base=2,
        nrow=8,
        max_images=8,
        clamp=True,
        rescale=False,
        increase_log_steps=True,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_last=True,
        disabled=False,
    ):
        super().__init__()
        self.batch_frequency = batch_frequency
        self.max_images = max_images
        self.nrow = nrow
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.rescale = rescale
        self.disabled = disabled
        self.clamp = clamp
        self.log_last_batch = log_last
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs is not None else {}
        self.frequency_base = frequency_base
        self.epoch_frequency = epoch_frequency
        self.increase_log_steps = increase_log_steps
        self._setup_log_steps()
        self._logger_log_images = {
            TensorBoardLogger: self._tensorboard,
            MLFlowLogger: self._mlflow,
            CombinedLogger: self._mlflow,
        }

    def _NOOP_log(self, *args, **kwargs):
        pass

    def _normalize(self, images):
        if self.rescale:
            images = (images + 1.0) / 2.0
        if self.clamp:
            images = torch.clamp(images, 0.0, 1.0)
        return images

    @rank_zero_only
    def _tensorboard(
        self,
        pl_module,
        images,
        split,
        batch_idx=None,
    ):
        writer: SummaryWriter = pl_module.logger.experiment
        if not isinstance(writer, SummaryWriter):
            return
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)
            grid = self._normalize(grid)
            label = f"{split}/{k}" if batch_idx is None else f"{split}/{k}_{batch_idx}"
            writer.add_image(label, grid.detach().cpu(), pl_module.global_step)

    @rank_zero_only
    def _mlflow(
        self,
        pl_module: L.LightningModule,
        images: list[torch.Tensor],
        split: str,
        batch_idx: int | None = None,
    ):
        """Log images to MLFlow.

        Supports both MLFlowLogger and CombinedLogger.

        Args:
            pl_module (L.LightningModule): ...
            images (list[torch.Tensor]): each tensor is a batch of images of shape (N, C, H, W)
            split (str): _description_
            batch_idx (int, optional): . Defaults to None.
        """
        logger = pl_module.logger
        global_step = pl_module.global_step
        if not isinstance(logger, (CombinedLogger, MLFlowLogger)):
            return
        for k, img in enumerate(images):  # allows for multiple images per batch
            grid = torchvision.utils.make_grid(img, nrow=self.nrow)
            grid = self._normalize(grid)
            label = (
                f"{split}/epoch{global_step}/{k}"
                if batch_idx is None
                else f"{split}/epoch{global_step}/batch{batch_idx}_{k}"
            )
            if isinstance(logger, CombinedLogger):
                # pass to logger instead in case combined logger is using both tensorboard and MLFlow
                logger.log_image(tag=label, image=grid.detach().cpu(), global_step=global_step)
            else:
                client: MlflowClient = logger.experiment
                client.log_image(
                    logger.run_id,
                    images.detach().cpu(),
                    artifact_file=f"{label}.png",
                )

    @rank_zero_only
    def log_local(
        self,
        save_dir,
        split,
        images,
        global_step,
        current_epoch,
        batch_idx,
    ):
        root = Path(save_dir) / "images" / split
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)
            grid = self._normalize(grid)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy().astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = root / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_image(
        self,
        pl_module,
        batch,
        batch_idx,
        split="train",
    ):
        can_log = self.check_frequency(
            pl_module.global_step, self.epoch_frequency, self.epoch_log_steps
        ) and self.check_frequency(batch_idx, self.batch_frequency, self.batch_log_steps)

        if (
            can_log
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

            for k in range(len(images)):  # allows for multiple images per batch
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()

            # self.log_local(
            #     pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx
            # )
            _logger_log_images = self._logger_log_images.get(logger, self._NOOP_log)
            _logger_log_images(pl_module, images, split, batch_idx)

            if is_train:
                pl_module.train()

    @rank_zero_only
    def log_histogram(self, pl_module, histogram, batch_idx, split, factor_name):
        pl_module.logger.experiment.add_histogram(
            f"{split}/{factor_name}", histogram.detach().cpu(), pl_module.global_step
        )

    def check_frequency(self, idx, freq, steps):
        if (idx % freq) == 0 or idx in steps:
            # try:
            #     steps.pop(0)
            # except IndexError:
            #     pass
            return True

        return False

    def _setup_log_steps(self, trainer=None):
        self.batch_log_steps = [
            self.frequency_base**n for n in range(int(np.log(self.batch_frequency) / np.log(self.frequency_base)) + 1)
        ]
        self.epoch_log_steps = [
            self.frequency_base**n for n in range(int(np.log(self.epoch_frequency) / np.log(self.frequency_base)) + 1)
        ]

        if not self.increase_log_steps:
            self.batch_log_steps = [self.batch_frequency]
            self.epoch_frequency = [self.epoch_frequency]

        if self.log_first_step:
            self.epoch_log_steps.insert(0, 0)
            self.batch_log_steps.insert(0, 0)

        if self.log_last_batch and trainer is not None:
            self.batch_log_steps.append(trainer.num_training_batches - 1)

    def on_train_start(
        self,
        trainer,
        pl_module,
    ) -> None:
        self._setup_log_steps(trainer)

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.disabled:
            return

        if pl_module.global_step > 0 or self.log_first_step or self.log_last_batch:
            self.log_image(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.disabled:
            return

        if pl_module.global_step > 0:
            self.log_image(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        if self.disabled:
            return

        if pl_module.global_step > 0 or self.log_first_step:
            self.log_image(pl_module, batch, batch_idx, split="test")
