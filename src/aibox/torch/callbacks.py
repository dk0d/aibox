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

    Add this callback to the trainer callbacks
    and implement the `get_log_images(batch, split: str, **kwargs)` method in your LightningModule.

    """

    def __init__(
        self,
        step_frequency=100000,
        frequency_increase_base=2,
        nrow=8,
        max_images=8,
        clamp=True,
        rescale=False,
        increase_log_steps=True,
        log_on_batch_idx=False,  # Log in batch_idx instead of global step
        log_first_step=False,
        interlace_images=True,
        get_log_images_kwargs=None,
        log_last=True,
        disabled=False,
    ):
        """
        Create a new LogImagesCallback.

        Args:
            step_frequency (int, optional): Log images every `step_frequency` steps. Defaults to 100000.
            frequency_increase_base (int, optional): Base of the exponential increase of the log frequency.
                Defaults to 2.
            nrow (int, optional): Number of images per row. Defaults to 8.
            max_images (int, optional): Maximum number of images to log. Defaults to 8.
            clamp (bool, optional): Clamp images to [0, 1]. Defaults to True.
            rescale (bool, optional): Rescale images to [0, 1]. Defaults to False.
            increase_log_steps (bool, optional): Increase the log frequency exponentially. Defaults to True.
            log_on_batch_idx (bool, optional): Log images on batch_idx instead of global_step. Defaults to False.
            log_first_step (bool, optional): Log images on the first step. Defaults to False.
            log_images_kwargs (dict, optional): Additional kwargs to pass to `get_log_images`. Defaults to None.
            log_last (bool, optional): Log images on the last step. Defaults to True.
            disabled (bool, optional): Disable the callback. Defaults to False.
        """
        super().__init__()
        self.step_frequency = step_frequency
        self.max_images = max_images
        self.nrow = nrow
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.rescale = rescale
        self.disabled = disabled
        self.clamp = clamp
        self.log_last_batch = log_last
        self.get_log_images_kwargs = get_log_images_kwargs if get_log_images_kwargs is not None else {}
        self.interlace_images = interlace_images
        self.frequency_increase_base = frequency_increase_base
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

    @staticmethod
    def _img_path(split, global_step, epoch_idx, batch_idx=None, k=None):
        if batch_idx is None:
            out = f"{split}/epoch{epoch_idx}/gs{global_step}"
        else:
            out = f"{split}/epoch{epoch_idx}/gs{global_step}_b{batch_idx}"

        if k is not None:
            out += f"_{k}"
        return out

    @staticmethod
    def _interlace_images(images: list[torch.Tensor], maxImages: int = 8) -> torch.Tensor:
        """
        assumes image tensors are of shape (batch, channels, height, width)

        takes list of images and interlaces them into a single tensor of images of size
        (batch * len(images), channels, height, width)

        """
        if len(images) == 1:
            return images[0]

        numImages = min(images[0].shape[0], maxImages)
        logIms = [torch.stack(row, dim=0) for row in zip(*[im[:numImages].detach().cpu() for im in images])]
        return torch.cat(logIms, dim=0)

    @rank_zero_only
    def _tensorboard(
        self,
        pl_module,
        image,
        split,
        batch_idx=None,
        k=None,
    ):
        writer: SummaryWriter = pl_module.logger.experiment
        if not isinstance(writer, SummaryWriter):
            return
        grid = torchvision.utils.make_grid(image, nrow=self.nrow)
        grid = self._normalize(grid)
        label = self._img_path(
            split,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            k,
        )
        writer.add_image(label, grid.detach().cpu())

    @rank_zero_only
    def _mlflow(
        self,
        pl_module: L.LightningModule,
        image: list[torch.Tensor],
        split: str,
        batch_idx: int | None = None,
        k: int | None = None,
    ):
        """Log images to MLFlow.

        Supports both MLFlowLogger and CombinedLogger.

        Args:
            pl_module (L.LightningModule): ...
            images (list[torch.Tensor]): each tensor is a batch of images of shape (N, C, H, W)
            k (int): image index if not interlaced, defaults to None
            split (str): _description_
            batch_idx (int, optional): . Defaults to None.
        """
        logger = pl_module.logger
        if not isinstance(logger, (CombinedLogger, MLFlowLogger)):
            return
        # for k, image in enumerate(images):  # allows for multiple images per batch
        grid = torchvision.utils.make_grid(image, nrow=self.nrow)
        grid = self._normalize(grid)
        label = self._img_path(
            split,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            k,
        )
        if isinstance(logger, CombinedLogger):
            # pass to logger instead in case combined logger is using both tensorboard and MLFlow
            logger.log_image(
                tag=label,
                image=grid.detach().cpu(),
                global_step=pl_module.global_step,
            )
        else:
            client: MlflowClient = logger.experiment
            run_id = logger.run_id
            if run_id is not None:
                client.log_image(
                    run_id,
                    image,
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
        can_log = self.check_frequency(pl_module.global_step, self.step_frequency, self.log_steps)
        if (
            can_log
            and hasattr(pl_module, "get_log_images")  # batch_idx % self.batch_freq == 0
            and callable(pl_module.get_log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images: list[torch.Tensor] = pl_module.get_log_images(batch, split=split, **self.get_log_images_kwargs)

            _logger_log_images = self._logger_log_images.get(logger, self._NOOP_log)

            if self.interlace_images:  # log all images together to single image grid
                all_images = self._interlace_images(images, self.max_images)
                _logger_log_images(pl_module, all_images, split, batch_idx)
            else:  # log each batch of images separately
                for k, img in enumerate(images):
                    N = min(img.shape[0], self.max_images)
                    img = img[:N]
                    if isinstance(img, torch.Tensor):
                        img = img.detach().cpu()
                    _logger_log_images(pl_module, img, split, batch_idx, k=k)

            # self.log_local(
            #     pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx
            # )

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
        self.log_steps = [
            self.frequency_increase_base**n
            for n in range(int(np.log(self.step_frequency) / np.log(self.frequency_increase_base)) + 1)
        ]
        if not self.increase_log_steps:
            self.log_steps = [self.step_frequency]
        if self.log_first_step:
            self.log_steps.insert(0, 0)
        if self.log_last_batch and trainer is not None:
            # second to last in the case of an incomplete last batch # (image grid and batch assumptions)
            self.log_steps.append(trainer.num_training_batches - 2)

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

        if pl_module.global_step > 0 or (pl_module.global_step == 0 and self.log_first_step) or self.log_last_batch:
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

        if pl_module.global_step > 0 or (pl_module.global_step == 0 and self.log_first_step):
            self.log_image(pl_module, batch, batch_idx, split="test")
