import lightning as L
import torch
from lightning.fabric.loggers.logger import Logger
from torch.utils.tensorboard.writer import SummaryWriter


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
