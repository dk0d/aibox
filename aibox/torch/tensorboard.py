try:
    import torch.nn as nn
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    print("pytorch required for these utilities")
    exit(1)


def weight_histograms(
    writer: SummaryWriter, step: int, model: nn.Module, prefix="", per_kernel=True
):
    print("Visualizing model weights...")

    histParams = dict(global_step=step, bins="tensorflow")
    # Iterate over all model parameters
    for name, param in model.named_parameters(prefix=prefix):
        if per_kernel and len(param.shape) >= 4:
            # if len(param.shape) == 4: # Conv2d or kernel
            #     num_kernels, in_channels = param.shape[:2]
            # if len(param.shape) == 5: # Conv3d or kernel
            #     num_kernels, in_channels = param.shape[:2]
            num_kernels, in_channels = param.shape[:2]
            for k in range(num_kernels):
                writer.add_histogram(
                    f"{name}.k{k}".replace(".", "/"), param[k].flatten(), **histParams
                )
        else:
            writer.add_histogram(name.replace(".", "/"), param.flatten(), **histParams)


try:
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

    class TBLogger(TensorBoardLogger):
        @property
        def writer(self) -> SummaryWriter:
            return self.experiment

        def log_named_parameters(self, model: nn.Module, global_step, prefix=""):
            weight_histograms(self.writer, step=global_step, model=model, prefix=prefix)

except ImportError:
    print("Unable to use Pytorch Lightning TensorboardLogger ")
