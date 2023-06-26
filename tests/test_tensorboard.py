# %%
from mlflow.tracking.artifact_utils import shutil
import torch
from aibox.torch.logging import CombinedLogger
from aibox.torch.tensorboard import make_image_grid_figure, figure_to_image, tensor_to_figure
from pathlib import Path

import matplotlib.pyplot as plt


def test_make_image_grid():
    plt.close()
    images = [torch.rand((1, 128, 128)) * i for i in range(4)]
    figure = make_image_grid_figure(images, "title", nrow=2, cmap="afmhot")
    # plt.show()
    image = figure_to_image(figure)
    plt.close()
    figure = tensor_to_figure(images[0], "title")
    # plt.show()


def test_mlflow_log_dir():
    log_dir = Path("./logs").resolve()
    logger = CombinedLogger("./logs")
    assert log_dir.exists()

    shutil.rmtree(log_dir)
    assert not log_dir.exists()
