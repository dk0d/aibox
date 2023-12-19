# %%
from pathlib import Path

import pytest
import torch


def test_make_image_grid():
    import matplotlib.pyplot as plt

    from aibox.torch.tensorboard import figure_to_image, make_image_grid_figure, tensor_to_figure

    plt.close()
    images = [torch.rand((1, 128, 128)) * i for i in range(4)]
    figure = make_image_grid_figure(images, "title", nrow=2, cmap="afmhot")
    # plt.show()
    image = figure_to_image(figure)
    plt.close()
    figure = tensor_to_figure(images[0], "title")
    # plt.show()


@pytest.mark.parametrize(
    "N,expected",
    [
        (2, (2, 1)),
        (5, (3, 2)),
        (7, (4, 2)),
        (11, (4, 3)),
        (12, (4, 3)),
        (15, (5, 3)),
        (16, (4, 4)),
        (21, (7, 3)),
        (40, (8, 5)),
        (41, (7, 6)),
    ],
)
def test_nearest_square_root(N, expected):
    from aibox.utils import nearest_square_grid

    assert nearest_square_grid(N) == expected


def test_mlflow_log_dir():
    from mlflow.tracking.artifact_utils import shutil

    from aibox.torch.logging import CombinedLogger

    log_dir = Path("./logs").resolve()
    logger = CombinedLogger("./logs")
    assert log_dir.exists()

    shutil.rmtree(log_dir)
    assert not log_dir.exists()
