from functools import reduce
import pytest

from aibox.cli import CLIException, cli_main

DEFAULT_CONFIG_DIR = "tests/resources/configs"


def test_cli_no_args():
    cli_main(args=[])


def test_cli_args():
    config = cli_main(["-e", "debug", "--model.args.name", "TESTMODEL", "-cd", DEFAULT_CONFIG_DIR])
    assert hasattr(config, "model")
    assert config.model.args.name == "TESTMODEL"


def test_cli_folder_model():
    config = cli_main(["-e", "debug", "-m", "test", "-cd", DEFAULT_CONFIG_DIR])
    assert config.model.class_path == "ae.models.DFCVAE"
    assert config.model.args.name == "non-default"
    assert config.trainer.accelerator == "ddp"


def test_cli_args_dotlist():
    config = cli_main(["-cd", DEFAULT_CONFIG_DIR, "--model.args.name=TESTMODEL"])
    assert config.model.args.name == "TESTMODEL"


def test_cli_bad_args():
    with pytest.raises(CLIException):
        cli_main(["-cd", DEFAULT_CONFIG_DIR, "--model.args.name"])


def in_place_mse(A, B, A_idxs, B_idxs):
    outer_sum = 0.0
    for r_a, r_b in zip(A_idxs, B_idxs, strict=True):
        inner_sum = reduce(lambda x, y: x + y, [(c_a - c_b) ** 2 for c_a, c_b in zip(A[r_a], B[r_b], strict=True)])
        inner_sum /= len(A[r_a])
        outer_sum += inner_sum
    return outer_sum / len(A_idxs)


def in_place_mse_alt(A, B, A_idxs, B_idxs):
    outer_sum = 0.0
    for r_a, r_b in zip(A_idxs, B_idxs, strict=True):
        inner_sum = 0.0
        for c_a, c_b in zip(A[r_a], B[r_b], strict=True):
            inner_sum += (c_a - c_b) ** 2
        inner_sum /= len(A[r_a])
        outer_sum += inner_sum
    return outer_sum / len(A_idxs)


def test_mse_loop():
    import torch
    import torch.nn.functional as F
    import random

    A = torch.rand(256, 100)
    B = torch.rand(256, 100)

    num_shuffles = 100

    for _ in range(num_shuffles):
        A_idxs = list(torch.arange(0, 256, 1))
        random.shuffle(A_idxs)

        B_idxs = list(torch.arange(0, 256, 1))
        random.shuffle(B_idxs)

        ground_truth = F.mse_loss(A[A_idxs], B[B_idxs])
        in_place = in_place_mse(A, B, A_idxs, B_idxs)
        in_place_alt = in_place_mse_alt(A, B, A_idxs, B_idxs)

        assert torch.allclose(ground_truth, torch.tensor(in_place))
        assert torch.allclose(ground_truth, torch.tensor(in_place_alt))
