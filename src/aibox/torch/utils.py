try:
    import torch
    import torch.multiprocessing as mp
except ImportError:
    import sys

    print("pytorch required for these utilities")
    sys.exit(1)


def get_num_workers(zero_if_mps=True):
    if zero_if_mps:
        return 0 if torch.has_mps else mp.cpu_count()
    return (mp.cpu_count() / 2) if torch.has_mps else mp.cpu_count()


def get_device():
    if torch.has_cuda:
        return "cuda"
    if torch.has_mps:
        return torch.device("mps")

    return torch.device("cpu")
