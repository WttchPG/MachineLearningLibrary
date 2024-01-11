import torch
from torch import device


def get_device() -> device:
    return torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.mps is not None else 'cpu')