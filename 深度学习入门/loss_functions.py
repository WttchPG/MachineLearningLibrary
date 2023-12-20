import numpy as np


def mean_squared_error(y: np.ndarray, y_bar: np.ndarray):
    """
    均方误差
    Args:
        y:
        y_bar:

    Returns:

    """
    return np.mean((y - y_bar) ** 2)


def cross_entropy_error(y: np.ndarray, p: np.ndarray):
    """
    交叉熵误差
    """
    delta = 1e-7

    batch_size = y.shape[0]
    return - np.sum(y * np.log(p + delta)) / batch_size
