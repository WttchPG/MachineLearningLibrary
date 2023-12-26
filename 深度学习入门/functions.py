import numpy as np
from sklearn.preprocessing import Normalizer


def sigmoid(z: np.ndarray) -> float:
    """
    s型曲线激活函数
    """
    z = z.clip(-500, 500)
    g = 1 / (1 + np.exp(-z))

    return g


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
