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
    x = x - x.max()
    exp_x = np.exp(x)  # 防止溢出
    sum_exp = np.sum(exp_x)
    y = exp_x / sum_exp
    return y
