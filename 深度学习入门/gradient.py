from typing import Callable

import numpy as np


def numerical_diff(func: Callable[[float], float], x: float):
    """
    数值微分
    """
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2 * h)


def numerical_gradient(f: Callable[[np.ndarray], float], x: np.ndarray):
    """
    梯度
    Parameters
    ----------
    f
    x

    Returns
    -------

    """
    h = 1e-4
    grad = np.zeros_like(x)
    x = x.astype(float)

    for i in range(x.shape[0]):
        tmp_val = x[i]
        # f(x + h)
        x[i] = tmp_val + h
        fxh1 = f(x)
        # f(x - h)
        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)

        x[i] = tmp_val

    return grad


def gradient_descent(f: Callable[[np.ndarray], float], init_x: np.ndarray, lr=0.01, step_num=100) -> np.ndarray:
    """
    梯度下降
    Parameters
    ----------
    f: 要计算的函数
    init_x: 初始化的 x
    lr: 学习率
    step_num: 下降次数

    Returns
    -------

    """
    x = init_x.astype(float)

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x
