import abc
from typing import Dict

import numpy as np


class Optimizer(abc.ABC):
    """
    优化器的超类。
    """

    @abc.abstractmethod
    def update(self, params: Dict, grad: Dict):
        pass


class SGD(Optimizer):
    """
    随机梯度下降。
    Stochastic Gradient Descent.

    SGD 低效的原因：梯度的方向并没有指向最小值方向。
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params: Dict, grad: Dict):
        for key in params.keys():
            params[key] -= self.lr * grad[key]


class Momentum(Optimizer):
    """
    动量。
    """

    v: Dict

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def update(self, params: Dict, grad: Dict):
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grad[key]
            params[key] += self.v[key]


class AdaGrad(Optimizer):
    """
    Adaptive Grad.
    """

    h: Dict[str, np.ndarray]

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params: Dict[str, np.ndarray], grad: Dict[str, np.ndarray]):
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}

        for key in params.keys():
            self.h[key] += grad[key] * grad[key]
            params[key] -= self.lr * grad[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam(Optimizer):
    """
    Adam (http://arxiv.org/abs/1412.6980v8)
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: Dict, grads: Dict):
        if self.m is None:
            self.m = {key: np.zeros_like(val) for key, val in params.items()}
            self.v = {key: np.zeros_like(val) for key, val in params.items()}
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
