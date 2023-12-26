import abc
from typing import Dict, Optional

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

    def update(self, params: Dict[str, Dict], grad: Dict[str, Dict[str, np.ndarray]]):
        for layer, layer_params in params.items():
            for key, val in layer_params.items():
                params[layer][key] -= self.lr * grad[layer][key]


class Momentum(Optimizer):
    """
    动量。
    """

    v: Dict

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum

    def update(self, params: Dict[str, Dict], grad: Dict[str, Dict[str, np.ndarray]]):
        if self.v is None:
            self.v = {layer: {key: np.zeros_like(val) for key, val in layer_params.items()}
                      for layer, layer_params in params.items()}

        for layer, layer_params in params.items():
            for key, val in layer_params.items():
                self.v[layer][key] = self.momentum * self.v[layer][key] - self.lr * grad[layer][key]
                params[layer][key] += self.v[layer][key]


class AdaGrad(Optimizer):
    """
    Adaptive Grad.
    """

    h: Optional[Dict[str, Dict[str, np.ndarray]]]

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params: Dict[str, Dict], grad: Dict[str, Dict[str, np.ndarray]]):
        if self.h is None:
            self.h = {layer: {key: np.zeros_like(val) for key, val in layer_params.items()}
                      for layer, layer_params in params.items()}

        for layer, layer_params in params.items():
            for key, val in layer_params.items():
                self.h[layer][key] += grad[layer][key] * grad[layer][key]
                params[layer][key] -= self.lr * grad[layer][key] / (np.sqrt(self.h[layer][key]) + 1e-7)


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

    def update(self, params: Dict[str, Dict], grads: Dict[str, Dict]):
        if self.m is None:
            self.m = {}
            self.v = {}
            for layer, layer_params in params.items():
                self.m[layer] = {key: np.zeros_like(val) for key, val in layer_params.items()}
                self.v[layer] = {key: np.zeros_like(val) for key, val in layer_params.items()}

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for layer_key in params.keys():
            for layer_param in params[layer_key]:
                self.m[layer_key][layer_param] += (1 - self.beta1) * (
                        grads[layer_key][layer_param] - self.m[layer_key][layer_param])
                self.v[layer_key][layer_param] += (1 - self.beta2) * (
                        grads[layer_key][layer_param] ** 2 - self.v[layer_key][layer_param])

                params[layer_key][layer_param] -= lr_t * self.m[layer_key][layer_param] / (
                        np.sqrt(self.v[layer_key][layer_param]) + 1e-7)
