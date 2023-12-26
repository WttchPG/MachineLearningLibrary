from collections import OrderedDict
from typing import Dict

import numpy as np

from functions import sigmoid, softmax
from loss_functions import cross_entropy_error
from gradient import numerical_gradient
from layers import Affine, SoftmaxWithLoss, Relu, Sigmoid, AffineLayer, ReluLayer, SoftmaxWithLossLayer


class Net:

    def __init__(self):
        self.layers = [
            AffineLayer("AffineLayer1", 28 * 28, 100),
            ReluLayer("Relu1", 100),
            AffineLayer("AffineLayer2", 100, 50),
            ReluLayer("Relu2", 50),
            AffineLayer("AffineLayer3", 50, 10)
        ]
        self.output_layer = SoftmaxWithLossLayer("OutputLayer")
        # 检查 layer 名称
        layer_names = set([t.name for t in self.layers])
        layer_names.add(self.output_layer.name)
        if len(layer_names) != len(self.layers) + 1:
            raise Exception("层名称不唯一.")
        # 检查 layer
        for i in range(len(self.layers) - 1):
            cur_layer = self.layers[i]
            next_layer = self.layers[i + 1]
            if cur_layer.output_size != next_layer.input_size:
                raise Exception(
                    f"Layer {cur_layer.name} -> {next_layer.name}: 输出不等于输出层大小{cur_layer.output_size} != {next_layer.input_size}")

        # 初始化参数
        for layer in self.layers:
            layer.init_params()

    def predict(self, x: np.ndarray):
        """
        预测结果
        Args:
            x: 要预测的向量的值

        Returns:
            预测的结果
        """
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def loss(self, x: np.ndarray, t: np.ndarray):
        """
        计算损失
        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            使用输入数据进行预测,然后计算结果和监督数据进行交叉熵损失函数计算
        """
        pred = self.predict(x)

        return self.output_layer.forward(pred, t)

    def params(self) -> Dict:

        params = {}
        for layer in self.layers:
            params[layer.name] = {}
            for key, val in layer.params().items():
                params[layer.name][key] = val

        return params

    def gradient(self, x: np.ndarray, t: np.ndarray):
        # forward
        loss = self.loss(x, t)

        # backward
        dout = loss
        dout = self.output_layer.backward(dout)

        layers = list(self.layers)
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        for layer in self.layers:
            grads[layer.name] = {}
            for key, val in layer.grad_params().items():
                grads[layer.name][key] = val

        return grads

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
