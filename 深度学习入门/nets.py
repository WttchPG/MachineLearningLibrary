from collections import OrderedDict

import numpy as np

from functions import sigmoid, softmax
from loss_functions import cross_entropy_error
from gradient import numerical_gradient
from layers import Affine, SoftmaxWithLoss, Relu, Sigmoid


class TwoLayerNet:
    """
    两层神经网络
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std=0.01):
        """

        Args:
            input_size: 输入层大小
            hidden_size: 隐藏层大小
            output_size: 输出层大小
            weight_init_std: 初始化时的权重矩阵标准差
        """
        # 初始化权重
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size),
            'b1': np.zeros(hidden_size),
            'W2': weight_init_std * np.random.randn(hidden_size, output_size),
            'b2': np.zeros(output_size)
        }
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray):
        """
        预测结果
        Args:
            x: 要预测的向量的值

        Returns:
            预测的结果
        """
        for layer in self.layers.values():
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

        return self.lastLayer.forward(pred, t)

    def graident1(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        return {
            'W1': numerical_gradient(loss_w, self.params['W1']),
            'b1': numerical_gradient(loss_w, self.params['b1']),
            'W2': numerical_gradient(loss_w, self.params['W2']),
            'b2': numerical_gradient(loss_w, self.params['b2'])
        }

    def gradient(self, x: np.ndarray, t: np.ndarray):
        # forward
        loss = self.loss(x, t)

        # backward
        dout = loss
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {
            'W1': self.layers['Affine1'].dw,
            'b1': self.layers['Affine1'].db,
            'W2': self.layers['Affine2'].dw,
            'b2': self.layers['Affine2'].db,
        }

        return grads

    def accuracy(self, x: np.ndarray, t: np.ndarray):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
