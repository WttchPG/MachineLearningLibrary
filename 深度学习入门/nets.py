import numpy as np

from functions import sigmoid, softmax
from loss_functions import cross_entropy_error
from gradient import numerical_gradient


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

    def predict(self, x: np.ndarray):
        """
        预测结果
        Args:
            x: 要预测的向量的值

        Returns:
            预测的结果
        """
        # 获取两层权重
        W1, W2 = self.params['W1'], self.params['W2']
        # 获取两个偏置参数
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = x @ W1 + b1
        # 第一层使用 sigmoid 激活函数
        z1 = sigmoid(a1)
        # 输出层使用 softmax 激活函数
        a2 = z1 @ W2 + b2
        y = softmax(a2)

        return y

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

        return cross_entropy_error(pred, t)

    def numerical_gradient(self, x: np.ndarray, t: np.ndarray):
        """
        计算数值微分梯度
        Args:
            x: 输入数据
            t: 监督数据

        Returns:
            使用输入数据和监督数据计算权重和偏置参数的梯度
        """
        loss_w = lambda w: self.loss(x, t)
        grads = {
            'W1': numerical_gradient(loss_w, self.params['W1']),
            'b1': numerical_gradient(loss_w, self.params['b1']),
            'W2': numerical_gradient(loss_w, self.params['W2']),
            'b2': numerical_gradient(loss_w, self.params['b2'])
        }

        return grads
