import numpy as np
from sklearn.utils.extmath import softmax

from loss_functions import cross_entropy_error


class AddLayer:
    x: np.ndarray
    y: np.ndarray

    def __init__(self):
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    def backward(self, dout) -> tuple[np.ndarray, np.ndarray]:
        """
        反向传播。
        加法: f = x + y, dL / df = dout
        链式求导: dL / dx = dL / df * df / dx = dout * 1
        同理: dL / dy = dout * 1
        Args:
            dout:

        Returns:
            结果对反向的影响。
                x 对 dout 的影响倍数为 dout * 1; 如果 dout 改变 1, 则 x 值改变  1 / dout
                y 对 dout 的影响倍数为 dout * 1; 如果 dout 改变 1, 则 y 值改变  1 / dout
        """
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class MulLayer:
    x: np.ndarray
    y: np.ndarray

    def __init__(self):
        pass

    def forward(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.x = x
        self.y = y

        out = x * y
        return out

    def backward(self, dout) -> tuple[np.ndarray, np.ndarray]:
        """
        反向传播。
        乘法: f = x * y, dL / df = dout
        链式求导: dL / dx = dL / df * df / dx = dout * y
        同理: dL / dy = dout * x

        Args:
            dout: 改层输出改变的偏导

        Returns:
            结果对反向的影响。
                x 对 dout 的影响倍数为 dout * y; 如果 dout 改变 1, 则 x 值改变  1 / (dout * y)
                y 对 dout 的影响倍数为 dout * x; 如果 dout 改变 1, 则 y 值改变  1 / (dout * x)
        """
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class Relu:
    mask: np.ndarray

    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout) -> np.ndarray:
        """
        反向传播。
        relu: x > 0: y = x; x <= 0: y = 0
        链式求导:
            x > 0: dL / dy * dy / dx = dout * 1
            x <= 0: dL / dy * dy / dx = 0
        Args:
            dout:

        Returns:
            结果对反向的影响。
                x > 0: x 对 dout 的影响倍数为 dout * 1; 如果 dout 改变 1, 则 x 值改变  1 / (dout * 1)
                x <= 0: x 对 dout 的影响为 0
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    """
    y = 1 / (1 + e^(-x))
    a = -x
    b = e^a
    c = 1 + b
    y = 1 / c
    """

    out: np.ndarray

    def __init__(self):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout) -> np.ndarray:
        """
        反向传播。
        第一次反向传播: dL / dc = dL / dy * dy / dc = dout * - 1 / c ** 2 = - dout * y ** 2
        第二次反向传播: dL / db = dL / dc * dc / db = - dout * y ** 2
        第三次反向传播: dL / dx
            = dL / db * db / da * da / dx
            = - dout * y ** 2 * e ** (-x) * -1
            = dout * y ** 2 * e ** (-x)
            = dout * 1 / (1 + e ** (-x)) * e ** (-x) / (1 + e ** (-x))
            = dout * y * (1 - y)
        Args:
            dout:
        Returns:
            结果对反向的影响。
                x 对 dout 的影响倍数为 dout * y * (1 - y); 如果 dout 改变 1, 则 x 值改变  1 / (dout * y * (1- y))

        """
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    W: np.ndarray
    b: np.ndarray
    x: np.ndarray
    dw: np.ndarray
    db: np.ndarray

    def __init__(self, W: np.ndarray, b: np.ndarray):
        # 不是深拷贝, 实际更新的时候，这里的值也会更新
        self.W = W
        self.b = b

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x

        return x @ self.W + self.b

    def backward(self, dout: np.ndarray):
        """
        反向传播。
        dout(N, T)
        f = x(N, M) * W(M, T) + b(N, T)
        dL / dx (N, M) = dL / df * df / dx = dout (N, T) * W^T (T, M)
        dL / dw (M, T) = dL / df * df / dw = x^T(M, N) * dout(N, T)
        dL / db (T) = dL / df * df / db = sum(dout (N, T), axis=0)
        Args:
            dout:

        Returns:

        """
        dx = dout @ self.W.T
        self.dw = self.x.T @ dout
        self.db = dout.sum(axis=0)

        return dx


class SoftmaxWithLoss:
    loss: float
    y: np.ndarray
    t: np.ndarray

    def __init__(self):
        pass

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
