import numpy as np


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
