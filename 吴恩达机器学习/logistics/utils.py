import numpy as np


def sigmoid(z):
    """
    计算z的s型曲线
    :param z:
    :type z: np.ndarray
    :return:
    """
    z = z.clip(-500, 500)
    g = 1 / (1 + np.exp(-z))

    return g


def compute_cost_logistic(X, y, w, b):
    """
    计算逻辑回归的 loss 函数
    :param X: 输入样例
    :type X: np.ndarray
    :param y: 目标值
    :type y: np.ndarray
    :param w: 模型参数
    :type w: np.ndarray
    :param b: 模型参数
    :return:
    """
    m = X.shape[0]
    cost = 0.0

    for i in range(m):
        z_i = X[i] @ w + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost /= m
    return cost

