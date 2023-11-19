import copy
import math

import numpy as np
from numpy import ndarray


def compute_model_output(x, w, b):
    f"""
    计算模型输出, y-bar = x * w + b
    :param x:
    :type x: ndarray
    :param w:
    :type w: float
    :param b:
    :type b: float
    :return:
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


def compute_cost(x, y, w, b):
    """
    代价函数
    :param x: x数组
    :type x: ndarray
    :param y: y数组
    :type y: ndarray
    :param w:
    :type w: float
    :param b:
    :type b: float
    :return:
    """

    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


def compute_multiple_cost(X, y, w, b):
    """
    代价函数
    :param X:
    :param y:
    :param w:
    :param b:
    :return:
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)

    return cost


def compute_multiple_gradient(X, y, w, b):
    """
    计算多维梯度
    :param X:
    :type X: ndarray
    :param y:
    :type y: ndarray
    :param w:
    :type w: ndarray
    :param b:
    :type b: float
    :return:
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        dt = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + dt * X[i, j]
        dj_db = dj_db + dt

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def compute_multiple_gradient_descent(X, y, w_in, b_in, cost_func, gradient_func, alpha, num_iters):
    """
    多维梯度下降
    :param X:
    :param y:
    :param w_in:
    :param b_in:
    :param cost_func:
    :param gradient_func:
    :param alpha:
    :param num_iters:
    :return:
    """
    j_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    w_array = [w]
    b_array = [b]

    for i in range(num_iters):
        dj_dw, dj_db = gradient_func(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        w_array.append(w)
        b_array.append(b)

        if i < 100000:
            j_history.append(cost_func(X, y, w, b))

        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")

    return w_array, b_array, j_history


def compute_gradient_matrix(X, y, w, b):
    """
    计算矩阵梯度
    :param X: (m, n)
    :type X: ndarray
    :param y: (m, 1)
    :type y: ndarray
    :param w: (n, 1)
    :type w: ndarray
    :param b:
    :type b: float
    :return:
    """
    m, n = X.shape
    f_wb = X @ w + b
    e = f_wb - y
    # (f_wb - y) * x(i)
    # 第一行 X.T 代表 X1 的多个维度
    dj_dw = (1 / m) * (X.T @ e)
    dj_db = (1 / m) * np.sum(e)

    return dj_dw, dj_db


def gradient_descent_houses(X, y, w_in, b_in, cost_func, gradient_func, alpha, num_iters):
    """
    房屋价格梯度下降
    :param X: (m, n) 样例数据
    :param y: (m, 1) 目标数值
    :param w_in: (n,) w 的训练初始值
    :param b_in: float b 的训练初始值
    :param cost_func: 代价函数
    :param gradient_func: 计算梯度的函数
    :param alpha: 学习率
    :param num_iters: 迭代次数
    :return:
    """