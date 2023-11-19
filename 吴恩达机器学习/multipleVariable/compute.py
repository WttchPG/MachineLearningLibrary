import copy
import math

import numpy as np


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    代价函数
    :param X: 样本
    :param y: 目标值
    :param w: 模型参数
    :param b: 模型参数
    :return:
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)

    return cost


def compute_gradient(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    计算梯度
    :param X: 样本
    :param y: 目标值
    :param w: 模型参数
    :param b: 模型参数
    :return:
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err

    return dj_dw / m, dj_db / m


def compute_cost_matrix(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    计算代价函数矩阵
    :param X: 样本
    :param y: 目标值
    :param w: 模型参数
    :param b: 模型参数
    :return:
    """
    m, n = X.shape
    f_wb = X @ w + b
    err = f_wb - y

    return np.sum(err ** 2) / (2 * m)


def compute_gradient_matrix(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float):
    """
    计算梯度矩阵
    :param X:
    :param y:
    :param w:
    :param b:
    :return:
    """
    m, n = X.shape
    # 计算值
    f_wb = X @ w + b
    # 偏差
    err = f_wb - y
    # 偏差乘对应维变量求和
    dj_dw = (X.T @ err) / m
    # 偏差和
    dj_db = np.sum(err) / m

    return dj_dw, dj_db


def gradient_descent_houses(X: np.ndarray, y: np.ndarray, w_in: np.ndarray, b_in: float, cost_function,
                            gradient_function, alpha, num_iters):
    """
    房价梯度下降
    :param X: 样本值
    :param y: 目标值
    :param w_in: 输入模型变量
    :param b_in: 输入模型变量
    :param cost_function: 代价函数
    :param gradient_function: 梯度函数
    :param alpha: 学习率
    :param num_iters: 迭代次数
    :return:
    """
    m, n = X.shape

    # 迭代记录
    hist = {'cost': [], 'params': [], 'grads': [], 'iter': []}

    w = copy.deepcopy(w_in)
    b = b_in

    save_interval = np.ceil(num_iters / 10000)

    print(f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    "
          f"djdw3    djdb  ")
    print(f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|"
          f"--------|--------|")

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        # 新的模型值
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # 保存迭代记录
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

        # 打印迭代
        if i % math.ceil(num_iters / 10) == 0:
            cst = cost_function(X, y, w, b)
            print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} "
                  f"{dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")

    return w, b, hist


def run_gradient_descent(X, y, iterations=1000, alpha=1e-6):
    m, n = X.shape
    # initialize parameters
    initial_w = np.zeros(n)
    initial_b = 0
    # run gradient descent
    w_out, b_out, hist_out = gradient_descent_houses(X, y, initial_w, initial_b,
                                                     compute_cost, compute_gradient_matrix, alpha, iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")

    return w_out, b_out, hist_out
