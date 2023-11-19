import numpy as np
import matplotlib.pyplot as plt


def load_data() -> (np.ndarray, np.ndarray, np.ndarray):
    """
    读取数据
    """
    X = np.load("data/X_part1.npy")
    X_val = np.load("data/X_val_part1.npy")
    y_val = np.load("data/y_val_part1.npy")
    return X, X_val, y_val


def load_data_multi() -> (np.ndarray, np.ndarray, np.ndarray):
    """
    读取多维特征的数据
    """
    X = np.load("data/X_part2.npy")
    X_val = np.load("data/X_val_part2.npy")
    y_val = np.load("data/y_val_part2.npy")
    return X, X_val, y_val


def estimate_gaussian(X: np.ndarray):
    """
    计算所有特征的均值和方差
    """
    m, n = X.shape
    mu = X.mean(axis=0)
    var = np.sum((X - mu) ** 2, axis=0) / m
    return mu, var


def multivariate_gaussian(X: np.ndarray, mu: np.ndarray, var: np.ndarray):
    """
    计算可能性
    :param X: (m, n)
    :param mu: (m, )
    :param var: (m, )
    """
    m = mu.shape[0]

    # 方差如果是一维，转换为对角阵
    if var.ndim == 1:
        var = np.diag(var)

    X = X - mu
    p = ((2 * np.pi) ** (-m / 2)  # 2 pi
         * np.linalg.det(var) ** (-0.5)  # sigma 行列式的值
         * np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1)))  # exp 相乘，指数相加即可

    return p


def visualize_fit(X, mu, var, p=None):
    """
    This visualization shows you the
    probability density function of the Gaussian distribution. Each example
    has a location (x1, x2) that depends on its feature values.
    """

    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)

    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10 ** (np.arange(-20., 1, 3)), linewidths=1)

    if p:
        # 绘出最佳 F1
        level = [p, p * 11 / 10]
        plt.contour(X1, X2, Z, levels=level, colors='r')

    # Set the title
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    # Set the y-axis label
    plt.ylabel('Throughput (mb/s)')
    # Set the x-axis label
    plt.xlabel('Latency (ms)')


def select_threshold(y_val: np.ndarray, p_val: np.ndarray) -> (float, float):
    """
    选择最好的阈值，使 F1 最高
    :param y_val: 目标值
    :param p_val: 目标概率
    :return:
    """
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    # 步长
    step_size = (p_val.max() - p_val.min()) / 1000

    for epsilon in np.arange(p_val.min(), p_val.max(), step_size):
        # 预测精准度
        predictions = (p_val < epsilon)

        tp = sum((predictions == 1) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        fn = sum((predictions == 0) & (y_val == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)

        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon

    return best_epsilon, best_F1
