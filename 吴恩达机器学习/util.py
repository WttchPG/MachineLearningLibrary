import numpy as np


def load_data():
    def format_ret(line):
        """

        :param line:
        :type line: str
        :return:
        :rtype: []
        """
        return [float(value) for value in line.replace("\n", "").split(",")]

    ret = []
    with open("./data.txt", mode="r") as file:
        ret = [format_ret(line) for line in file.readlines()]
    return ret


def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',')
    X = data[:, :4]
    y = data[:, 4]
    return X, y


def zscore_normalize_features(X, rtn_ms=False):
    """
    标准化
    :param X:
    :type X: np.ndarray
    :param rtn_ms:
    :type rtn_ms: bool
    :return:
    """
    # 均值
    mu = np.mean(X, axis=0)
    # 标准差
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    if rtn_ms:
        return mu, sigma, X_norm
    else:
        return X_norm

