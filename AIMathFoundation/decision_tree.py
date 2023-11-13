# ID3
import numpy as np
import pandas as pd


def split_dataset(dataset: pd.DataFrame, axis, value):
    """
    根据特征划分数据
    :param dataset: 数据集
    :param axis: 轴
    :param value: 划分值
    :return: 划分的数据集合
    """
    return dataset[dataset[axis] == value]


def prob_by_axis(dataset: pd.DataFrame, axis):
    """
    计算给定维度的可能值
    :param dataset:
    :param axis:
    :return:
    """
    group_data = dataset.groupby(axis)[axis]

    value_counts = group_data.value_counts()
    _sum = value_counts.sum()

    _prob = value_counts / _sum  # type: pd.DataFrame
    return _prob


def calc_shanon_ent(dataset: pd.DataFrame, axis):
    """
    计算信息熵
    :param dataset: 数据集
    :return:
    """
    prob = prob_by_axis(dataset, axis)  # type: pd.DataFrame

    entropy = - prob * np.log2(prob)  # type: pd.DataFrame

    return entropy.sum()


def calc_conditional_ent(dataset: pd.DataFrame, axis, uniqueVals):
    """
    计算条件熵
    :param dataset:
    :param axis: 维度，特征名称
    :param uniqueVals: 数据集特征集合
    :return:
    """
    ce = 0.0
    for value in uniqueVals:
        sub_dataset = split_dataset(dataset, axis, value)
        prob = len(sub_dataset) / float(len(dataset))
        ce += prob * calc_shanon_ent(dataset, axis)
    return ce


def calc_information_gain(dataset: pd.DataFrame, base_entropy, axis):
    """
    计算信息增量
    :param dataset:
    :param base_entropy:
    :param i:
    :return:
    """
    unique_values = dataset[axis].unique()
    new_entropy = calc_conditional_ent(dataset, axis, uniqueVals=unique_values)

    return base_entropy - new_entropy


def best_feat_get_sub_dataset(dataset: pd.DataFrame):
    """
    算法框架
    :param dataset:
    :return:
    """
    first_column = dataset.columns[0]
    base_entropy = calc_shanon_ent(dataset, first_column)
    best_info_gain = 0.0
    best_feature = first_column

    for column in dataset.columns:
        info_gain = calc_information_gain(dataset, base_entropy, column)
        print(f"{column}: {info_gain}")
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = column

    return best_feature


def majority_cnt(classes: np.ndarray):
    """
    特征若已经划分完，节点下的样本还没有统一取值，进行投票
    :param classes:
    :return:
    """
    return ''
