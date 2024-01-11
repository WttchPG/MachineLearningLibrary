import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.torch_util import get_device

# 数据集根目录
root_path = "/Volumes/WTTCH/datasets"


class HousePriceAdvancedRegressionTechniquesDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.train_features, self.train_labels, self.predict_data = load_house_prices_advanced_regression_techniques()
        self.shape = self.train_features.shape

    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, idx):
        return self.train_features[idx], self.train_labels[idx]


def load_house_prices_advanced_regression_techniques(
        numeric_norm=True, use_one_hot=True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    读取房价数据集 https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
    Args:
        numeric_norm: 是否归一化
        use_one_hot: 是否使用热独编码

    Returns:
        数据集合，数据标签, 测试数据集(没有数据标签)
    """
    folder = root_path + "/house-prices-advanced-regression-techniques"
    train_data = pd.read_csv(folder + "/train.csv").iloc[:, 1:]
    test_data = pd.read_csv(folder + "/test.csv").iloc[:, 1:]
    device = get_device()

    train_number = len(train_data)
    all_data = pd.concat([train_data, test_data])
    train_price = all_data.iloc[:train_number, -1]

    if numeric_norm:
        all_data.iloc[:, :-1] = numeric_normalization(all_data.iloc[:, :-1])

    if use_one_hot:
        all_data = one_hot_encode(all_data.iloc[:, :-1])

    all_data = all_data.astype(float)

    train_data = all_data.iloc[:train_number, :]  # type: pd.DataFrame
    prec_data = all_data.iloc[train_number:, :]  # type: pd.DataFrame

    return (torch.tensor(train_data.values, device=device, dtype=torch.float32),
            torch.tensor(train_price.values.reshape((-1, 1)), device=device, dtype=torch.float32),
            torch.tensor(prec_data.values, device=device, dtype=torch.float32))


def numeric_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    归一化所有数字字段
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: (x - x.mean()) / x.std()
    )
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    热独编码
    """
    return pd.get_dummies(df, dummy_na=True)
