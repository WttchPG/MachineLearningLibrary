import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml
import pandas as pd
from utils.joblib_wrapper import joblib_wrapper


def load_mnist():
    data_df = fetch_openml('mnist_784', cache=True, data_home='../datasets', as_frame=True, parser='auto')[
        'frame']  # type: pd.DataFrame
    data_train = data_df.iloc[:60000, :].to_numpy()  # type: np.ndarray
    data_test = data_df.iloc[60000:, :].to_numpy()

    return normalizer(data_train[:, :-1]), one_hot_label(data_train[:, -1]), normalizer(
        data_test[:, :-1]), one_hot_label(data_test[:, -1])


def normalizer(data: np.ndarray) -> np.ndarray:
    return data.astype(float) / 255.0


def one_hot_label(X: np.ndarray):
    X = X.astype(int)
    ret = np.zeros((X.size, 10))
    for idx, row in enumerate(ret):
        row[X[idx]] = 1

    return ret


def to_image(data: np.ndarray) -> Image:
    image = data.reshape(28, 28)
    return Image.fromarray(np.uint8(image))
