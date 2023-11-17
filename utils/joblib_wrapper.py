import joblib
from functools import wraps
import os.path


def joblib_wrapper(save_path):
    """
    装饰器，训练完保存模型，如果模型存在则加载。
    :param save_path: 模型保存位置
    """

    # 外层为了获取 save_path 参数
    # wrapper 才是真正的装饰器
    def wrapper(train_func):
        """

        :param train_func: 训练模型的函数
        """
        @wraps(train_func)
        def inner():
            if os.path.exists(save_path):
                print(f'{train_func.__name__}发现位于{save_path}的模型')
                return joblib.load(save_path)
            else:
                print('开始训练...')
                model = train_func()
                print(f'训练完成, 保存{type(model)} -> {save_path}')
                return model

        return inner

    return wrapper
