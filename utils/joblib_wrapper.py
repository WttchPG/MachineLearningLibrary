import joblib
from functools import wraps
import os.path


def joblib_wrapper(save_path, use_cache=True):
    """
    装饰器，训练完保存模型，如果模型存在则加载。
    :param use_cache: 是否使用缓存
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
            if use_cache and os.path.exists(save_path):
                print(f'{train_func.__name__}发现位于{save_path}的任务结果')
                return joblib.load(save_path)
            else:
                print('开始任务...')
                model = train_func()
                joblib.dump(model, save_path)
                print(f'任务完成, 保存{type(model)} -> {save_path}')
                return model

        return inner

    return wrapper
