import numpy as np
import matplotlib.pyplot as plt
# interact 与 jupyter 交互
from ipywidgets import interact

import util
from compute import *

plt.style.use("./deeplearning.mplstyle")
dlblue = '#0096ff'
dlorange = '#FF9300'
dldarkred = '#C00000'
dlmagenta = '#FF40FF'
dlpurple = '#7030A0'
dlcolors = [dlblue, dlorange, dldarkred, dlmagenta, dlpurple]

data = util.load_data()
# 面积 m^2
x = np.array([i[0] for i in data])
# 金钱 万元
y = np.array([i[2] / 1000 for i in data])


def show_model_plt(w, b):
    fwb = compute_model_output(x, w, b)

    plt.plot(x, fwb, c='b', label="Our Prediction", linewidth=1)
    plt.scatter(x=x, y=y, marker='x', c='r', linewidths=1, label="Actual Values")
    plt.title(label="House Prices")
    plt.ylabel(ylabel="Price($k)")
    plt.xlabel(xlabel="Size(feet^2)")
    plt.legend()
    plt.show()


def plt_house_x(X, y, f_wb=None, ax=None):
    ''' plot house with aXis '''
    if not ax:
        fig, ax = plt.subplots(1, 1)
    ax.scatter(X, y, marker='x', c='r', label="Actual Value")

    ax.set_title("Housing Prices")
    ax.set_ylabel('Price (in 1000s of dollars)')
    ax.set_xlabel(f'Size (1000 sqft)')
    if f_wb is not None:
        ax.plot(X, f_wb, c=dlblue, label="Our Prediction")
    ax.legend()


def show_cost_function_plt(x_train=x, y_train=y):
    """
    展示代价函数
    :param x_train: 训练用的x数组
    :type x_train: ndarray
    :param y_train: 训练用的y数组
    :type y_train: ndarray
    :return:
    """
    # w 的范围
    w_range = np.array([-0.2, 0.4])
    tmp_b = 100
    step = 0.01
    w_array = np.arange(*w_range, step)
    cost = np.zeros_like(w_array)
    for i in range(len(w_array)):
        tmp_w = w_array[i]
        cost[i] = compute_cost(x_train, y_train, tmp_w, tmp_b)

    # 和文档交互 w 值的范围, 步进
    @interact(w=(*w_range, step), continuous_update=False)
    def func(w=0.2):
        f_wb = np.dot(x_train, w) + tmp_b

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 4))
        fig.canvas.toolbar_position = 'bottom'

        # 展示价格和线性模型
        plt_house_x(x_train, y_train, f_wb=f_wb, ax=ax[0])

        ax[1].plot(w_array, cost)
        cur_cost = compute_cost(x_train, y_train, w, tmp_b)
        ax[1].scatter(w, cur_cost, s=100, color=dldarkred, zorder=10, label=f"cost at w={w}")
        ax[1].hlines(cur_cost, ax[1].get_xlim()[0], w, lw=4, color=dlpurple, ls='dotted')
        ax[1].vlines(w, ax[1].get_ylim()[0], cur_cost, lw=4, color=dlpurple, ls='dotted')
        ax[1].set_title("Cost vs. w, (b fixed at {})".format(tmp_b))
        ax[1].set_ylabel('Cost')
        ax[1].set_xlabel('w')
        ax[1].legend(loc='upper center')
        fig.suptitle(f"Minimize Cost: Current Cost = {cur_cost:0.0f}", fontsize=12)
        plt.show()


def soup_bowl():
    """ Create figure and plot with a 3D projection"""
    fig = plt.figure(figsize=(8, 8))

    # Plot configuration
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_rotate_label(False)
    ax.view_init(45, -120)

    # 对 w b 进行范围切片
    w_array = np.linspace(-0.4, 0.4, 100)
    b_array = np.linspace(-1000, 1000, 100)

    # Get the z value for a bowl-shaped cost function
    z = np.zeros((len(w_array), len(b_array)))
    j = 0
    for w in w_array:
        i = 0
        for b in b_array:
            z[i, j] = compute_cost(x, y, w, b)
            i += 1
        j += 1

    # Meshgrid used for plotting 3D functions
    W, B = np.meshgrid(w_array, b_array)

    # Create the 3D surface plot of the bowl-shaped cost function
    ax.plot_surface(W, B, z, cmap="Spectral_r", alpha=0.7, antialiased=False)
    ax.plot_wireframe(W, B, z, color='k', alpha=0.1)
    ax.set_xlabel("$w$")
    ax.set_ylabel("$b$")
    ax.set_zlabel("$J(w,b)$", rotation=90)
    ax.set_title("$J(w,b)$", size=15)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].contour(w_array, b_array, z, levels=[2000, 2500, 3000, 4000, 6000, 8000, 16000, 24000, 48000, 72000, 100000],
                   linewidths=1)

    plt.show()
