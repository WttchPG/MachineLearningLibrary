import numpy as np
import matplotlib.pyplot as plt
from compute import compute_cost


def plot_cost_i_w(X, y, hist):
    ws = np.array([p[0] for p in hist["params"]])
    rng = max(abs(ws[:, 0].min()), abs(ws[:, 0].max()))
    wr = np.linspace(-rng + 0.27, rng + 0.27, 20)
    cst = [compute_cost(X, y, np.array([wr[i], -32, -67, -1.46]), 221) for i in range(len(wr))]

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(hist["iter"], (hist["cost"]))
    ax[0].set_title("Cost vs Iteration")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("Cost")
    ax[1].plot(wr, cst)
    ax[1].set_title("Cost vs w[0]")
    ax[1].set_xlabel("w[0]")
    ax[1].set_ylabel("Cost")
    ax[1].plot(ws[:, 0], hist["cost"])
    plt.show()
