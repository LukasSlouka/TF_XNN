from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .utils import get_cmap, get_position_of_one

SIZE_MIN = 100  # change here
SIZE_STEP = 100  # change here
SIZE_MAX = 1000  # change here


def sizes() -> List[int]:
    return sorted(list(range(SIZE_MIN, SIZE_MAX + 1, SIZE_STEP)))


def files() -> List[str]:
    return ['change_this_path/and_file_{}.log'.format(x) for x in sizes()]


if __name__ == '__main__':

    x = np.linspace(SIZE_MIN, SIZE_MAX, SIZE_MAX // SIZE_MIN)  # change here probably
    y = np.linspace(SIZE_MIN, SIZE_MAX, SIZE_MAX // SIZE_MIN)  # change here probably
    z = list()

    for file in files():
        sublist = np.array([val for _, val in np.genfromtxt(file, delimiter='\t')])
        z.append(sublist)

    x, y = np.meshgrid(x, y)
    z = np.array(z)

    fig, ax = plt.subplots()
    mid = get_position_of_one(min=z.min(), max=z.max())
    p = ax.pcolor(x, y, z, cmap=get_cmap(mid), alpha=0.7)
    cb = fig.colorbar(p)
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('Speedup', rotation=270)
    plt.xlabel('X_LABEL')
    plt.ylabel('Y_LABEL')
    plt.title('GRAPH_TITLE')
    plt.show()
