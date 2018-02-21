# plot the results

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as scale
import matplotlib.patches as patches

def plot(x, y, pattern, **kwargs):
    rx = x[:len(y)]
    plt.plot(rx, y, pattern, lw=1.5, markersize=5, **kwargs)

# beam-size=10, ZH-EN-03to06
def draw(ys):
    y1, y2, y3, y4 = ys
    xlabels = np.asarray([1,2,4,6,8,10,12,16,20,30,50,100])
    x = np.log2(xlabels+1)
    plt.figure(figsize=(10,6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    ax = plt.gca()
    # ax.set_xscale()
    ax.set_xticklabels(xlabels)
    ax.set_xticks(x)
    plot(x, y1, "b--d", label="base + lr")
    plot(x, y2, "k-.x", label="base + norm")
    plot(x, y3, "m-+", label="merge + lr")
    plot(x, y4, "r-*", label="merge + norm")
    # for i in range(34, 38):
    #     plt.axhline(y=i, ls="--", lw=0.5, color="black")
    plt.legend(loc='lower right', fontsize=22)
    plt.ylabel('BLEU', fontsize=22)
    plt.xlabel('Beam Size (in log scale)', fontsize=22)
    plt.grid()
    plt.tight_layout()
    plt.show()

# \lambda = 1.0
data_zhen = (
    [33.09, 35.20, 36.48, 36.82, 37.03, 37.13, 37.12, 37.24, 37.20, 37.34, 37.38],  # +lr
    [33.09, 35.14, 36.42, 36.71, 36.94, 37.07, 37.11, 37.18, 37.17, 37.20, 37.33],  # +lr+norm
    [33.09, 35.41, 36.68, 37.03, 37.16, 37.27, 37.23, 37.06, 36.92, 36.83, 36.46],  # +merge+lr
    [33.09, 35.39, 36.78, 37.17, 37.36, 37.51, 37.56, 37.68, 37.76, 37.87, 37.95],  # +merge+lr+norm
)

# \lambda = 0.4
data_ende = (
    [23.57, 24.32, 24.57, 24.71, 24.81, 24.85, 24.86, 24.90, 24.90, 24.87, 24.89],  # +lr
    [23.57, 24.25, 24.45, 24.55, 24.65, 24.66, 24.67, 24.70, 24.67, 24.68, 24.68],  # +lr+norm
    [23.57, 24.40, 24.74, 24.82, 24.85, 24.86, 24.89, 24.92, 24.88, 24.81, 24.71],  # +merge+lr
    [23.57, 24.32, 24.59, 24.64, 24.66, 24.65, 24.68, 24.69, 24.71, 24.75, 24.71],  # +merge+lr+norm
)

if __name__ == '__main__':
    draw(data_zhen)
    draw(data_ende)
