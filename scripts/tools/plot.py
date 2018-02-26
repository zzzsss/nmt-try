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
# 100: 37.11, 37.34, 35.99, 37.88
# zzzzz 0b-30 b'BLEU = 37.34, 74.3/46.8/29.8/19.1 (BP=0.995, ratio=0.995, hyp_len=170314, ref_len=171134)\n'
# zzzzz 0c-30 b'BLEU = 37.20, 76.4/49.0/31.6/20.6 (BP=0.942, ratio=0.943, hyp_len=156067, ref_len=165416)\n'
# zzzzz 1c-30 b'BLEU = 36.83, 73.1/46.0/29.2/18.7 (BP=1.000, ratio=1.016, hyp_len=176182, ref_len=173404)\n'
# zzzzz 1e-30 b'BLEU = 37.87, 75.1/47.8/30.7/19.8 (BP=0.986, ratio=0.986, hyp_len=166383, ref_len=168794)\n'
# zzzzz 0b-50 b'BLEU = 37.38, 74.0/46.6/29.7/19.1 (BP=1.000, ratio=1.002, hyp_len=172197, ref_len=171805)\n'
# zzzzz 0c-50 b'BLEU = 37.33, 76.4/49.0/31.7/20.7 (BP=0.943, ratio=0.944, hyp_len=156137, ref_len=165325)\n'
# zzzzz 1c-50 b'BLEU = 36.46, 72.5/45.5/28.9/18.5 (BP=1.000, ratio=1.027, hyp_len=179244, ref_len=174475)\n'
# zzzzz 1e-50 b'BLEU = 37.95, 74.8/47.6/30.6/19.8 (BP=0.990, ratio=0.990, hyp_len=167845, ref_len=169464)\n'
# zzzzz 0b-100 b'BLEU = 37.11, 73.5/46.3/29.5/18.9 (BP=1.000, ratio=1.012, hyp_len=174812, ref_len=172784)\n'
# zzzzz 0c-100 b'BLEU = 37.34, 76.4/49.0/31.7/20.7 (BP=0.943, ratio=0.945, hyp_len=156246, ref_len=165371)\n'
# zzzzz 1c-100 b'BLEU = 35.99, 71.8/44.9/28.5/18.3 (BP=1.000, ratio=1.039, hyp_len=182288, ref_len=175401)\n'
# zzzzz 1e-100 b'BLEU = 37.88, 74.4/47.2/30.3/19.6 (BP=0.997, ratio=0.997, hyp_len=169158, ref_len=169670)\n'

# \lambda = 0.4
data_ende = (
    [23.57, 24.32, 24.57, 24.71, 24.81, 24.85, 24.86, 24.90, 24.90, 24.87, 24.89],  # +lr
    [23.57, 24.25, 24.45, 24.55, 24.65, 24.66, 24.67, 24.70, 24.67, 24.68, 24.68],  # +lr+norm
    [23.57, 24.40, 24.74, 24.82, 24.85, 24.86, 24.89, 24.92, 24.88, 24.81, 24.71],  # +merge+lr
    [23.57, 24.32, 24.59, 24.64, 24.66, 24.65, 24.68, 24.69, 24.71, 24.75, 24.71],  # +merge+lr+norm
)
#
# zzzzz n4-50 b'BLEU = 24.89, 57.2/30.6/18.6/11.8 (BP=1.000, ratio=1.001, hyp_len=169670, ref_len=169521)\n'
# zzzzz nn-50 b'BLEU = 24.68, 56.8/30.4/18.4/11.7 (BP=1.000, ratio=1.018, hyp_len=172522, ref_len=169521)\n'
# zzzzz m40-50 b'BLEU = 24.71, 56.9/30.4/18.4/11.7 (BP=1.000, ratio=1.011, hyp_len=171384, ref_len=169521)\n'
# zzzzz m41-50 b'BLEU = 24.71, 57.4/30.7/18.6/11.9 (BP=0.989, ratio=0.989, hyp_len=167622, ref_len=169521)\n'
# zzzzz n4-100 b'BLEU = 24.88, 57.2/30.6/18.6/11.9 (BP=0.998, ratio=0.998, hyp_len=169120, ref_len=169521)\n'
# zzzzz nn-100 b'BLEU = 24.65, 56.8/30.3/18.4/11.7 (BP=1.000, ratio=1.017, hyp_len=172484, ref_len=169521)\n'
# zzzzz m40-100 b'BLEU = 24.67, 56.8/30.3/18.4/11.7 (BP=1.000, ratio=1.011, hyp_len=171372, ref_len=169521)\n'
# zzzzz m41-100 b'BLEU = 24.59, 57.4/30.7/18.7/11.9 (BP=0.983, ratio=0.983, hyp_len=166618, ref_len=169521)\n'

if __name__ == '__main__':
    draw(data_zhen)
    draw(data_ende)
