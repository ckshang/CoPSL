import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerPathCollection


def das_dennis_recursion(ref_dirs, ref_dir, n_partitions, beta, depth):
    if depth == len(ref_dir) - 1:
        ref_dir[depth] = beta / (1.0 * n_partitions)
        ref_dirs.append(ref_dir[None, :])
    else:
        for i in range(beta + 1):
            ref_dir[depth] = 1.0 * i / (1.0 * n_partitions)
            das_dennis_recursion(ref_dirs, np.copy(ref_dir), n_partitions, beta - i, depth + 1)


def das_dennis(n_partitions, n_dim):
    if n_partitions == 0:
        return np.full((1, n_dim), 1 / n_dim)
    else:
        ref_dirs = []
        ref_dir = np.full(n_dim, np.nan)
        das_dennis_recursion(ref_dirs, ref_dir, n_partitions, n_partitions, 0)
        return np.concatenate(ref_dirs, axis=0)


class HandlerDashedCollection(HandlerPathCollection):
    def create_collection(self, orig_handle, sizes, *args, **kwargs):
        p = super().create_collection(orig_handle, sizes, *args, **kwargs)
        p.set_edgecolor(p.get_facecolor())  # Set edge color to face color for scatter plot
        return p


def load_pf():
    re31 = pd.read_csv('pf_re/reference_points_RE31.dat', header=None).values
    re32 = pd.read_csv('pf_re/reference_points_RE32.dat', header=None).values
    re33 = pd.read_csv('pf_re/reference_points_RE33.dat', header=None).values
    re34 = pd.read_csv('pf_re/reference_points_RE34.dat', header=None).values
    re37 = pd.read_csv('pf_re/reference_points_RE37.dat', header=None).values

    f1_pf = pd.read_csv('pf_re/reference_points_F1.dat', header=None).values
    re31_pf = []
    re32_pf = []
    re33_pf = []
    re34_pf = []
    re37_pf = []
    for i in range(len(re31)):
        re31_pf_ = list(map(float, re31[i][0].split()))
        re31_pf.append(re31_pf_)
    for i in range(len(re32)):
        re32_pf_ = list(map(float, re32[i][0].split()))
        re32_pf.append(re32_pf_)
    for i in range(len(re33)):
        re33_pf_ = list(map(float, re33[i][0].split()))
        re33_pf.append(re33_pf_)
    for i in range(len(re34)):
        re34_pf_ = list(map(float, re34[i][0].split()))
        re34_pf.append(re34_pf_)
    for i in range(len(re37)):
        re37_pf_ = list(map(float, re37[i][0].split()))
        re37_pf.append(re37_pf_)

    return f1_pf, re31_pf, re32_pf, re33_pf, re34_pf, re37_pf
