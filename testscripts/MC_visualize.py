import pathlib

from matplotlib import pyplot as plt
import math
from microtool.utils_IO import get_df_from_pickle, get_pickle

resultdir = pathlib.Path("MC_results")
filename = "TPD_alexander2008_nsim10_npulse2_noise0.02.pkl"
df = get_df_from_pickle(resultdir / filename)
gt = get_pickle(resultdir / "alexander2008_ground_truth.pkl")

n_rows = math.ceil(df.shape[1] / 3)

for i, parameter in enumerate(df.keys()):
    ax = plt.subplot(n_rows, 3, i + 1)
    # making the histogram
    ax.hist(df[parameter], bins=20)
    ax.vlines(gt[parameter], 0, 1, transform=ax.get_xaxis_transform(), colors="red", label="Ground Truth")
    ax.set_title(parameter)

plt.tight_layout()
plt.show()
