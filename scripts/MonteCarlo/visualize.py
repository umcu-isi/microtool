import pathlib

from matplotlib import pyplot as plt
import math
from microtool.utils.IO import get_df_from_pickle, get_pickle

currentdir = pathlib.Path(__file__).parent
resultdir = currentdir / "results"

filename = "alexander_nofixed_n_sim_2_noise_0.02.pkl"
df = get_df_from_pickle(resultdir / filename)
gt = get_pickle(resultdir / "alexander2008_ground_truth.pkl")

n_rows = math.ceil(df.shape[1] / 3)

print(df.describe())
for i, parameter in enumerate(df.keys()):
    ax = plt.subplot(n_rows, 3, i + 1)

    gt_parameter = gt[parameter]
    scale = gt_parameter.scale
    value = gt_parameter.value / gt_parameter.scale

    # making the histogram
    ax.hist(df[parameter] / scale - value, bins=100)
    ax.set_xlabel(r"$\Delta$")
    # plotting ground truth as vertical lines
    ax.vlines(0, 0, 1, transform=ax.get_xaxis_transform(), colors="red", label="Ground Truth")
    ax.set_title(parameter)

plt.tight_layout()
plt.show()
