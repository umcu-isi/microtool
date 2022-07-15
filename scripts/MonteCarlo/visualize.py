import math
import pathlib
import pprint

from matplotlib import pyplot as plt

from microtool.utils.IO import get_df_from_pickle, get_pickle

resultdir = pathlib.Path(__file__).parent / 'results'
filename = "alexander_nofixed_n_sim_1000_noise_0.02.pkl"
df = get_df_from_pickle(resultdir / filename)
gt = get_pickle(resultdir / "alexander2008_ground_truth.pkl")

n_rows = math.ceil(df.shape[1] / 3) + 1

print(df.describe())
for i, parameter in enumerate(df.keys()):
    ax = plt.subplot(n_rows, 3, i + 1)

    gt_parameter = gt[parameter]
    scale = gt_parameter.scale
    value = gt_parameter.value / gt_parameter.scale

    # making the histogram
    ax.hist(df[parameter] / scale - value, bins='scott')
    ax.set_xlabel(r"$\Delta$")
    # plotting ground truth as vertical lines
    ax.vlines(0, 0, 1, transform=ax.get_xaxis_transform(), colors="red", label="Ground Truth")
    ax.set_title(parameter)

# Adding a table for the ground truth values
gt_dict = gt.parameters
pprint.pprint(gt_dict)

plt.tight_layout()
plt.show()
