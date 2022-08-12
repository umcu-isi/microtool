import math
import pathlib
import pprint

from matplotlib import pyplot as plt

from microtool.utils.IO import get_df_from_pickle, get_pickle

resultdir = pathlib.Path(__file__).parent / 'results'

gt = get_pickle(resultdir / "alexander2008_ground_truth.pkl")

files = ["alexander_shells_[8, 32, 80]_n_sim_100_noise_0.02.pkl",
         "alexander_shells_[20, 20, 80]_n_sim_100_noise_0.02.pkl",
         "alexander_shells_[30, 30, 60]_n_sim_100_noise_0.02.pkl",
         "alexander_shells_[40, 40, 40]_n_sim_100_noise_0.02.pkl"]


def main():
    for filename in files:
        visualize_pickle(filename)

    plt.legend()
    plt.show()


def visualize_pickle(filename: str) -> None:
    df = get_df_from_pickle(resultdir / filename)
    n_rows = math.ceil(df.shape[1] / 3) + 1

    # print(df.describe())
    for i, parameter in enumerate(df.keys()):
        ax = plt.subplot(n_rows, 3, i + 1)

        gt_parameter = gt[parameter]
        scale = gt_parameter.scale
        value = gt_parameter.value / gt_parameter.scale

        # making the histogram

        ax.hist(df[parameter] / scale - value, bins='scott', alpha=0.5, label=filename.split('_')[2])
        ax.set_xlabel(r"$\Delta$")
        # plotting ground truth as vertical lines
        ax.vlines(0, 0, 1, transform=ax.get_xaxis_transform(), colors="black")
        ax.set_title(parameter)

    # Adding a table for the ground truth values
    gt_dict = gt.parameters
    plt.tight_layout()


if __name__ == "__main__":
    main()
