import pathlib

from matplotlib import pyplot as plt

from microtool.utils import get_df_from_pickle

resultdir = pathlib.Path("MC_results")
filename = "TPD_dmipy_large.pkl"
better_df = get_df_from_pickle(resultdir/filename)
better_df.hist()
plt.show()
