import pathlib

from matplotlib import pyplot as plt
import seaborn as sns

from microtool.utils import get_df_from_pickle

resultdir = pathlib.Path("MC_results")
# insrt filename here
filename = "TPD_relaxation.pkl"
better_df = get_df_from_pickle(resultdir/filename)

better_df.hist()

plt.figure()
sns.histplot(better_df, x="C1Stick_1_mu_0" , y = "C1Stick_1_mu_1")
plt.title(r"$N_{sim} = 10000$")
plt.show()
