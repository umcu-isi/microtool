import pickle

import numpy as np
import pandas as pd

import pathlib
from matplotlib import pyplot as plt


def clean_dict(parameter_dict):
    newdict = {}
    for key, value in parameter_dict.items():
        newdict[key] = value[0]
    return newdict


resultdir = pathlib.Path("MC_results")

filename = "TPD_dmipy.pkl"
with open(resultdir / filename, "rb") as f:
    listdict = pickle.load(f)
    cleaned = map(clean_dict, listdict)
    df = pd.DataFrame(cleaned)
    df.hist()
    plt.show()
