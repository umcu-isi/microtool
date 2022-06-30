"""
Helper functions for handling the input and output for this module. Especially usefull for the Monte Carlo stuff.
But also thinks like hiding print statements
"""
import pickle
import pandas as pd
import numpy as np
from typing import Union, List, Dict
from os import PathLike
import os
import sys


class HiddenPrints:
    """
    Helper class for silencing print statements. Use with HiddenPrints(): whatever you wanna do without print statements
    """
    def __enter__(self):
        """
        Is called when the object is constructed: sets the output steam to null (everything is written to
        nowhere)
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Resetting the out stream to the original outstream before creation of this object"""
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_df_from_pickle(path: Union[str, bytes, PathLike]) -> pd.DataFrame:
    """

    :param path: Path to a .pkl file containing dmipy MonteCarlo result
    :return: a pandas dataframe of the parameter distributions aquired during the monte carlo simulation
    """
    parameter_list = get_pickle(path)
    parameter_dict = collapse_dict(parameter_list)
    better_parameter_dict = unpack_vectors(parameter_dict)
    return pd.DataFrame(better_parameter_dict)


# TODO: better performence if unpack vectors and collapse dict are merged
def collapse_dict(parameter_list: List[Dict[str, Union[float, np.ndarray]]]) -> Dict[str, list]:
    # Making one big dictionary out of the list of seperate parameters (dictionaries)
    collapsed = {}
    for parameters in parameter_list:
        for key, value in parameters.items():
            if key in collapsed:
                # unpacking the first weird layer
                collapsed[key].append(value[0])
            else:
                collapsed[key] = [value[0]]
    return collapsed


def unpack_vectors(parameters: Dict[str, list]) -> Dict[str, Union[np.ndarray, float]]:
    # unpacking vector parameters into components
    unpacked = {}
    for key, value in parameters.items():
        shp = np.shape(value)

        if len(shp) > 2:
            raise ValueError("Expected vector or float type tissue parameters.")
        if len(shp) == 2:
            # Storing the components as seperate dictionary entries
            for i in range(shp[1]):
                newkey = key + f"_{i}"
                unpacked[newkey] = np.array(value)[:, i]
        else:
            unpacked[key] = value

    return unpacked


def get_pickle(path):
    # shorthand for unpickling
    with open(path, 'rb') as f:
        return pickle.load(f)
