"""
Helper functions for handling the input and output for the microtool module.
"""
import os
import pickle
import sys
from typing import Any


class HiddenPrints:
    """
    Helper class for silencing print statements. Use with HiddenPrints(): whatever you want to do without print statements
    """

    def __enter__(self):
        """
        Is called when the object is constructed: sets the output stream to null (everything is written to
        nowhere)
        """
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Resetting the output stream to the original output stream before creation of this object"""
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_pickle(path) -> Any:
    # shorthand for unpickling
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, path) -> None:
    # shorthand for pickling
    with open(path, "wb") as f:
        pickle.dump(obj, f)
