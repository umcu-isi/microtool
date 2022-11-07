"""
Helper functions for handling the input and output for this module. Especially usefull for the Monte Carlo stuff.
But also thinks like hiding print statements
"""
import os
import pickle
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


def get_pickle(path):
    # shorthand for unpickling
    with open(path, 'rb') as f:
        return pickle.load(f)
