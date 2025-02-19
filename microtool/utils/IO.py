"""
Helper functions for handling the input and output for the microtool module.
"""
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Union


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

def save_pickle(obj: Any, path: Union[Path, str]) -> None:
    """
    Saves an object at a directory (used to save MonteCarlo simulations).

    :param obj: what is to be saved at a directory
    :param path: directory to be saved at
    """
    # shorthand for pickling
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def get_pickle(path) -> Any:
    """
    Retrieves an object from a directory.

    :param path: directory to retrieve obj from
    """
    # shorthand for unpickling
    with open(path, 'rb') as f:
        return pickle.load(f)

def initiate_logging_directory(root=None):
    """
    Creates a directory for log files in the current working directory, or in the optionally provided root directory.

    :return: Path to log file directory
    """
    if root is None:
        root = os.getcwd()

    log_dir = os.path.join(root, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir
