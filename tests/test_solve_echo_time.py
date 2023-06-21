import numpy as np
from matplotlib import pyplot as plt

from microtool.scanner_parameters import default_scanner
from microtool.utils.solve_echo_time import minimal_echo_time


def test_solve_echo_time():
    """
    The relation found should present as a cubic
    """
    b = np.linspace(0.05, 3, num=500) * 1e3  # s/mm^2

    print(minimal_echo_time(b, default_scanner))
    plt.plot(minimal_echo_time(b, default_scanner), b)
    plt.xlabel("TE_min [s]")
    plt.ylabel(r"b [s/mm$^2$]")
    plt.tight_layout()
    plt.show()
