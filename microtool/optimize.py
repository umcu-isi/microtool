from typing import Sequence, Callable, Tuple, List, Optional
import itertools

from .acquisition_scheme import AcquisitionScheme
from numba import jit
from numba.experimental import jitclass
import numpy as np
from scipy.optimize.optimize import OptimizeResult
from scipy.optimize import LinearConstraint, brute
import matplotlib.pyplot as plt
from copy import deepcopy

# A LossFunction takes an N×M Jacobian matrix, a sequence of M parameter scales, a boolean sequence that specifies which
# parameters should be included in the loss, and the noise variance. It should return a scalar loss.
LossFunction = Callable[[np.ndarray, Sequence[float], Sequence[bool], float], float]


def fisher_information(jac: np.ndarray, noise_var: float) -> np.ndarray:
    """
    Calculates the Fisher information matrix, assuming Gaussian noise.
    This is the sum of the matrices of squared gradients, for all samples, divided by the noise variance.
    See equation A2 in Alexander, 2008 (DOI 0.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param noise_var: Noise variance.
    :return: The M×M information matrix.
    """
    # TODO: Add Rician noise version, as explained in the appendix to Alexander, 2008 (DOI 0.1002/mrm.21646).
    return (1 / noise_var) * jac.T @ jac


def crlb_loss(jac: np.ndarray, scales: Sequence[float], include: Sequence[bool], noise_var: float) -> float:
    """
    Objective function for minimizing the total parameter variance (Cramer-Rao lower bounds), as defined in Alexander,
    2008 (DOI https://doi.org/10.1002/mrm.21646)

    :param jac: An N×M Jacobian matrix, where N is the number of samples and M is the number of parameters.
    :param scales: A sequence with M parameter scales.
    :param include: A boolean sequence specifying which parameters should be included in the loss.
    :param noise_var: Noise variance.
    :return: Estimated total weighted parameter variance.
    """
    # Calculate the Fisher information matrix on the rescaled Jacobian. The Cramer-Rao lower bound on parameter variance
    # is the inverse of the information matrix. A properly scaled matrix gives an interpretable condition number and a
    # more robust inverse.
    information = fisher_information(jac * scales, noise_var)

    # An ill-conditioned information matrix should result in a high loss.
    if np.linalg.cond(information) > 1e9:
        return 1e9

    # The loss is the sum of the diagonal values of Cramer-Rao lower bound matrix, which is the inverse information
    # matrix; see equation 2 in Alexander, 2008 (DOI 0.1002/mrm.21646). This is the same as the sum of the reciprocal
    # eigenvalues of the information matrix, which can be calculated at a lower computational cost because of symmetry.
    # Rescaling has already been done by multiplying the Jacobian by the parameter scales.
    return np.linalg.eigvalsh(information)[include].sum()


class Optimizer:
    """
    Optimizer base class. just to ensure all optimizers are compatible with the optimization routine....
    However we decide to do this in the future.
    """

    def __call__(self, fun: callable, x0: np.ndarray, args, **options) -> OptimizeResult:
        """
        This is the interface with scipy.optimize. So the child class should have exactly this __call__ implemented

        :param fun: loss function
        :param x0: initial parameter guess
        :param args: additional parameters to the loss function
        :param moreargs: optional arguments to scipy.optimize.minimize (see documentation scipy)
        :param kwargs: Keyword options to scipy optimize or whichever additional options you wish to add during
                       optimization.
        :return: Optimization result (for now using scipy wrapper)
        """
        raise NotImplementedError()

    @staticmethod
    def is_constrained(x: np.ndarray, constraints: LinearConstraint) -> bool:
        """A function for checking if a given parameter combination breaks a given linear constraint.

        :param x: Parameter combination
        :param constraints: scipy linear constraint object
        :return: boolean that is true if the parameter combination breaks the constraint
        """
        if constraints == ():
            return False
        # Readability variables
        A = constraints.A
        lb = constraints.lb
        ub = constraints.ub

        transformed_parameters = A @ x
        return np.any((lb >= transformed_parameters) | (transformed_parameters >= ub))

    @staticmethod
    def check_bounded(allbounds: List[Tuple[Optional[float], Optional[float]]]) -> None:
        """This function checks the boundedness of a set of given bounds such that brute force optimizers
        can assume boundedness after calling this function.

        :param allbounds: A list of bounds
        :raises ValueError: Raises a value error in case the there are no bounds or if bounds are to large
        """
        # Check for finite boundaries
        if allbounds is None:
            raise ValueError(
                " No bounds provided in optimize: this optimization can only be executed on a finite domain")
        for bounds in allbounds:
            for bound in bounds:
                if bound is None:
                    raise ValueError(" Infinite boundaries not supported for this optimizer")


class BruteForce(Optimizer):
    def __init__(self, Ns: int = 10):
        self.Ns = Ns

    @jit
    def __call__(self, fun: callable, x0: np.ndarray, args=(),
                 bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints=(),
                 **options) -> OptimizeResult:
        """
        Wrapping around the optimizer implemented as method to this class. Done s.t. this optimizer is compatible
        with the scipy.optimize interface.
        """
        """
        :param fun: The objective function that we wish to minimize
        :param x0: starting values for the parameters we wish to optimize in this case not used!!!
        :param args: I have no idea why this is here, defaults to ()
        :param bounds: This needs to be provided otherwise bruteforce can't be used, defaults to None
        :param constraints: , defaults to None
        :return: OptimizeResult object from scipy
        """

        # this checks if boundaries actually contains values for lower bound and upperbound
        self.check_bounded(bounds)

        nx = len(x0)
        # make the individual discretized domains
        domains = []
        for bound in bounds:
            domains.append(np.linspace(bound[0], bound[1], num=self.Ns))

        # iterate over the grid
        y_optimal = np.inf
        x_optimal = x0
        for combination in itertools.product(*domains):
            combination = np.array(combination)
            # check constraint
            if self.is_constrained(combination, constraints):
                loss = np.inf
            else:
                loss = fun(combination)

            # update optimal value
            if loss < y_optimal:
                x_optimal = combination
                y_optimal = loss

        return OptimizeResult(fun=y_optimal, x=x_optimal, succes=True)
