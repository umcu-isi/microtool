from typing import Sequence, Callable, Tuple, List, Optional

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.optimize.optimize import OptimizeResult

from .optimize_utils import is_constrained, check_bounded

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


class BruteForce(Optimizer):
    def __init__(self, Ns: int = 10):
        self.Ns = Ns

    def __call__(self, fun: callable, x0: np.ndarray, args=(),
                 bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints=None,
                 **options) -> OptimizeResult:
        """
        :param fun: The objective function that we wish to minimize
        :param x0: starting values for the parameters we wish to optimize
        :param args: I have no idea why this is here, defaults to ()
        :param bounds: This needs to be provided otherwise bruteforce cant be used, defaults to None
        :param constraints: , defaults to None
        :return: OptimizeResult object from scipy
        """

        check_bounded(bounds)

        # Discretizing the domain to compute objective function values
        grid = self._make_grid(bounds)

        # Removing parameter combinations that are not conform constraints
        if constraints:
            grid = self._constrain_grid(grid, constraints)

        # Evaluating the objective function for all possible parameter combinations
        loss = np.zeros(grid.shape[0])
        for i in range(grid.shape[0]):
            loss[i] = fun(grid[i, :])

        # Getting the optimal values by finding the minimum loss.
        min_index = np.argmin(loss)
        x_optimal = grid[min_index, :]
        y_optimal = loss[min_index]

        return OptimizeResult(fun=y_optimal, x=x_optimal, succes=True)

    def _make_grid(self, bounds: List[Tuple[Optional[float], Optional[float]]]) -> np.ndarray:
        """Constructs a grid such that grid[i,:] contains the ith combination of parameters

        :param bounds: The range over which we wish to optimize
        :return: The grid with shape Ns x Nx that we can use to compute the objective function values
        """
        N = len(bounds)
        for k in range(N):
            if type(bounds[k]) is not type(slice(None)):
                if len(bounds[k]) < 3:
                    bounds[k] = tuple(bounds[k]) + (complex(self.Ns),)
                bounds[k] = slice(*bounds[k])
        if N == 1:
            bounds = bounds[0]

        grid = np.mgrid[bounds]

        # obtain an array of parameters that is iterable by a map-like callable
        inpt_shape = grid.shape
        if N > 1:
            grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T
        return grid

    @staticmethod
    def _constrain_grid(grid: np.ndarray, constraints: LinearConstraint) -> np.ndarray:
        """Drops parameter combinations from the grid that do not comply with the provided linear constraints.
        The constraint is formulated as lb <= A.x <= ub

        :param grid: The full parameter grid
        :param constraints: The LinearConstraints object used to constrain the grid
        :return: The grid values that satisfy the constraints
        """
        index_to_drop = []
        for i in range(grid.shape[0]):
            # check if result of above operations breaks inequalities
            if is_constrained(grid[i, :], constraints):
                index_to_drop.append(i)

        # if not satisfied drop parameter combination
        return np.delete(grid, np.array(index_to_drop), axis=0)
