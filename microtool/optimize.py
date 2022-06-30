
from typing import Sequence, Callable, Tuple, List
from dataclasses import dataclass,field,astuple, asdict
import numpy as np
from scipy.optimize.optimize import OptimizeResult
from tqdm.contrib.itertools import product

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
    information = fisher_information((jac * scales), noise_var)

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

    def __call__(self, fun: callable, x0: np.ndarray, args=(), **options) -> OptimizeResult:
        """
        :param fun: The objective function that we wish to minimize
        :param x0: starting values for the parameters we wish to optimize in this case not used!!!
        :param args: I have no idea why this is here, defaults to ()
        :param bounds: This needs to be provided otherwise bruteforce can't be used, defaults to None
        :param constraints: , defaults to None
        :return: OptimizeResult object from scipy
        """
        bounds = options['bounds']
        constraints = options['constraints']
        # this checks if boundaries actually contains values for lower bound and upperbound
        check_bounded(bounds)

        nx = len(x0)
        # make the individual discretized domains
        domains = []
        previous_bound = ()
        for bound in bounds:
            if bound == previous_bound:
                # making a pointer to the previous array if the same bounds are provided sequentially
                domains.append(domains[-1].view())
            else:
                domains.append(np.linspace(bound[0], bound[1], num=self.Ns))

        x_optimal, y_optimal = compute_losses(fun, domains, constraints)
        return OptimizeResult(fun=y_optimal, x=x_optimal, succes=True)


def compute_losses(fun: callable, domains: List[np.ndarray], constraints) -> Tuple[np.ndarray, float]:
    if constraints != ():
        A = constraints.A
        lb = constraints.lb
        ub = constraints.ub

        # iterate over the grid
        y_optimal = np.inf
        x_optimal = None
        for combination in product(*domains, desc='Running brute force grid computation'):
            combination = np.array(combination)
            # check constraint
            if is_constrained(combination, A, lb, ub):
                loss = np.inf
            else:
                loss = fun(combination)

            # update optimal value
            if loss < y_optimal:
                x_optimal = combination
                y_optimal = loss
    else:
        # iterate over the grid
        y_optimal = np.inf
        x_optimal = None
        for combination in product(*domains, desc='Running brute force grid computation'):
            combination = np.array(combination)
            loss = fun(combination)
            # update optimal value
            if loss < y_optimal:
                x_optimal = combination
                y_optimal = loss

    return x_optimal, y_optimal
