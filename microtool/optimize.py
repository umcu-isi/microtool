from typing import Sequence, Callable, Tuple, List, Optional

from .acquisition_scheme import AcquisitionScheme
import numpy as np
from scipy.optimize.optimize import OptimizeResult
from scipy.optimize import LinearConstraint, brute
import matplotlib.pyplot as plt

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

    def __call__(self, fun: callable, x0: np.ndarray, args=(), *moreargs, **kwargs) -> OptimizeResult:
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


def plot_loss(result: OptimizeResult, fun: callable, domain_lengths: np.ndarray,
              plot_parameters: np.ndarray) -> None:
    """
    Allows for plotting a loss function as a function of 2 or 1 parameter(s) close to a minimum found by optimization.

    :param result: The optimization result we wish to plot.
    :param fun: the loss function used during optimization
    :param domain_lengths: The length of the parameter domains we wish to visualize
    :param plot_parameters: The index of the parameters we wish to visualize
    :return: None, instantiates a matplotlib.pyplot figure.
    """
    # checking if things are okay
    if len(plot_parameters) > 2:
        raise ValueError("Cant plot loss a function of more than 2 parameters")

    # extracting the minimum
    x_optimal = result['x']
    # extracting the parameters of interest value at the minimum
    parameters_optimal = x_optimal[plot_parameters]
    # discretizing the domain using the domain length
    domains = np.linspace(parameters_optimal - 0.5 * domain_lengths, parameters_optimal + 0.5 * domain_lengths,
                          endpoint=True)

    # 3d plots for 2 investigated parameters
    if len(plot_parameters) == 2:
        def loss(x1, x2):
            # return loss given two parameters we are changing and other parameters left constant
            params = x_optimal
            params[plot_parameters[0]] = x1
            params[plot_parameters[1]] = x2
            return fun(params)

        # Vectorization so we can pass the discretized domains and compute on all values
        vloss = np.vectorize(loss)
        # make a meshgrid out of the two changing parameters so we can compute loss on the entire grid
        X1, X2 = np.meshgrid(domains[:, 0], domains[:, 1])

        # make a plot of this loss versus the parameters
        plt.figure()
        ax = plt.axes(projection='3d')
        Z = vloss(X1, X2)
        ax.plot_surface(X1, X2, Z)
    else:
        # Normal plot if investigating 1 parameter
        def loss(x1):
            # return loss given x1
            params = x_optimal
            params[plot_parameters[0]] = x1
            return fun(params)

        # vectorization if so we can pass the domain
        vloss = np.vectorize(loss)
        plt.plot(domains[:, 0], vloss(domains[:, 0]))


class BruteForce(Optimizer):
    def __init__(self, Ns: int = 10, plot_result: bool = False, plot_mask: np.ndarray = None):
        self.Ns = Ns
        self.plot_result = plot_result
        self.plot_mask = plot_mask

    def __call__(self, fun: callable, x0: np.ndarray, args=(),
                 bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints=None,
                 **options) -> OptimizeResult:
        """
        Wrapping around the optimizer implemented as method to this class. Done s.t. this optimizer is compatible
        with the scipy.optimize interface.
        """
        return self.brute_force(fun, x0, arg=args, bounds=bounds, constraints=constraints, **options)

    def brute_force(self, fun: callable, x0: np.ndarray, args=(),
                    bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints=None,
                    **options) -> OptimizeResult:
        """
        :param fun: The objective function that we wish to minimize
        :param x0: starting values for the parameters we wish to optimize
        :param args: I have no idea why this is here, defaults to ()
        :param Ns: Number of samples computed along the range for variables, defaults to 10
        :param bounds: This needs to be provided otherwise bruteforce cant be used, defaults to None
        :param constraints: , defaults to None
        :return: OptimizeResult object from scipy
        """

        self._check_bounded(bounds)

        # Discretizing the domain to compute objective function values
        grid = self._make_grid(bounds)

        # Removing parameter combinations that are not conform constraints
        if constraints != ():
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

    def brute_wrapper(self, fun: callable, x0: np.ndarray, args=(),
                      bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints=None,
                      **options) -> OptimizeResult:
        """A wrapper around the bruteforce optimizer from scipy.optimize such that it is compatible with
        scipy.optimize.minimize

        :param fun: The objective function to be minimized
        :param x0: initial guess for parameters (not used in bruteforce)
        :param args: additional objective function parameters incase you OF has them, defaults to ()
        :param Ns: Sample points per parameter for the parameter grid, defaults to 10
        :param bounds: The domain in which we sample the parameters (optional arg in higher level function), defaults to None
        :param constraints: Constraint objects defining the constraints between parameters, defaults to None
        :return: A scipy.optimize.OptimizeResult object containing the optimal parameter values
        :raises NotImplementedError: If constraints are provided this function doesnt know what to do.
        """
        if constraints is not None:
            raise NotImplementedError("Brute wrapper cant deal with constraints yet.")
        if self.plot_result:
            raise NotImplementedError("brute_wrapper cant deal with plotting yet")

        self._check_bounded(bounds)
        ranges = tuple(bounds)
        result = brute(fun, ranges, args=args, Ns=self.Ns)
        return OptimizeResult(x=result, succes=True)

    def _make_grid(self, bounds: List[Tuple[Optional[float], Optional[float]]]) -> np.ndarray:
        """Constructs a grid such that grid[i,:] contains the ith combination of parameters

        :param bounds: The range over which we wish to optimize
        :param Ns: The number of samples along the range
        :return: The grid with shape Ns x Nx that we can use to compute the objective function values
        """
        N = len(bounds)
        for k in range(N):
            if type(bounds[k]) is not type(slice(None)):
                if len(bounds[k]) < 3:
                    bounds[k] = tuple(bounds[k]) + (complex(self.Ns),)
                bounds[k] = slice(*bounds[k])
        if (N == 1):
            bounds = bounds[0]

        grid = np.mgrid[bounds]

        # obtain an array of parameters that is iterable by a map-like callable
        inpt_shape = grid.shape
        if (N > 1):
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
        # Readability variables
        A = constraints.A
        lb = constraints.lb
        ub = constraints.ub

        index_to_drop = []
        for i in range(grid.shape[0]):
            # Apply matrix dot product to all parameter combinations
            transformed_parameters = np.dot(A, grid[i, :])
            # check if result of above operations breaks inequalities
            if not (np.all(lb <= transformed_parameters) and np.all(transformed_parameters <= ub)):
                index_to_drop.append(i)

        # if not satisfied drop parametercombination
        return np.delete(grid, np.array(index_to_drop), axis=0)

    @staticmethod
    def _check_bounded(allbounds: List[Tuple[Optional[float], Optional[float]]]) -> None:
        """This function checks the boundedness of a set of given bounds such that brute force optimizers
        can assume boundedness after calling this function.

        :param allbounds: A list of bounds
        :raises ValueError: Raises a value error in case the there are no bounds or if bounds are to large
        """
        # Check for finite boundaries
        if allbounds is None:
            raise ValueError(
                " No bounds provided in optimize: Brute force optimization can only be executed on a finite domain")

        for bounds in allbounds:
            for bound in bounds:
                if bound is None:
                    raise ValueError(" Infinite boundaries not supported for brute force optimizer ")
                if np.any(np.abs(bound) > 1e3):
                    raise ValueError("Boundary range is too large for meaningful result")
