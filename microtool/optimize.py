from typing import Sequence, Callable, Tuple, List, Optional

import numpy as np
from scipy.optimize.optimize import OptimizeResult
from scipy.optimize import LinearConstraint, brute


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

def brute_wrapper(fun: callable, x0: np.ndarray, args=(), Ns: int = 10, bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints = None,  **options) -> OptimizeResult:
    check_bounded(bounds)
    ranges = tuple(bounds)
    result = brute(fun, ranges, args=args, Ns=Ns)
    return OptimizeResult(x=result, succes = True) 




def brute_force(fun: callable, x0: np.ndarray, args=(), Ns: int = 10, bounds: List[Tuple[Optional[float], Optional[float]]] = None, constraints = None,  **options) -> OptimizeResult:
    """Practicing with the bruteforce function, this is a custom minimizer used in the scipy.optimize interface

    :param fun: The objective function that we wish to minimize
    :param x0: starting values for the parameters we wish to optimize
    :param args: I have no idea why this is here, defaults to ()
    :param Ns: Number of samples computed along the range for variables, defaults to 10
    :param bounds: This needs to be provided otherwise bruteforce cant be used, defaults to None
    :param constraints: I dont know how brute force could deal with this, defaults to None
    :return: OptimizeResult object to for output
    :raise ValueError: bounds or constraints are not appropriate for brute force optimization
    """
    
    check_bounded(bounds)
    # Check for no constraints
    if constraints != None:
        raise ValueError("brute_force cant deal with constraints")
    
    # Number of parameters to optimize
    Nx = len(x0)

    # Creating a discretized domain to compute objective funtion values on
    bounds = np.array(bounds)
    domains = np.linspace(bounds[:,0],bounds[:,1],num=Ns,axis = -1)

    # Meshgrid only takes individual domain vectors so we use unpacking operator *
    mesh = np.array(np.meshgrid(*domains))

    # Reshaping so that grid[i,:] is an array of parameter values that we can 
    # evaluate using the objective function
    grid = np.dstack(mesh).reshape(-1,Nx)
    
    # Evaluating the objective function for all possible parameter combinations
    loss = np.zeros(Ns*Nx)
    for i in range(Ns*Nx):
        loss[i] = fun(grid[i,:])
    
    # Getting the optimal values by finding the minimum loss.
    min_index = np.argmin(loss)
    x_optimal = grid[min_index,:]
    y_optimal = loss[min_index]

    return OptimizeResult(fun = y_optimal,x=x_optimal, succes = True)

def check_bounded(allbounds : List[Tuple[float]]) -> None:
    
    # Check for finite boundries
    if allbounds == None:
        raise ValueError(" No bounds provided in optimize: Brute force optimization can only be executed on a finite domain")
    
    for bounds in allbounds:
        for bound in bounds:
            if bound == None:
                raise ValueError(" Infinite boundaries not supported for brute force optimizer ")
            if np.any(np.abs(bound) > 1e3):
                raise ValueError("Boundary range is too large for meaningful result")


   
    