
from typing import Sequence, Callable
from dataclasses import dataclass,field,astuple
import numpy as np
from scipy.optimize.optimize import OptimizeResult


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

def SOMA(fun : callable, x0 : np.ndarray, args = (),**options) -> OptimizeResult:
    """The SOMA optimization as described in https://github.com/diepquocbao/SOMA-T3A-Python (version 2)

    :param fun: The loss function we wish to minimize
    :param x0: initial parameter choices
    :param args: Additional arguments to the loss function, defaults to ()
    :return: scipy.optimizeresult with optimization outcome
    """    
    # %%%%%%%%%%%%%%% Prelimenaries %%%%%%%%%%%%%%%%%
    # Defining function for evaluating a population cost
    def population_cost(pop:np.ndarray) -> np.ndarray:

        _, pop_size = np.shape(pop)
        cost = np.zeros(pop_size)
        for i in range(pop_size):
            cost[i] = fun(pop[:,i])
        return cost

    # Setting the control parameters to default if not supplied
    if "control_parameters" not in options.keys():
        cparams = ControlParametersSOMA(len(x0))
    else:
        cparams = options["control_parameters"]

    Nx,N_jump,pop_size,max_migrations,m,n,k,max_FEs = astuple(cparams)


    # Initial population
    bounds = np.array(options["bounds"])
    # making lower bound and upperbound arrays
    lb = np.repeat(bounds[:,0].reshape(Nx,1),pop_size,axis=1)
    ub = np.repeat(bounds[:,1].reshape(Nx,1),pop_size,axis=1)
    pop = lb + np.random.rand(Nx, pop_size) * (ub - lb)
    fitness = population_cost(pop)
    FEs = pop_size
    best_cost = min(fitness)
    id = np.argmin(fitness)
    best_x = pop[:,id]

    # ------------ Migrations
    migration = 0
    while FEs < max_FEs:
        migration +=1
        # select migrant m
        M = np.random.choice(range(pop_size),m,replace=False)
        M_sort = np.argsort(fitness[M])
        newpop = np.zeros((Nx, n * N_jump))
        for j in range(n):
            Migrant = pop[:, M[M_sort[j]]].reshape(Nx, 1)
            # select leader
            K = np.random.choice(range(pop_size),k,replace=False)
            K_sort = np.argsort(fitness[K])
            Leader = pop[:, K[K_sort[0]]].reshape(Nx, 1)
        if M[M_sort[j]] == K[K_sort[0]]:
            Leader = pop[:, K[K_sort[1]]].reshape(Nx, 1)
        PRT = 0.05 + 0.9*(FEs/max_FEs)
        step = 0.15 - 0.08*(FEs/max_FEs)    
        nstep = np.arange(0,N_jump)*step + step
        PRTvector = (np.random.rand(Nx,N_jump) < PRT) * 1
        indi_new = Migrant + (Leader - Migrant) * nstep * PRTvector
        # Putback into search range
        for cl in range(N_jump):
            for rw in range(Nx):
                if indi_new[rw,cl] < bounds[rw,0] or indi_new[rw,cl] > bounds[rw,1]:
                    indi_new[rw,cl] = bounds[rw,0] + np.random.rand() * (bounds[rw,1] - bounds[rw,0])
        newpop[:,N_jump * j:N_jump * (j+1)] = indi_new
        
        newfitpop = population_cost(newpop)
        FEs += n*N_jump

        for j in range(n):
            newfit = newfitpop[N_jump*j:N_jump*(j+1)]
            min_newfit = min(newfit)
        # ----- Accepting: Place the best offspring into the current population
        if min_newfit <= fitness[M[M_sort[j]]]:
            fitness[M[M_sort[j]]] = min_newfit
            id = np.argmin(newfit)
            pop[:, M[M_sort[j]]] = newpop[:, (N_jump*j)+id]
            # ----- Update the global best value --------------------
            if min_newfit < best_cost:
                best_cost = min_newfit
                best_x = newpop[:, (N_jump*j)+id]

    return OptimizeResult(x = best_x,succes = True, fun=best_cost, nfev = FEs)


    



@dataclass
class ControlParametersSOMA:
    Nx : int
    N_jump : int = 45
    pop_size : int = 100
    max_migrations: int = 100
    m: int = 10
    n: int = 5
    k: int = 10
    max_FEs : int = field(init=False)

    def __post_init__(self):
        self.max_FEs = self.Nx*10**4
