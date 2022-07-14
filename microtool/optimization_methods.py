import sys
import warnings
from typing import List, Tuple, Optional

import numpy as np
from scipy.optimize import OptimizeResult
from tqdm.contrib.itertools import product


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
        if nx > 8:
            raise ValueError("The dimensionality of the problem is too large for brute force optimization. Only up to "
                             "8 dimensions currently supported.")
        # make the individual discretized domains
        domains = []
        previous_bound = ()
        for bound in bounds:
            if bound == previous_bound:
                # making a pointer to the previous array if the same bounds are provided sequentially
                domains.append(domains[-1].view())
            else:
                domains.append(np.linspace(bound[0], bound[1], num=self.Ns))

        x_optimal, y_optimal = self._find_minimum(fun, domains, constraints)
        return OptimizeResult(fun=y_optimal, x=x_optimal, succes=True)

    @staticmethod
    def _find_minimum(fun: callable, domains: List[np.ndarray], constraints) -> Tuple[np.ndarray, float]:
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


class SOMA(Optimizer):
    """ A class for defining a classic all to one soma algorithm. The __call__ method executes the algorithm The
    other class methods are made to initialize all parts of the algorithm with control parameters defined on
    initialization. Visit https://ivanzelinka.eu/somaalgorithm/About.html for the more information on this algorithm
    and codes for which this implementation is the source.
    """

    def __init__(self,
                 population_sz: int = 50,
                 max_migrations: int = 200,
                 max_fevals=None,
                 PRT: float = 0.1,
                 path_length: float = 3.0,
                 step: float = 0.11
                 ):
        """Constructor initializes the control parameters and saves them as attributes

        :param population_sz: Number of individuals in the population, defaults to 50
        :param max_migrations: Stopping condition based on maximum number of individuals, defaults to 100
        :param max_fevals: Stopping condition based
                            on maximum number of objective function evaluations, defaults to None
        :param PRT: probability of stepping, defaults to 0.1
        :param path_length: length of jumps, defaults to 3.0
        :param step: steplength along the jump,
        defaults to 0.11
        :raise ValueError: if controlparameters are of incorrect value
        """
        # Check that parameters are reasonable
        if path_length < step:
            raise ValueError("SOMA: The following condition is violated step > path_length.")
        if PRT <= 0 or PRT > 1:
            raise ValueError("SOMA: Acceptance probability PRT needs to be in (0,1].")

        # loading the control parameters
        self.population_sz, self.max_migrations = population_sz, max_migrations
        self.max_fevals = max_fevals

        # defining parameters for the SOMA migration behaviour
        self.PRT, self.path_length = PRT, path_length
        self.step = step
        self.N_jump = int(np.ceil(path_length / step))

    def __str__(self):
        message = "SOMA optimizer with the following control parameters:\n"
        for key, val in vars(self).items():
            message += f"{key} = {val}\n"
        return message

    def __call__(self, fun: callable, x0: np.ndarray, args=(), **options) -> OptimizeResult:
        """This is where the optimization routine is ran.

        :param fun: Objective function to be minimized
        :param x0: starting parameters (only used for length)
        :param args: additional parameters to the objective function, defaults to ()
        :return: scipy.OptimizeResult object, (really just a dictionary with a wrapper)
        """
        self.Nx = len(x0)
        if self.max_fevals is None:
            self.max_fevals = self.Nx * 10 ** 4

        # TODO: guarantee bounds in higherlevel functions in tissuemodel
        bounds = options["bounds"]
        constraints = options['constraints']

        population = Population(self.population_sz, bounds, constraints, fun, self.max_fevals)

        migration = 0
        while (population.fevals < self.max_fevals) and (migration < self.max_migrations):
            migration += 1
            population.migrate(self.path_length, self.N_jump, self.PRT)

        return OptimizeResult(x=population.best_individual, fun=population.best_cost, nfev=population.fevals,
                              success=True, message=f"stopped at migration: {migration}")


class Population:
    """This class defines the population in the SOMA algorithm
    """

    def __init__(self, sz: int, bounds: np.ndarray, constraint, fun: callable, max_evals: int,
                 values: np.ndarray = None):
        """
        :param sz: The population size
        :param bounds: The parameter bounds (or the search domain if you prefer this terminology)
        :param fun: Objective function to be optimized
        :param max_evals: Maximum allowed function evaluation before stopping the SOMA algorithm
        :param values: Starting parameter values of the population, defaults to None
        """
        self.sz = sz
        bounds = np.array(bounds)
        self.constraint = constraint
        self.lower_bound = bounds[:, 0]
        self.upper_bound = bounds[:, 1]
        self.Nx = len(bounds)
        self.fun = fun
        self.max_fevals = max_evals

        self.fevals = 0
        self.best_cost = np.inf
        self.best_individual = None

        if values == None:
            # initializing a randomly distributed population over the provided domains
            self._values = bounds[:, 0] + np.random.rand(sz, len(bounds)) * (bounds[:, 1] - bounds[:, 0])

        # computing the starting fitness values
        self._set_fitness()

    def cost_fun(self, x: np.ndarray):
        if is_constrained(x, self.constraint.A, self.constraint.lb, self.constraint.ub):
            return np.inf
        else:
            return self.fun(x)

    def migrate(self, path_length: float, N_jump: int, PRT: float):
        """Executes a single population migration

        :param path_length: The path an individual will follow
        :param N_jump: Number of jumps along the path
        :param PRT: Probability of accepting a step vector
        """
        # Get the leader of the pack
        leader_id = np.argmin(self._fitness)
        leader = self._values[leader_id]

        # The other individuals (losers)
        rest_id = [i for i in range(self.sz) if i != leader_id]

        # the steps the individuals might take
        steps = np.linspace(0, path_length, num=N_jump)
        for j in rest_id:
            moving = self._values[j, :]
            journey = np.zeros((N_jump, self.Nx))

            for k, step in enumerate(steps):
                # Determine along which variables (if any) the individual will take a step
                PRTvector = np.random.rand(self.Nx) < PRT
                # Saving the step towards the leading individual
                journey[k] = moving + (leader - moving) * step * PRTvector

            # Putting lost individuals back in the search domain
            journey = self._putback(journey)

            # This is the amount of evaluations needed to update the cost along the route
            self.fevals += N_jump

            # stop condition (before updating the costs.... why I dont know)
            if self.fevals > self.max_fevals:
                self.fevals = self.max_fevals
                break

            # get the best place along the route from the moving individual
            new_cost = np.array([self.cost_fun(journey[i, :]) for i in range(N_jump)])
            bestmove_id = np.argmin(new_cost)
            bestmove_cost = new_cost[bestmove_id]
            bestmove_parameters = journey[bestmove_id, :]

            # Move the individual j along the route if it improved its cost by moving
            if bestmove_cost < self._fitness[j]:
                self._values[j, :] = bestmove_parameters
                self._fitness[j] = bestmove_cost

            # update globalbest, this is the output of the algorithm after the stopping condition is reached
            if bestmove_cost < self.best_cost:
                self.best_cost = bestmove_cost
                self.best_individual = bestmove_parameters

    def _putback(self, journey: np.ndarray) -> np.ndarray:
        """This function puts the travelling individuals that move out of the search space back in.
        It does so by assigning a random value lying inside the searchspace along the parameters that violate the bounds.

        :param journey: A numpy array containing the travelling individuals parameters, shape (N_jumps, N_x)
        :return: corrected journey array s.t. all combinations of N_x lie in the search domain
        """
        for i in range(len(journey)):
            violated = (journey[i, :] > self.upper_bound) | (journey[i, :] < self.lower_bound)
            if np.any(violated):
                journey[i, :][violated] = self.lower_bound[violated] + np.random.rand(np.sum(violated)) * (
                        self.upper_bound[violated] - self.lower_bound[violated])
        return journey

    def _set_fitness(self):
        """Computes fitness values for the current population and updates the function evaluations associated with this.
        """
        loss = np.zeros(self.sz)
        for i in range(self.sz):
            parameter_combination = self._values[i, :]
            loss[i] = self.cost_fun(parameter_combination)
        self._fitness = loss
        self.fevals += self.sz


def is_constrained(x: np.ndarray, A: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
    """A function for checking if a given parameter combination breaks a given linear constraint.

    :param A:
    :param lb:
    :param ub:
    :param x: Parameter combination
    :return: boolean that is true if the parameter combination breaks the constraint
    """
    transformed_parameters = A @ x
    return np.any((lb >= transformed_parameters) | (transformed_parameters >= ub))


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
