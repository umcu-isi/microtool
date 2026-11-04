from typing import List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, Bounds, NonlinearConstraint
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

        if isinstance(constraints, list):
            raise NotImplementedError("Bruteforce optimizer cant handle multiple constraints")

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

        x_optimal, y_optimal = self._find_minimum(fun, args, domains, constraints)
        return OptimizeResult(fun=y_optimal, x=x_optimal, succes=True)

    @staticmethod
    def _find_minimum(fun: callable, args: tuple, domains: List[np.ndarray], constraints) -> Tuple[np.ndarray, float]:
        if constraints is None or constraints == ():
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
        else:
            # iterate over the grid
            y_optimal = np.inf
            x_optimal = None
            for combination in product(*domains, desc='Running brute force grid computation'):
                combination = np.array(combination)
                # check constraint
                if is_constrained(combination, constraints):
                    loss = np.inf
                else:
                    loss = fun(combination, *args)

                # update optimal value
                if loss < y_optimal:
                    x_optimal = combination
                    y_optimal = loss

        return x_optimal, y_optimal


class SOMA(Optimizer):
    """A class for defining a classic all to one soma algorithm. The __call__ method executes the algorithm The
    other class methods are made to initialize all parts of the algorithm with control parameters defined on
    initialization. Visit https://ivanzelinka.eu/somaalgorithm/About.html for the more information on this algorithm
    and codes for which this implementation is the source.
    """

    def __init__(
        self,
        population_sz: int = 50,
        max_migrations: int = 200,
        max_fevals=None,
        PRT: float = 0.1,
        path_length: float = 3.0,
        step: float = 0.11,
    ):
        if path_length < step:
            raise ValueError(
                "SOMA: The following condition is violated step > path_length."
            )
        if PRT <= 0 or PRT > 1:
            raise ValueError(
                "SOMA: Acceptance probability PRT needs to be in (0,1]."
            )

        self.population_sz = population_sz
        self.max_migrations = max_migrations
        self.max_fevals = max_fevals
        self.PRT = PRT
        self.path_length = path_length
        self.step = step
        self.N_jump = int(np.ceil(path_length / step))

    def __str__(self):
        message = "SOMA optimizer with the following control parameters:\n"
        for key, val in vars(self).items():
            message += f"{key} = {val}\n"
        return message

    def __call__(
        self, fun: callable, x0: np.ndarray, args=(), **options
    ) -> OptimizeResult:
        self.Nx = len(x0)
        if self.max_fevals is None:
            self.max_fevals = self.Nx * 10**4

        bounds = options["bounds"]
        constraints = options.get("constraints", ())
        if constraints is None:
            constraints = ()
        elif not isinstance(constraints, (list, tuple)):
            constraints = (constraints,)

        print("=" * 70)
        print("[SOMA] Starting Optimization")
        print(f"  Dimension (Nx)     : {self.Nx}")
        print(f"  Population Size    : {self.population_sz}")
        print(f"  Max Migrations     : {self.max_migrations}")
        print(f"  Max Fevals         : {self.max_fevals}")
        print(f"  Jumps per movement : {self.N_jump}")
        print("=" * 70)

        population = Population(
            self.population_sz,
            bounds,
            constraints,
            fun,
            args,
            self.max_fevals,
            #initial_value=np.asarray(x0, dtype=float).ravel(),
        )

        prev_best = population.best_cost
        print(
            f"[SOMA Init] Initial Best Cost: {prev_best:.6e} | Initial Fevals: {population.fevals}"
        )
        print("-" * 70)

        migration = 0
        while (population.fevals < self.max_fevals) and (
            migration < self.max_migrations
        ):


            population.migrate(self.path_length, self.N_jump, self.PRT)

            current_best = population.best_cost
            improvement = prev_best - current_best

            print(
                f"[SOMA] Mig {migration:03d}/{self.max_migrations:03d} | "
                f"Fevals: {population.fevals:06d}/{self.max_fevals:06d} | "
                f"Best Cost: {current_best:.6e} | "
                f"Delta: {improvement:+.3e} | "
            )

            prev_best = current_best
            migration += 1

        print("=" * 70)
        print(f"  Total Migrations : {migration}")
        print(f"  Total Fevals      : {population.fevals}")
        print(f"  Final Best Cost   : {population.best_cost:.6e}")
        print("=" * 70)

        if population.best_individual is None:
            raise RuntimeError(
                "SOMA did not find a feasible candidate with a finite cost. "
                "Check the bounds, constraints, and objective function."
            )

        return OptimizeResult(
            x=population.best_individual,
            fun=population.best_cost,
            nfev=population.fevals,
            nit=migration,
            success=True,
            message="Optimization completed.",
        )

class Population:
    """This class defines the population in the SOMA algorithm
    """

    def __init__(self, sz: int, bounds: Bounds, constraint, fun: callable, args: tuple, max_evals: int,
                 values: np.ndarray = None, initial_value: np.ndarray = None):
        """
        :param sz: The population size
        :param bounds: The parameter bounds (or the search domain if you prefer this terminology)
        :param fun: Objective function to be optimized
        :param max_evals: Maximum allowed function evaluation before stopping the SOMA algorithm
        :param values: Starting parameter values of the population, defaults to None
        """
        self.sz = sz

        self.constraints = constraint
        self.lower_bound = np.asarray(bounds.lb, dtype=float).ravel()
        self.upper_bound = np.asarray(bounds.ub, dtype=float).ravel()
        self.Nx = len(bounds.lb)
        self.fun = fun
        self.args = args
        self.max_fevals = max_evals

        self.fevals = 0
        self.best_cost = np.inf
        self.best_individual = None

        if np.any(~np.isfinite(self.lower_bound)) or np.any(~np.isfinite(self.upper_bound)):
            raise ValueError("SOMA requires finite lower and upper bounds.")
        if np.any(self.lower_bound > self.upper_bound):
            raise ValueError("SOMA lower bounds must not exceed upper bounds.")

        if values is None:
            self._values = self._initialize_population(sz, initial_value)
        else:
            self._values = np.asarray(values, dtype=float)
            if self._values.shape != (sz, self.Nx):
                raise ValueError(
                    f"Population values must have shape {(sz, self.Nx)}."
                )

        # computing the starting fitness values
        self._set_fitness()

        best_id = int(np.argmin(self._fitness))
        if np.isfinite(self._fitness[best_id]):
            self.best_cost = float(self._fitness[best_id])
            self.best_individual = self._values[best_id].copy()

    def _initialize_population(self, sz: int, initial_value: np.ndarray = None) -> np.ndarray:
        candidates = []
        if initial_value is not None:
            initial_value = np.asarray(initial_value, dtype=float).ravel()
            if initial_value.size != self.Nx:
                raise ValueError("Initial value length does not match the bounds.")
            if np.all(np.isfinite(initial_value)) and not np.any(
                (initial_value < self.lower_bound)
                | (initial_value > self.upper_bound)
            ) and not is_constrained(initial_value, self.constraints):
                candidates.append(initial_value.copy())

        max_attempts = max(1000, sz * 100)
        attempts = 0
        while len(candidates) < sz and attempts < max_attempts:
            candidate = self.lower_bound + np.random.rand(self.Nx) * (
                self.upper_bound - self.lower_bound
            )
            attempts += 1
            if not is_constrained(candidate, self.constraints):
                candidates.append(candidate)

        if not candidates:
            raise ValueError(
                "SOMA could not find a feasible starting point. Check that "
                "the constraints and bounds have a feasible intersection."
            )

        while len(candidates) < sz:
            candidates.append(candidates[np.random.randint(len(candidates))].copy())

        return np.asarray(candidates)

    def cost_fun(self, x: np.ndarray):
        if (
            np.any(~np.isfinite(x))
            or np.any(x < self.lower_bound)
            or np.any(x > self.upper_bound)
            or is_constrained(x, self.constraints)
        ):
            return np.inf

        cost = float(self.fun(x, *self.args))
        return cost if np.isfinite(cost) else np.inf

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
        It does so by assigning a random value lying inside the searchspace along the parameters
        that violate the bounds.

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


def is_constrained(x: np.ndarray, constraints: List[NonlinearConstraint]) -> bool:
    """A function for checking if a given parameter combination breaks a given constraint.

    :param constraints: A scipy compatible constraint dictionary
    :param x: Parameter combination
    :return: boolean that is true if the parameter combination breaks the constraint
    """
    if constraints is None:
        return False

    if not isinstance(constraints, (list, tuple)):
        constraints = (constraints,)

    for constraint in constraints:
        constraint_value = np.asarray(constraint.fun(x), dtype=float)
        if not np.all(np.isfinite(constraint_value)):
            return True
        if np.any(constraint_value < constraint.lb) or np.any(
            constraint_value > constraint.ub
        ):
            return True

    return False
