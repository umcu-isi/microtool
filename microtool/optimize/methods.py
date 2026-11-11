from typing import List, Tuple

import numpy as np
from scipy.optimize import OptimizeResult, Bounds, NonlinearConstraint, minimize
from tqdm.contrib.itertools import product
from scipy.stats import qmc
import logging


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
        PRT: float = 0.7,
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
            initial_value=np.asarray(x0, dtype=float).ravel(),
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
    """This class defines the population in the SOMA algorithm."""

    def __init__(self, sz: int, bounds: Bounds, constraint, fun: callable, args: tuple, max_evals: int,
                 values: np.ndarray = None, initial_value: np.ndarray = None):
        """
        :param sz: The population size
        :param bounds: The parameter bounds (or the search domain if you prefer this terminology)
        :param constraint: Constraints of the optimization problem
        :param fun: Objective function to be optimized
        :param max_evals: Maximum allowed function evaluation before stopping the SOMA algorithm
        :param values: Starting parameter values of the population, defaults to None
        :param initial_value: A known-feasible point, used as the start of the random walk
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

        # computing the starting fitness values
        self._set_fitness()

        best_id = int(np.argmin(self._fitness))
        if np.isfinite(self._fitness[best_id]):
            self.best_cost = float(self._fitness[best_id])
            self.best_individual = self._values[best_id].copy()

    # ------------------------------------------------------------------
    # feasibility
    # ------------------------------------------------------------------
    def _is_feasible(self, x):
        """Errors from constraint functions (e.g. 'Pulse interval too short') count as infeasible."""
        try:
            return bool(np.all(np.isfinite(x))
                        and np.all(x >= self.lower_bound)
                        and np.all(x <= self.upper_bound)
                        and not is_constrained(x, self.constraints))
        except Exception:
            return False

    def _ray_extent(self, x, d, n=20, bisect=12):
        """Largest s >= 0 such that x + s*d stays feasible (march along the ray, then bisect the first break)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            s_lim = np.where(d > 0, (self.upper_bound - x) / d,
                    np.where(d < 0, (self.lower_bound - x) / d, np.inf))
        s_box = np.min(s_lim)
        if not np.isfinite(s_box) or s_box <= 0:
            return 0.0

        ok, bad = 0.0, None
        for s in np.linspace(0, s_box, n + 1)[1:]:
            if self._is_feasible(x + s * d):
                ok = s
            else:
                bad = s
                break
        if bad is None:
            return ok

        for _ in range(bisect):
            mid = 0.5 * (ok + bad)
            if self._is_feasible(x + mid * d):
                ok = mid
            else:
                bad = mid
        return ok

    # ------------------------------------------------------------------
    # initialization: hit-and-run from a known-feasible point
    # ------------------------------------------------------------------
    def _initialize_population(self, sz, initial_value=None, burn_in=2000, thin=100,
                               n_active=None, include_start=True):
        """
        Random walk inside the feasible region, starting at initial_value.

        :param burn_in: steps taken before points are collected (moves away from x0)
        :param thin: collect every `thin`-th step (reduces correlation between points)
        :param n_active: number of coordinates moved per step. None = all coordinates.
                         Use 1-3 if segments are very short (thin feasible region).
        :param include_start: also put the starting point itself in the population
        """
        if initial_value is None:
            raise ValueError("Hit-and-run initialization needs a feasible initial_value (x0).")

        rng = np.random.default_rng()
        span = self.upper_bound - self.lower_bound

        x = np.asarray(initial_value, float).ravel()
        if x.size != self.Nx:
            raise ValueError("Initial value length does not match the bounds.")
        if not self._is_feasible(x):
            raise ValueError("initial_value is not feasible; cannot start the random walk.")

        candidates = [x.copy()] if include_start else []
        steps, stuck, moved = 0, 0, 0
        max_steps = burn_in + thin * sz * 10
        print(f"[Init] hit-and-run from x0 | Nx={self.Nx}, target={sz}, "
              f"burn_in={burn_in}, thin={thin}, n_active={n_active}")

        while len(candidates) < sz and steps < max_steps:
            steps += 1

            # random direction (optionally sparse), scaled per parameter range
            d = rng.normal(size=self.Nx)
            if n_active is not None and n_active < self.Nx:
                mask = np.zeros(self.Nx, dtype=bool)
                mask[rng.choice(self.Nx, size=n_active, replace=False)] = True
                d[~mask] = 0.0
            d = d * span
            norm = np.linalg.norm(d / span)
            if norm == 0:
                continue
            d /= norm

            s_pos = self._ray_extent(x, d)
            s_neg = self._ray_extent(x, -d)
            seg = s_pos + s_neg
            if seg < 1e-9:
                stuck += 1                       # blocked in both directions
                continue

            # uniform point on the feasible segment (interior, not the boundary)
            s = rng.uniform(-s_neg, s_pos)
            x_new = x + s * d
            if self._is_feasible(x_new):
                x = x_new
                moved += 1

            if steps > burn_in and steps % thin == 0:
                candidates.append(x.copy())
                print(f"[Init] {len(candidates):3d}/{sz} | step={steps} "
                      f"| segment (rel)={np.linalg.norm(seg * d / span):.3e} "
                      f"| dist from x0 (rel)={np.linalg.norm((x - np.asarray(initial_value, float).ravel()) / span):.3e}")

        pop = np.asarray(candidates)
        print("=" * 60)
        print(f"[Init] Done: {len(pop)}/{sz} points, {steps} steps, moved={moved}, stuck={stuck}, ")
        print(f"[Init] Per-coordinate spread (std/range): {np.round(pop.std(axis=0) / span, 3)}")
        print("=" * 60)

        if len(pop) < sz:
            logging.warning(f"Only {len(pop)} of {sz} feasible points found.")
        return pop

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

def build_idx(scheme, x0):
    """Map parameter names to their indices in the scaled free-parameter vector."""
    names = {
        "delta": "DiffusionPulseDuration",
        "Delta": "DiffusionPulseInterval",
        "G":     "DiffusionPulseMagnitude",
        "TE":    "EchoTime",
    }
    x0 = np.asarray(x0, float)
    base = {k: np.asarray(scheme._copy_and_update_scaled_parameter(n, x0), float)
            for k, n in names.items()}
    idx = {k: [] for k in names}
    for j in range(len(x0)):
        xp = x0.copy()
        xp[j] += 1.0
        for k, n in names.items():
            new = np.asarray(scheme._copy_and_update_scaled_parameter(n, xp), float)
            if not np.allclose(new, base[k]):
                idx[k].append(j)
    return {k: np.array(v, dtype=int) for k, v in idx.items()}