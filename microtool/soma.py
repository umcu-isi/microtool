
from dataclasses import dataclass
from matplotlib import path
import numpy as np
from scipy.optimize import OptimizeResult


class SOMA:
    """ A class for defining a classic all to one soma algorithm. The run method executes the algorithm
    The other class methods are made to initialize all parts of the algorithm with control parameters defined on initialization.
    """    
    def __init__(self,
        population_sz:int=50,
        max_migrations:int=100,
        max_fevals = None,
        PRT : float = 0.1,
        path_length:float = 3.0,
        step:float = 0.11
        ):

        #TODO: raise value error if unreasonable controlparameters are provided
        # loading the control parameters
        self.population_sz, self.max_migrations = population_sz, max_migrations
        self.max_fevals = max_fevals

        # defining parameters for the SOMA migration behaviour
        self.PRT, self.path_length = PRT, path_length
        self.step = step
        self.N_jump = int(np.ceil(path_length/step))
        
    def __str__(self):
        message = "SOMA optimizer with the following control parameters:\n"
        for key,val in vars(self).items():
            message += f"{key} = {val}\n"
        return message

    # optimization routine
    def run(self, fun: callable, x0:np.ndarray, args =(), **options) -> OptimizeResult:
        
        self.Nx = len(x0)
        if self.max_fevals == None:
            self.max_fevals = self.Nx*10**4

        bounds = options["bounds"]
        population = Population(self.population_sz, bounds, fun, self.max_fevals)
    
        migration = 0
        while (population.fevals < self.max_fevals) and (migration < self.max_migrations):
            migration += 1
            population.migrate(self.path_length,self.N_jump,self.PRT)
        print("stopped at migration: ",migration)
        return OptimizeResult(x=population.best_individual, fun = population.best_cost, nfev = population.fevals,success=True)

    

class Population:
    def __init__(self, sz:int, bounds:np.ndarray, fun:callable, max_evals:int, values:np.ndarray = None):
        
        self.sz = sz
        self.lower_bound = bounds[:,0]
        self.upper_bound = bounds[:,1]
        self.Nx = len(bounds)
        self.fun = fun
        self.max_fevals = max_evals

        self.fevals = 0
        self.best_cost = np.inf
        self.best_individual = None

        if values == None:
        # initializing a randomly distributed population over the provided domains
            self._values = bounds[:,0] + np.random.rand(sz,len(bounds)) * (bounds[:,1] - bounds[:,0])
        
        self.set_fitness()
        
    def migrate(self,path_length:float,N_jump:int, PRT:float):
        # for every individual walk along pathlength and save the journey
        leader_id = np.argmin(self._fitness)
        leader = self._values[leader_id]

        rest_id = [i for i in range(self.sz) if i!=leader_id ]

        steps = np.linspace(0,path_length, num=N_jump)
        for j in rest_id:
            moving = self._values[j,:]
            journey = np.zeros((N_jump, self.Nx))

            for k, step in enumerate(steps):
                PRTvector = np.random.rand(self.Nx) < PRT
                journey[k] = moving + (leader - moving) * step * PRTvector
            
            journey = self._putback(journey)

            # This is the amount of evaluations needed to update the cost along the route
            self.fevals += N_jump

            # stop condition
            if(self.fevals > self.max_fevals):
                self.fevals = self.max_fevals
                break
            
            # get the best place along the route from the moving individual
            new_cost = np.array([self.fun(journey[i,:]) for i in range(N_jump)])
            bestmove_id = np.argmin(new_cost)
            bestmove_cost = new_cost[bestmove_id]
            bestmove_parameters = journey[bestmove_id,:]
            
            # Move the individual j along the route if it improved its cost by moving
            if bestmove_cost < self._fitness[j]:
                self._values[j,:] = bestmove_parameters
                self._fitness[j] = bestmove_cost

            # update globalbest
            if bestmove_cost<self.best_cost:
                self.best_cost = bestmove_cost
                self.best_individual = bestmove_parameters

    def _putback(self, journey:np.ndarray) -> np.ndarray:
        
        for i in range(len(journey)):
            if np.any(journey[i,:] < self.upper_bound) or np.any(journey[i,:] > self.lower_bound):
                journey[i,:] = self.lower_bound + np.random.rand(self.Nx) * (self.upper_bound- self.lower_bound)
        return journey


    def set_fitness(self):

        loss = np.zeros(self.sz)
        for i in range(self.sz):
            loss[i] = self.fun(self._values[i,:])
        self._fitness = loss
        self.fevals += self.sz
    
    def get_fitness(self):
        return self._fitness

    
