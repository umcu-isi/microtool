# MICROtool

*Framework for Diffusion MRI Experiment Optimization*

**MICROtool** stands for Microstructural Imaging experiment Cramer-Rao lower bound Optimisation, a  framework for the optimization of microstructural imaging in diffusion MRI. This is all performed by means of the Cramer-Rao Lower Bound and Fisher matrix to define a loss function that, combined with an optimal gradient direction selection, provides estimation on MRI scheme acquisitions for a more approachable practice.

Its main functionalities are differentiated into a series of modules, these being: tissue models, acquisition schemes, loss functions and Monte Carlo Simulation.

**Tissue Model**

Allows the utilization of external packages, such as Dmipy or Misst, for the characterization of multi-compartment modeling (MCM), and addition of transversal relaxation effects for the optimization of acquisition parameters based on the requirements thereby stablished. 

**Acquisition Scheme**

A series of characteristic schemes consisting of basic diffusion and inversion-recovery MR acquisition schemes. Computations are derived from predefined pulse relations and scanner parameters, that establish the thresholds for optimization. Moreover, a series of additional scheme definitions for reproducibility of the experiments performed by Alexander [1] and its corresponding tissue models.

**Optimization**

Consisting of a minimization based on the Cramer Rao Lower Bound. Optimization is currently implemented to utilize differential evolution or SciPy pipelines, as well a set of constraints conformed out of scheme acquisition parameter and pulse relations.  

**Fitting and Monte Carlo Simulations**

Tissue model fitting to signal is computed through non-linear least-squares optimization and tailored to the specifics of each MCM. Subsequently, a module for Monte Carlo simulations is available for estimation of parameter fitting. 


[1] Alexander, D. C. (2008). A general framework for experiment design in diffusion MRI and its application in measuring direct tissue‐microstructure features. Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine, 60(2), 439-448.

## Installation

MICROtool requires Python 3.8.
To install MICROtool in development-mode, run the following commands from this directory:

```shell
pip install -r requirements.txt
pip install -e .
```

## Testing

To run all tests, simply call pytest from this directory:

```shell
pytest
```