from sympy.solvers import solve
from sympy import Symbol, minimum, Interval
import numpy as np

# 1/ ms mT
gamma = 42.57747892 * 2 * np.pi
# ms?
t90 = 4
t180 = 6
# s /micro m^2 ?
# b = 3
# ms?
t_half = 14

G_max = 200e-6  # mt/um
S_max = 1300e-6  # mt/um/ms
t_rise = G_max / S_max
#
# t180 = Symbol('t180', positive=True)
# t90 = Symbol('t90',positive=True)
# t_half = Symbol('t_half', positive=True)
# t_rise = Symbol('t_rise', positive=True)
delta = Symbol('delta', positive=True)
Delta = Symbol('Delta', positive=True)
for b in np.linspace(0.05, 3):
    # defining the b-equation
    b_eq = gamma ** 2 * G_max ** 2 * (
            delta ** 2 * (Delta - delta / 3) + (1 / 30) * t_rise ** 3 - (delta / 6) * t_rise ** 2) - b
    # Delta as function of delta
    Delta_sol = solve(b_eq, Delta)[0]
    # TE as function of delta
    TE = 0.5 * t90 + Delta_sol + delta + t_rise + t_half

    # First constrain on delta is found by solving the following equation
    delta_max_1_eq = 0.5 * t180 + delta + t_rise + t_half - 0.5 * TE
    delta_max_1_sol = solve(delta_max_1_eq, delta)

    # if no solutions are found we set the boundary to infinity
    if delta_max_1_sol != []:
        delta_max_1 = max(delta_max_1_sol)
    else:
        delta_max_1 = np.inf
    # print(solve(delta_max_1_eq, delta))

    delta_max_2_eq = 0.5 * t90 + 0.5 * t180 + delta + t_rise - 0.5 * TE
    delta_max_2 = max(solve(delta_max_2_eq, delta))

    # The maximum allowed value for delta
    delta_max = min([delta_max_1, delta_max_2])
    TE_min = minimum(TE, delta, Interval(0, delta_max))
    delta_sol = solve(TE - TE_min, delta)
    print(TE_min)