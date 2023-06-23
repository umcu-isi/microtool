from pint import UnitRegistry

from microtool.constants import GAMMA, GAMMA_UNIT

ureg = UnitRegistry()
Q_ = ureg.Quantity
gamma_wunits = Q_(GAMMA, GAMMA_UNIT)
