import os
from typing import Any

import numpy as np


class Identity:
    # A dummy class that allows multiplication and division, in which it behaves like 1.
    def __array_ufunc__(self, ufunc, _method, *inputs, **_kwargs):
        # Support numpy array multiplication.
        if ufunc == np.multiply:
            return inputs[0]
        else:
            raise NotImplementedError

    def __mul__(self, other):
        return other

    def __rmul__(self, other):
        return other

    def __rtruediv__(self, other):
        return other


if os.environ.get('MICROTOOL_USE_UNITS') == '1':
    from pint import UnitRegistry

    # Create an empty UnitRegistry and add the units that we use explicitly.
    # By default, pint defines base units, like "meter = [length] = m" and prefixes like "milli- = 1e-3 = m-".
    # This allows pint to add e.g. millimeters to meters. We don't do that, since we're using pint for testing only. Our
    # code should also work without using pint. To test unit conversion, we therefore need to define conversion factors
    # explicitly too, like: m_to_mm = 1000 * unit('mm/m'). This ensures that not only the units, but also their
    # magnitudes are checked. Note that m_to_mm would evaluate to 1 (dimensionless) when pint's default UnitRegistry
    # would be used.
    unit = UnitRegistry(None)
    unit.define('meter = [meter] = m')
    unit.define('millimeter = [millimeter] = mm')
    unit.define('second = [second] = s')
    unit.define('millisecond = [millisecond] = ms')
    unit.define('Tesla = [Tesla] = T')
    unit.define('milliTesla = [milliTesla] = mT')
    unit.define('radian = [radian] = rad')

    def cast_to_ndarray(x: Any, unit_str: str = 'dimensionless') -> np.ndarray:
        # Check if the array has the proper units before casting it to a numpy array.
        if isinstance(x, unit.Quantity) and x.units != unit(unit_str):
            raise ValueError(f"Casted array has units '{x.units}', but expected '{unit_str}'")
        elif isinstance(x, (list, tuple)):
            for elem in x:
                if isinstance(elem, unit.Quantity) and elem.units != unit(unit_str):
                    raise ValueError(f"Casted array has units '{elem.units}', but expected '{unit_str}'")

        return np.array(x, copy=False)

else:
    def unit(_: str):
        """
        When pytest is loaded, this is a pint UnitRegistry() object and calling it returns a Unit() object.
        When pytest is not loaded, the function returns Identity(). This object allows multiplication and division,
        in which it behaves like 1.
        """
        return Identity()

    def cast_to_ndarray(x, unit_str: str = 'dimensionless'):
        """
        When pytest is loaded, this function will check if the argument has the proper units before (down)casting
         it to a numpy array.
        """
        return np.array(x, copy=False)
