import sys

if "pytest" in sys.modules:
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
    unit.define('milliTesla = [milliTesla] = mT')
    unit.define('radian = [radian] = rad')
else:
    class Dimensionless:
        def __rmul__(self, other):
            return other

    def unit(_: str):
        """
        When pytest is loaded, this is a pint UnitRegistry() object and calling it returns a Unit() object.
        When pytest is not loaded, the function returns Dimensionless(). Dimensionless() only allows right-hand
         multiplication and returns the object it is multiplied by.
        """
        return Dimensionless()
