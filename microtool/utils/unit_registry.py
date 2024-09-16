import sys

if "pytest" in sys.modules:
    from pint import UnitRegistry
    unit = UnitRegistry()
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
