from .utils.unit_registry import unit


class ScannerParameters:
    """

    :param t_90: 90 degree pulse duration [s].
    :param t_180: 180 degree pulse duration [s].
    :param t_half: Half readout time (EPI pulse time) [s].
    :param g_max: Maximum gradient strength [mT/mm].
    :param s_max: Maximum slew rate [mT/mm/s].
    """

    def __init__(self,
                 t_90: float = 4e-3 * unit('s'),
                 t_180: float = 6e-3 * unit('s'),
                 t_half: float = 14e-3 * unit('s'),
                 g_max: float = 400e-3 * unit('mT/mm'),
                 s_max: float = 1300 * unit('mT/mm/s')):
        self.t_90 = t_90
        self.t_180 = t_180
        self.t_half = t_half
        self.g_max = g_max
        self.s_max = s_max

    @property
    def t_rise(self) -> float:
        """
        :return: The inferred rise time in seconds
        """
        return self.g_max / self.s_max


default_scanner = ScannerParameters()
