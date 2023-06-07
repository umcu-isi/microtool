from . import Q_


class ScannerParameters:
    def __init__(self, t_90: float, t_180: float, t_half: float, G_max: float, S_max: float):
        """

        :param t_90: Time for the 90 degree radio pulse in milliseconds
        :param t_180: Time for the 180 degree radio pulse in milliseconds
        :param t_half: The half readout time (or EPI pulse time) in milliseconds
        :param G_max: The maximum gradient strength in millitesla/millimeters
        :param S_max: The maximum slew rate in millitesla/millimeters/milliseconds
        """
        self.S_max = Q_(S_max, 'mT/(mm * ms)')
        self.G_max = Q_(G_max, 'mT/mm')
        self.t_half = Q_(t_half, 'ms')
        self.t_180 = Q_(t_180, 'ms')
        self.t_90 = Q_(t_90, 'ms')

    @property
    def t_rise(self) -> float:
        """
        :return: The inferred rise time in milliseconds
        """
        return self.G_max / self.S_max


default_scanner = ScannerParameters(4., 6., 14., 200e-3, 1300e-3)
