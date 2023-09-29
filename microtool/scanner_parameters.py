class ScannerParameters:
    """

    :param t_90: Time for the 90 degree radio pulse in seconds
    :param t_180: Time for the 180 degree radio pulse in seconds
    :param t_half: The half readout time (or EPI pulse time) in seconds
    :param G_max: The maximum gradient strength in millitesla/millimeters
    :param S_max: The maximum slew rate in millitesla/millimeters/seconds
    """

    def __init__(self, t_90: float, t_180: float, t_half: float, G_max: float, S_max: float):
        self.S_max = S_max
        self.G_max = G_max
        self.t_half = t_half
        self.t_180 = t_180
        self.t_90 = t_90

    @property
    def t_rise(self) -> float:
        """
        :return: The inferred rise time in milliseconds
        """
        return self.G_max / self.S_max


default_scanner = ScannerParameters(4.e-3, 6.e-3, 14.e-3, 400e-3, 1300)
