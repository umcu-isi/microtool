class ScannerParameters:
    """

    :param t_90: Time for the 90 degree radio pulse in milliseconds
    :param t_180: Time for the 180 degree radio pulse in milliseconds
    :param t_half: The half readout time (or EPI pulse time) in milliseconds
    :param G_max: The maximum gradient strength in millitesla/millimeters
    :param S_max: The maximum slew rate in millitesla/millimeters/milliseconds
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
        :return: The inferred rise time in seconds
        """
        return self.G_max / self.S_max


default_scanner = ScannerParameters(4., 6., 14., 200e-3, 1300)
