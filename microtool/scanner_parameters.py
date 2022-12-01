from dataclasses import dataclass


@dataclass
class ScannerParameters:
    t_90: float  # ms
    t_180: float  # ms
    # half readout time
    t_half: float
    G_max: float  # mT/mm
    # maximum slew rate
    S_max: float  # mT / mm / ms

    @property
    def t_rise(self):
        return self.G_max / self.S_max


default_scanner = ScannerParameters(4., 6., 14., 200e-3, 1300e-3)
