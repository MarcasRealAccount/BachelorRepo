from IProcessor import IDistance, ParameterSpec, Distances

import numpy as np

from timer import time

class Empirical(IDistance):
    def __init__(self):
        super().__init__("Empirical")

    def DefineDevParams(self):
        return [
            ParameterSpec("R1", default=-55.0, min=-65.0, max=-45.0),
            ParameterSpec("k", default=2.2, min=1.8, max=3.0)
        ]
    
    def _Base(self, rssis, devParams, params):
        # devParams[0] => "R1"
        # devParams[1] => "k"
        R1 = devParams[0]
        k  = devParams[1]
        f = 1.0 / (10.0 * k)
        return 10**((R1 - rssis) * f)
Distances().Define(Empirical)

class LDPL(IDistance):
    def __init__(self):
        super().__init__("LDPL", fullname="Log Distance Path Loss")

    def DefineDevParams(self):
        return [
            ParameterSpec("R1", default=-55.0, min=-65.0, max=-45.0),
            ParameterSpec("k", default=2.2, min=1.8, max=3.0),
            ParameterSpec("d", default=1.0, min=0.0),
            ParameterSpec("Seed", default=lambda addr, params: time(), tunable=False)
        ]
    
    def _Base(self, rssis, devParams, params):
        # devParams[0] => "R1"
        # devParams[1] => "k"
        # devParams[2] => "d"
        # devParams[3] => "Seed"
        R1 = devParams[0]
        k  = devParams[1]
        d  = devParams[2]
        f  = 1.0 / (10.0 * k)
        
        rng = np.random.default_rng(seed=int(devParams[3] * 1e9))
        return 10**((R1 - rssis - rng.normal(loc=0.0, scale=d, size=rssis.shape)) * f)
Distances().Define(LDPL)
# TODO: Implement randomness handling in a better way
# For instance, it could be possible to compute the distances for some number of deviations around the average
#   and then for the graph position we decide which combinations of those deviations we care about, producing a
#   distribution of possible positions, which we can then feed forward.
# But for the moment I will not do this, this could however be a point of improvement for whoever is reading this.

class FSPL(IDistance):
    def __init__(self):
        super().__init__("FSPL", fullname="Free Space Path Loss")

    def DefineDevParams(self):
        return [
            ParameterSpec("Gain", default=-20.0),
            ParameterSpec("Tx", default=lambda addr, sessionParameters:sessionParameters.Beacons[addr].Tx, tunable=False)
        ]
    
    def DefineParams(self):
        return [
            ParameterSpec("Frequency", default=lambda _, sessionParameters:sessionParameters.Sniffer.Frequency, tunable=False)
        ]

    def _Base(self, rssis, devParams, params):
        # devParams[0] => "Gain"
        # devParams[1] => "Tx"
        # params[0] => "Frequency"
        gain = devParams[0]
        Tx   = devParams[1]
        freq = params[0]
        pl   = Tx + gain - 20 * np.log10(freq) - 32.4477832219
        return 10**((pl - rssis) / 20)
Distances().Define(FSPL)

class Friis(IDistance):
    def __init__(self):
        super().__init__("Friis", fullname="Friis Transmission Equation Model")

    def DefineDevParams(self):
        return [
            ParameterSpec("Gain", default=-20.0),
            ParameterSpec("Tx", default=lambda addr, sessionParameters:sessionParameters.Beacons[addr].Tx, tunable=False)
        ]
    
    def DefineParams(self):
        return [
            ParameterSpec("Frequency", default=lambda _, sessionParameters:sessionParameters.Sniffer.Frequency, tunable=False)
        ]

    def _Base(self, rssis, devParams, params):
        # devParams[0] => "Gain"
        # devParams[1] => "Tx"
        # params[0] => "Frequency"
        gain  = devParams[0]
        Tx    = devParams[1]
        freq  = params[0]
        const = 0.02385672579618471129444449166887 # c / (4 * math.pi * 10^9)
        f     = const / freq
        pl    = Tx + gain
        return f * 10**((pl - rssis) / 20)
Distances().Define(Friis)

class ITUIPM(IDistance):
    def __init__(self):
        super().__init__("ITUIPM", fullname="ITU Indoor Propagation Model")

    def DefineDevParams(self):
        return [
            ParameterSpec("Tx", default=lambda addr, sessionParameters:sessionParameters.Beacons[addr].Tx)
        ]
    
    def DefineParams(self):
        return [
            ParameterSpec("Frequency", default=lambda _, sessionParameters:sessionParameters.Sniffer.Frequency, tunable=False)
        ]

    def _Base(self, rssis, devParams, params):
        # devParams[0] => "Tx"
        # params[0] => "Frequency"
        Tx   = devParams[0]
        freq = params[0]
        pl   = Tx - 20 * np.log10(freq * 1000) + 28
        return 10**((pl - rssis) / 30)
Distances().Define(ITUIPM)