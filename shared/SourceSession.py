from MACAddress import MACAddress

import numpy as np

class Sample:
    def __init__(self, timestamp:int, cpuTimestamp:int, rssi:int, crcOK:bool):
        self.Timestamp    = timestamp
        self.cpuTimestamp = cpuTimestamp
        self.Time         = timestamp * 1e-6
        self.RSSI         = rssi
        self.CRCOK        = crcOK

class Graph:
    def __init__(self):
        self.Samples:list[Sample] = []

    def __iter__(self):
        return self.Samples.__iter__()

    def __len__(self) -> int:
        return len(self.Samples)
    
    def __getitem__(self, key:int) -> Sample:
        return self.Samples.__getitem__(key)
    
    def __setitem__(self, key:int, value:Sample):
        self.Samples.__setitem__(key, value)

    def __contains__(self, key:int) -> bool:
        return self.Samples.__contains__(key)

class Period:
    def __init__(self):
        self.Graphs:dict[MACAddress, Graph] = {}

    def __iter__(self):
        return self.Graphs.items().__iter__()

    def __len__(self) -> int:
        return len(self.Graphs)
    
    def __getitem__(self, key:MACAddress) -> Graph:
        return self.Graphs.__getitem__(key)
    
    def __setitem__(self, key:MACAddress, value:Graph):
        self.Graphs.__setitem__(key, value)

    def __contains__(self, key:MACAddress) -> bool:
        return self.Graphs.__contains__(key)

class Session:
    def __init__(self):
        self.Periods:dict[str, Period] = {}

    def WithGoodCRC(self) -> dict[str, dict[MACAddress, np.ndarray]]:
        outSession:dict[str, dict[MACAddress, np.ndarray]] = {}
        for key, period in self:
            outPeriod:dict[MACAddress, np.ndarray] = {}
            for addr, graph in period:
                outGraph = [ (sample.Time, float(sample.RSSI)) for sample in graph if sample.CRCOK ]
                if len(outGraph) == 0:
                    continue
                outPeriod[addr] = np.array(outGraph)
            if len(outPeriod) > 0:
                outSession[key] = outPeriod
        return outSession

    def WithBadCRC(self) -> dict[str, dict[MACAddress, np.ndarray]]:
        outSession:dict[str, dict[MACAddress, np.ndarray]] = {}
        for key, period in self:
            outPeriod:dict[MACAddress, np.ndarray] = {}
            for addr, graph in period:
                outGraph = [ (sample.Time, float(sample.RSSI)) for sample in graph ]
                if len(outGraph) == 0:
                    continue
                outPeriod[addr] = np.array(outGraph)
            if len(outPeriod) > 0:
                outSession[key] = outPeriod
        return outSession

    def __iter__(self):
        return self.Periods.items().__iter__()

    def __len__(self) -> int:
        return len(self.Periods)
    
    def __getitem__(self, key:str) -> Period:
        return self.Periods.__getitem__(key)
    
    def __setitem__(self, key:str, value:Period):
        self.Periods.__setitem__(key, value)

    def __contains__(self, key:str) -> bool:
        return self.Periods.__contains__(key)