import math

from MACAddress import MACAddress
import SessionParameters as SP

import numpy as np

class DynamicPeriod:
    def __init__(self, parent = None):
        self.Graphs:list[np.ndarray]    = []
        self.TruePos:np.ndarray         = parent.TruePos if parent is not None else None
        """
        (T,3) as Time, X, Y
        """
        self.TrueDistances:list[np.ndarray] = parent.TrueDistances if parent is not None else None
        """
        List of (T,2) as Time, Distance for each beacon.
        """

class DynamicSession:
    def __init__(self, dynamicPaths:dict[str, list[tuple[float, float, float]]] = {}, source:dict[str, dict[MACAddress, np.ndarray]] = {}, sessionParameters:SP.Parameters = None, singular:bool = False):
        self.Periods:dict[int, DynamicPeriod] = {}
        self.Addresses:list[MACAddress]       = [ address for address in sessionParameters.Beacons.keys() ] if sessionParameters is not None else []

        for key, period in source.items():
            try:
                index = int(key)
            except:
                continue

            outPeriod           = DynamicPeriod()
            outPeriod.Graphs    = [ period[addr] if addr in period else None for addr in self.Addresses ] if not singular else [ period[None] ]
            self.Periods[index] = outPeriod

            if key in dynamicPaths:
                path = dynamicPaths[key]

                uniqueTimes = np.unique(np.concatenate([ graph[:,0] for graph in outPeriod.Graphs ]))
                
                interpolatedPoints:list[tuple[float, float]] = []
                for time in uniqueTimes.tolist():
                    for i in range(len(path)):
                        if path[i][0] >= time:
                            break
                    i = max(min(i, len(path) - 1), 1)

                    curPoint  = path[i - 1]
                    nextPoint = path[i]

                    alpha = (time - curPoint[0]) / (nextPoint[0] - curPoint[0])
                    x     = curPoint[1] + alpha * (nextPoint[1] - curPoint[1])
                    y     = curPoint[2] + alpha * (nextPoint[2] - curPoint[2])
                    
                    interpolatedPoints.append((x, y))
                interpolatedPoints = sessionParameters.Transform(interpolatedPoints)
                
                outPeriod.TruePos       = np.hstack((uniqueTimes[:,None], np.array(interpolatedPoints)))
                outPeriod.TrueDistances = []
                for i in range(len(self.Addresses)):
                    beacon = sessionParameters.Beacons[self.Addresses[i]]
                    pos    = np.array((beacon.RealPosX, beacon.RealPosY))
                    
                    trueDistances = np.linalg.norm(outPeriod.TruePos[:,1:3] - pos[None,:], axis=1)
                    trueDistances = np.stack((uniqueTimes, trueDistances), axis=-1)
                    outPeriod.TrueDistances.append(trueDistances)
    
    def __iter__(self):
        return self.Periods.items().__iter__()

    def __len__(self) -> int:
        return len(self.Periods)
    
    def __getitem__(self, key:int) -> DynamicPeriod:
        return self.Periods.__getitem__(key)
    
    def __setitem__(self, key:int, value:DynamicPeriod):
        self.Periods.__setitem__(key, value)

    def __contains__(self, key:int) -> bool:
        return self.Periods.__contains__(key)

class StaticPeriod:
    def __init__(self, parent = None):
        self.Graphs:list[np.ndarray]     = []
        self.TruePos:tuple[float, float] = parent.TruePos if parent is not None else None
        self.TrueDistances:np.ndarray    = parent.TrueDistances if parent is not None else None

class StaticSession:
    def __init__(self, source:dict[str, dict[MACAddress, np.ndarray]] = {}, sessionParameters:SP.Parameters = None, singular:bool = False):
        self.Periods:dict[tuple[float, float], StaticPeriod] = {}
        self.Addresses:list[MACAddress]                      = [ address for address in sessionParameters.Beacons.keys() ] if sessionParameters is not None else []
        self.TrueBeaconPos:np.ndarray                        = np.array([ (sessionParameters.Beacons[addr].RealPosX, sessionParameters.Beacons[addr].RealPosY) for addr in self.Addresses ])

        for key, period in source.items():
            try:
                coords = key.split("_")
                if len(coords) != 2:
                    continue
                pos = (float(coords[0]), float(coords[1]))
            except:
                continue
            outPeriod               = StaticPeriod()
            outPeriod.Graphs        = [ period[addr] if addr in period else None for addr in self.Addresses ] if not singular else [ period[None] ]
            outPeriod.TruePos       = sessionParameters.Transform(pos, snifferPoint=True)
            outPeriod.TrueDistances = np.linalg.norm(self.TrueBeaconPos - np.array(outPeriod.TruePos), axis=1)
            self.Periods[pos]       = outPeriod

            for i, graph in enumerate(outPeriod.Graphs):
                if graph is None:
                    continue
                end = np.searchsorted(graph[:,0], max(sessionParameters.WindowSizes), side="left")
                
                outPeriod.Graphs[i] = graph[:end,:].copy()

    def CutUp(self, windowSize:float):
        maxTime = 0.0
        for _, period in self:
            for graph in period.Graphs:
                if graph is None:
                    continue
                maxTime = max(maxTime, graph[:,0].max())
        
        curTime  = 0.0
        curIndex = 0
        outSessions:list[tuple[int, StaticSession]] = []
        while curTime <= maxTime:
            nextTime = curTime + windowSize

            session           = StaticSession()
            session.Addresses = self.Addresses.copy()
            for pos, period in self:
                outPeriod = StaticPeriod(period)
                
                containsData = False
                for i, graph in enumerate(period.Graphs):
                    if graph is None:
                        outPeriod.Graphs.append(None)
                        continue

                    start = np.searchsorted(graph[:,0], curTime, side="left")
                    end   = np.searchsorted(graph[:,0], nextTime, side="left")
                    if end <= start:
                        outPeriod.Graphs.append(None)
                        continue

                    outGraph = graph[start:end,:]
                    if len(outGraph) == 0:
                        outPeriod.Graphs.append(None)
                        continue

                    outPeriod.Graphs.append(np.array(outGraph))
                    containsData = True
                if containsData:
                    session[pos] = outPeriod
            if len(session) > 0:
                outSessions.append((curIndex, session))
            curTime   = nextTime
            curIndex += 1
        return outSessions
    
    def __iter__(self):
        return self.Periods.items().__iter__()

    def __len__(self) -> int:
        return len(self.Periods)
    
    def __getitem__(self, key:tuple[float, float]) -> StaticPeriod:
        return self.Periods.__getitem__(key)
    
    def __setitem__(self, key:tuple[float, float], value:StaticPeriod):
        self.Periods.__setitem__(key, value)

    def __contains__(self, key:tuple[float, float]) -> bool:
        return self.Periods.__contains__(key)