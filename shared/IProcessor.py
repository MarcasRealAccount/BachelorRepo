from typing import TypeVar, Generic
from collections.abc import Callable

from MACAddress import MACAddress
import SessionParameters as SP
import Session

import numpy as np

# TODO: Implement randomness for LDPL.

class ParameterSpec:
    def __init__(self, name:str, default:float|Callable[[MACAddress, SP.Parameters],float], *,
                 min:float = None,
                 max:float = None,
                 tunable:bool = True):
        self.name    = name
        self.default = default
        self.min     = min
        self.max     = max
        self.tunable = tunable

    def GetValue(self, address:MACAddress, sessionParameters:SP.Parameters) -> float:
        if type(self.default) == float:
            return self.default
        else:
            return self.default(address, sessionParameters)
    
class IImpl:
    def __init__(self, name:str, fullname:str = None):
        self.name     = name
        self.fullname = fullname if fullname is not None else self.name

    def DefineDevParams(self) -> list[ParameterSpec]:
        """
        Defines device parameters this Implementation uses.
        
        This function should always return the same parameters.
        """
        return []
    
    def DefineParams(self) -> list[ParameterSpec]:
        """
        Defines global parameters this Implementation uses.
        
        This function should always return the same parameters.
        """
        return []
    
    def HasTunableParameters(self) -> bool:
        """
        Checks if this Implementation has any tunable parameters.
        """
        for spec in self.DefineDevParams():
            if spec.tunable:
                return True
        for spec in self.DefineParams():
            if spec.tunable:
                return True
        return False
    
    def ParamsToDict(self,
                     parameters:tuple[
                        list[MACAddress], # Addresses
                        np.ndarray,       # DevParams
                        np.ndarray        # Params
                     ]) -> dict[str, float]:
        """
        Takes the set of parameters and stores them in a dictionary.
        """
        devParamSpecs = self.DefineDevParams()
        paramSpecs    = self.DefineParams()

        outDict:dict[str, float] = {}
        for i, address in enumerate(parameters[0]):
            for j, spec in enumerate(devParamSpecs):
                if not spec.tunable:
                    continue
                outDict[f"{address}_{spec.name}"] = parameters[1][i,j]
        for i, spec in enumerate(paramSpecs):
            if not spec.tunable:
                continue
            outDict[spec.name] = parameters[2][i]
        return outDict
    
    def SetupParams(self, sessionParameters:SP.Parameters, parameters:dict[str, float] = {}) -> \
        tuple[
            list[MACAddress], # Addresses
            np.ndarray,       # DevParams
            np.ndarray,       # Min DevParams
            np.ndarray,       # Max DevParams
            np.ndarray,       # Params
            np.ndarray,       # Min Params
            np.ndarray        # Max Params
        ]:
        """
        Sets up parameters for the Implementation.
        return 1: list of Addresses
        return 2: (N,I) as Parameters.
        return 3: (N,I) as Parameter Minimum.
        return 4: (N,I) as Parameter Maximum.
        return 5: (J,) as Parameters.
        return 6: (J,) as Parameter Minimum.
        return 7: (J,) as Parameter Maximum.
        """
        devParamSpecs = self.DefineDevParams()
        paramSpecs    = self.DefineParams()

        devParams    = np.zeros((len(sessionParameters.Beacons), len(devParamSpecs)), dtype=float)
        devParamMins = np.zeros_like(devParams)
        devParamMaxs = np.zeros_like(devParams)
        params       = np.zeros((len(paramSpecs),), dtype=float)
        paramMins    = np.zeros_like(params)
        paramMaxs    = np.zeros_like(params)

        addresses:list[MACAddress] = []

        beaconItems = sessionParameters.Beacons.items()
        for i, (address, beacon) in enumerate(beaconItems):
            addresses.append(address)
            for j, spec in enumerate(devParamSpecs):
                paramKey          = f"{address}_{spec.name}"
                devParams[i,j]    = parameters[paramKey] if paramKey in parameters else spec.GetValue(address, sessionParameters)
                devParamMins[i,j] = (-float("inf") if spec.min is None else spec.min) if spec.tunable else devParams[i,j]
                devParamMaxs[i,j] = (float("inf") if spec.max is None else spec.max) if spec.tunable else devParams[i,j]
        for i, spec in enumerate(paramSpecs):
            params[i]    = parameters[spec.name] if spec.name in parameters else spec.GetValue(None, sessionParameters)
            paramMins[i] = (-float("inf") if spec.min is None else spec.min) if spec.tunable else params[i]
            paramMaxs[i] = (float("inf") if spec.max is None else spec.max) if spec.tunable else params[i]
        return (addresses, devParams, devParamMins, devParamMaxs, params, paramMins, paramMaxs)
    
class IFilter(IImpl):
    """
    Interface for a Filter.
    """
    def Run(self, graph:np.ndarray, devParams:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Runs the filter across the entire graph, should be implemented as close to vectorized code as possible.
        Order of parameters are the same as how they're defined.
        graph: (T,2) as Time, RSSI.
        devParams: (I,) as Parameter.
        params: (J,) as Parameter.
        return: (T,2) as Time, FilteredRSSI.
        """
        return None
    
class IDistance(IImpl):
    """
    Interface for a Distance Estimator.
    """
    def _Base(self, rssis:np.ndarray, devParams:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Base Distance Estimator implementation, runs for all rssi values, should be implemented as close to vectorized code as possible.
        rssis: (T,) as RSSI.
        devParams: (I,) as Parameter.
        params: (J,) as Parameter.
        return: (T,) as Distance, or (T,U,2) as distribution of Weight, Distance for each RSSI.
        """
        return None

    def Run(self, graph:np.ndarray, devParams:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Runs the Distance Estimator across the entire graph.
        Order of parameters are the same as how they're defined.
        graph: (T,2) as Time, FilteredRSSI.
        devParams: (I,) as Parameter.
        params: (J,) as Parameter.
        return: (T,2) as Time, Distance.
        """
        distances = self._Base(graph[:,1], devParams, params)
        return np.stack((graph[:,0], distances), axis=1)
    
    def RunDistribution(self, rssis:np.ndarray, devParams:np.ndarray, params:np.ndarray) -> np.ndarray:
        """
        Runs the Distance Estimator on unique RSSI values and returns a weighted distribution.
        Order of parameters are the same as how they're defined.
        rssis: (T,) as RSSI.
        devParams: (I,) as Parameter.
        params: (J,) as Parameter.
        return: (U,2) as Weight, Distance.
        """
        uniqueRSSIs, counts = np.unique(rssis, return_counts=True)
        distances           = self._Base(uniqueRSSIs, devParams, params)
        if distances.ndim == 1:
            return np.stack((counts / rssis.shape[0], distances), axis=1)
        
        weights   = distances[:,:,0] * counts[:,None]
        weights   = np.concatenate(weights, axis=0) / rssis.shape[0]
        distances = np.concatenate(distances, axis=0)

        uniqueDistances, indices, counts = np.unique(distances, return_index=True, return_counts=True)
        uniqueWeights = weights[indices] * counts
        return np.stack((uniqueWeights / np.sum(uniqueWeights), uniqueDistances), axis=1)
    
class IPosition(IImpl):
    """
    Interface for a Position Estimator.
    """
    def __init__(self, name, fullname = None, minGraphs:int = 2):
        super().__init__(name, fullname)
        self.MinGraphs = minGraphs

    def _Base(self, distances:np.ndarray, devParams:np.ndarray, params:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Base Position Estimator implementation, runs for all distances, should be implemented as close to vectorized as possible.
        distances: (N,T) as Distance to each beacon.
        devParams: (N,I) as Parameter.
        params: (J,) as Parameter.
        return 1: (T,2) as X, Y.
        return 2: (T,) as Valid.
        """
        return (np.zeros((distances.shape[1],2), dtype=float), np.zeros((distances.shape[1],), dtype=bool))

    def Run(self, graphs:list[np.ndarray], devParams:np.ndarray, params:np.ndarray, *, onlyValid:bool = True) -> np.ndarray:
        """
        Runs the distance estimator across the entire graph.
        Order of parameters are the same as how they're defined.
        graph: list of (Tn,2) as Time, Distance for each beacon.
        devParams: (N,I) as Parameter.
        params: (J,) as Parameter.
        return: (U,3) as Time, X, Y.
        """
        if len(graphs) < self.MinGraphs:
            return np.empty((0, 3))

        uniqueTimes = np.unique(np.concatenate([ graph[:,0] for graph in graphs ])) # (U,)
        indices     = [ np.searchsorted(graphs[i][:,0], uniqueTimes, side="right") - 1 for i in range(len(graphs)) ] # N list of (U,)
        distances   = np.array([ graphs[i][indices[i],1] for i in range(len(graphs)) ]) # (N,U)
        
        positions, mask = self._Base(distances, devParams, params) # (U,2), (U,)
        if onlyValid:
            validPositions = positions[mask]
            return np.stack((uniqueTimes[mask], validPositions[:,0], validPositions[:,1]), axis=1)
        return np.stack((uniqueTimes, positions[:,0], positions[:,1]), axis=1)
    
    def RunDistribution(self, distributions:list[np.ndarray], devParams:np.ndarray, params:np.ndarray, sampleCount:int) -> np.ndarray:
        """
        Runs the Position Estimator on sampled distribution of distances for each beacon.
        Order of parameters are the same as how they're defined.
        distributions: list of (Tn,2) as Weight, Distance for each beacon.
        devParams: (N,I) as Parameter.
        params: (J,) as Parameter.
        return: (U,3) as Weight, X, Y.
        """
        if len(distributions) < self.MinGraphs:
            return np.empty((0, 3))

        maxSamples  = np.prod([ graph.shape[0] for graph in distributions ])
        numSamples  = min(sampleCount, maxSamples)
        flatIndices = np.unique(np.random.choice(maxSamples, numSamples)) # (U,)
        indices     = np.array(np.unravel_index(flatIndices, shape=tuple([ graph.shape[0] for graph in distributions ]))) # (N,U)

        values    = np.array([ distributions[i][indices[i]] for i in range(indices.shape[0]) ]) # (N,U,2)
        distances = values[:,:,1]             # (N,U)
        weights   = values[:,:,0]             # (N,U)
        weights   = np.prod(weights, axis=0)  # (U,)
        weights   = weights / np.sum(weights) # (U,)

        positions, mask = self._Base(distances, devParams, params) # (U,2), (U,)
        validPositions  = positions[mask]

        uniquePositions, uniqueIndices, uniqueCounts = np.unique(validPositions, return_index=True, return_counts=True, axis=0)
        return np.stack((uniqueCounts * weights[mask][uniqueIndices], uniquePositions[:,0], uniquePositions[:,1]), axis=1)
    
T = TypeVar("T")

class IImpls(Generic[T]):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(IImpls, cls).__new__(cls)
        return cls.instance
    
    def __init__(self):
        super().__init__()
        self._impls:dict[str, T] = getattr(self, "_impls", {})
        self._skipped:set[str]   = getattr(self, "_skipped", set())
    
    def Define(self, clazz:type[T], *args, **kwargs):
        impl:T = clazz(*args, **kwargs)
        assert(impl.name not in self._impls)
        self._impls[impl.name] = impl

    def Skip(self, impls:list[str]):
        for impl in impls:
            self._skipped.add(impl)

    def GetProcessors(self) -> list[T]:
        impls:list[T] = []
        for name, impl in self._impls.items():
            if name in self._skipped:
                continue
            impls.append(impl)
        return impls

class Filters(IImpls[IFilter]): pass
class Distances(IImpls[IDistance]): pass
class Positions(IImpls[IPosition]): pass

class Triple:
    def __init__(self, filter:IFilter, distance:IDistance, position:IPosition):
        self.Filter   = filter
        self.Distance = distance
        self.Position = position

    def GetName(self) -> str:
        return f"{self.Filter.name}, {self.Distance.name}, {self.Position.name}" if self.Position is not None else f"{self.Filter.name}, {self.Distance.name}"
    
    def GetPath(self) -> str:
        return f"{self.Filter.name}_{self.Distance.name}_{self.Position.name}" if self.Position is not None else f"{self.Filter.name}_{self.Distance.name}"

    def GetCalibrationPath(self) -> str:
        return f"{self.Filter.name}_{self.Distance.name}_{self.Position.name}" if self.Position is not None and self.Position.HasTunableParameters() else f"{self.Filter.name}_{self.Distance.name}"

    def RunDistanceError(self, trueDistances:list[np.ndarray], graphs:list[np.ndarray],
                         filterDevParams:np.ndarray, filterParams:np.ndarray,
                         distanceDevParams:np.ndarray, distanceParams:np.ndarray) -> float:
        """
        Runs the Distance Estimator on the provided graphs, and returns the total error.
        trueDistances: list of (I,2) as Time, Distance for each beacon.
        graphs: list of (Tn,) as RSSI.
        distanceDevParams: (N,I1) as Parameters.
        distanceDevParams: (J1) as Parameters.
        """

        loss  = 0.0
        count = 0
        for i, graph in enumerate(graphs):
            if graph is None:
                continue

            filteredGraph = self.Filter.Run(graph, filterDevParams[i,:], filterParams)
            distances     = self.Distance.Run(filteredGraph, distanceDevParams[i,:], distanceParams)

            indices           = np.searchsorted(trueDistances[i][:,0], distances[:,0], side="right") - 1
            expectedDistances = trueDistances[i][indices,1]

            loss  += np.sum((expectedDistances - distances[:,1])**2)
            count += 1
        return (loss / count) if count > 0 else None
    
    def RunPositionError(self, truePos:np.ndarray, graphs:list[np.ndarray],
                         filterDevParams:np.ndarray, filterParams:np.ndarray,
                         distanceDevParams:np.ndarray, distanceParams:np.ndarray,
                         positionDevParams:np.ndarray, positionParams:np.ndarray) -> float:
        """
        Runs the Pair on the provided graphs, and returns the total error.
        truePos: (I,3) as Time, X, Y.
        graphs: list of (Tn,) as RSSI.
        distanceDevParams: (N,I1) as Parameters.
        distanceDevParams: (J1) as Parameters.
        positionDevParams: (N,I2) as Parameters.
        positionDevParams: (J2) as Parameters.
        """
        distanceGraphs:list[np.ndarray] = []
        for i, graph in enumerate(graphs):
            filteredGraph = self.Filter.Run(graph, filterDevParams[i,:], filterParams)
            distanceGraphs.append(self.Distance.Run(filteredGraph, distanceDevParams[i,:], distanceParams))
        indices:list[int] = []
        for i, graph in enumerate(distanceGraphs):
            if graph is None:
                continue
            indices.append(i)
        indicesArr = np.array(indices, dtype=int)
        positionGraph = self.Position.Run([ distanceGraphs[i] for i in indices ], positionDevParams[indicesArr,:], positionParams)
        
        indices           = np.searchsorted(truePos[:,0], positionGraph[:,0], side="right") - 1
        expectedPositions = truePos[indices,1:3]

        distances = np.linalg.norm(expectedPositions - positionGraph[:,1:3])
        return np.sum(distances) / positionGraph.shape[0]

    def Run(self, graphs:list[np.ndarray],
            filterDevParams:np.ndarray, filterParams:np.ndarray,
            distanceDevParams:np.ndarray, distanceParams:np.ndarray,
            positionDevParams:np.ndarray, positionParams:np.ndarray) -> \
        tuple[
            list[np.ndarray], # FilteredGraphs
            list[np.ndarray], # DistanceGraphs
            np.ndarray        # PositionGraph
        ]:
        """
        Runs the Triple on the provided graphs.
        graphs: list of (Tn,2) as Time, RSSI.
        filterDevParams: (N,I1) as Parameters.
        filterDevParams: (J1) as Parameters.
        distanceDevParams: (N,I2) as Parameters.
        distanceDevParams: (J2) as Parameters.
        positionDevParams: (N,I3) as Parameters.
        positionDevParams: (J3) as Parameters.
        return 1: list of (Tn,2) as Time, FilteredRSSI.
        return 2: list of (Tn,2) as Time, Distance.
        return 3: (U,3) as Time, X, Y.
        """
        filteredGraphs:list[np.ndarray] = []
        distanceGraphs:list[np.ndarray] = []
        for i, graph in enumerate(graphs):
            if graph is None:
                filteredGraphs.append(None)
                distanceGraphs.append(None)
                continue

            filteredGraph = self.Filter.Run(graph, filterDevParams[i,:], filterParams)
            distanceGraph = self.Distance.Run(filteredGraph, distanceDevParams[i,:], distanceParams)
            filteredGraphs.append(filteredGraph)
            distanceGraphs.append(distanceGraph)
        indices:list[int] = []
        for i, graph in enumerate(distanceGraphs):
            if graph is None:
                continue
            indices.append(i)
        indicesArr = np.array(indices, dtype=int)
        positionGraph = self.Position.Run([ distanceGraphs[i] for i in indices ], positionDevParams[indicesArr,:], positionParams)
        return (filteredGraphs, distanceGraphs, positionGraph)
    
class Pair:
    def __init__(self, distance:IDistance, position:IPosition):
        self.Distance = distance
        self.Position = position

    def GetName(self) -> str:
        return f"{self.Distance.name}, {self.Position.name}" if self.Position is not None else self.Distance.name
    
    def GetPath(self) -> str:
        return f"{self.Distance.name}_{self.Position.name}" if self.Position is not None else self.Distance.name

    def GetCalibrationPath(self) -> str:
        return f"{self.Distance.name}_{self.Position.name}" if self.Position is not None and self.Position.HasTunableParameters() else self.Distance.name

    def RunDistanceError(self, trueDistances:np.ndarray, graphs:list[np.ndarray],
                         distanceDevParams:np.ndarray, distanceParams:np.ndarray) -> float:
        """
        Runs the Distance Estimator on the provided graphs, and returns the total error.
        graphs: list of (Tn,2) as Time, RSSI.
        distanceDevParams: (N,I1) as Parameters.
        distanceDevParams: (J1) as Parameters.
        """

        loss = 0.0
        count = 0
        for i, graph in enumerate(graphs):
            if graph is None:
                continue
            distances = self.Distance.RunDistribution(graph[:,1], distanceDevParams[i,:], distanceParams)
            loss     += (trueDistances[i] - np.sum(distances[:,0] * distances[:,1]))**2
            count    += 1
        return loss / count if count > 0 else None
    
    def RunPositionError(self, truePos:tuple[float, float], graphs:list[np.ndarray],
                         distanceDevParams:np.ndarray, distanceParams:np.ndarray,
                         positionDevParams:np.ndarray, positionParams:np.ndarray, *,
                         sampleCount:int = 1_000) -> float:
        """
        Runs the Pair on the provided graphs, and returns the total error.
        graphs: list of (Tn,2) as Time, RSSI.
        distanceDevParams: (N,I1) as Parameters.
        distanceDevParams: (J1) as Parameters.
        positionDevParams: (N,I2) as Parameters.
        positionDevParams: (J2) as Parameters.
        """
        distanceGraphs:list[np.ndarray] = []
        for i, graph in enumerate(graphs):
            if graph is None:
                distanceGraphs.append(None)
                continue

            distanceGraphs.append(self.Distance.RunDistribution(graph[:,1], distanceDevParams[i], distanceParams))
        indices:list[int] = []
        for i, graph in enumerate(distanceGraphs):
            if graph is None:
                continue
            indices.append(i)
        indicesArr = np.array(indices, dtype=int)
        positionGraph = self.Position.RunDistribution([ distanceGraphs[i] for i in indices ], positionDevParams[indicesArr,:], positionParams, sampleCount)
        
        meanX           = np.sum(positionGraph[:,0] * positionGraph[:,1])
        meanY           = np.sum(positionGraph[:,0] * positionGraph[:,2])
        covXX           = np.sum(positionGraph[:,0] * (positionGraph[:,1] - meanX)**2)
        covYY           = np.sum(positionGraph[:,0] * (positionGraph[:,2] - meanY)**2)
        covXY           = np.sum(positionGraph[:,0] * (positionGraph[:,1] - meanX) * (positionGraph[:,2] - meanY))
        covMatrix       = np.array([[ covXX, covXY ], [ covXY, covYY ]])
        diff            = np.column_stack((truePos[0] - positionGraph[:,1], truePos[1] - positionGraph[:,2]))
        mahalanobisDist = np.sum(diff @ np.linalg.inv(covMatrix) * diff, axis=1)
        return np.sum(positionGraph[:,0] * (0.5 * np.log(np.linalg.det(covMatrix)) + 0.5 * mahalanobisDist))

    def Run(self, graphs:list[np.ndarray],
            distanceDevParams:np.ndarray, distanceParams:np.ndarray,
            positionDevParams:np.ndarray, positionParams:np.ndarray, *,
            sampleCount:int = 100_000) -> \
        tuple[
            list[np.ndarray], # DistanceGraphs
            np.ndarray        # PositionGraph
        ]:
        """
        Runs the Pair on the provided graphs.
        graphs: list of (Tn,) as RSSI.
        distanceDevParams: (N,I1) as Parameters.
        distanceDevParams: (J1) as Parameters.
        positionDevParams: (N,I2) as Parameters.
        positionDevParams: (J2) as Parameters.
        return 1: list of (Un,2) as Weight, Distance.
        return 2: (U,3) as Weight, X, Y.
        """
        
        distanceGraphs:list[np.ndarray] = []
        for i, graph in enumerate(graphs):
            if graph is None:
                distanceGraphs.append(None)
                continue

            distanceGraph = self.Distance.RunDistribution(graph[:,1], distanceDevParams[i,:], distanceParams)
            distanceGraphs.append(distanceGraph)
        indices:list[int] = []
        for i, graph in enumerate(distanceGraphs):
            if graph is None:
                continue
            indices.append(i)
        indicesArr = np.array(indices, dtype=int)
        positionGraph = self.Position.RunDistribution([ distanceGraphs[i] for i in indices], positionDevParams[indicesArr,:], positionParams, sampleCount)
        return (distanceGraphs, positionGraph)

# In order to ensure the processors are registered, we have to import the files they reside in
import Processors.Filters
import Processors.Distances
import Processors.Positions

def GetProcessorTriples() -> list[Triple]:
    triples:list[Triple] = []
    for filter in Filters().GetProcessors():
        for distance in Distances().GetProcessors():
            for position in Positions().GetProcessors():
                triples.append(Triple(filter, distance, position))
    return triples

def GetTunableProcessorTriples() -> list[Triple]:
    triples:list[Triple] = []
    for filter in Filters().GetProcessors():
        for distance in Distances().GetProcessors():
            needsGenericTuning = False
            for position in Positions().GetProcessors():
                if position.HasTunableParameters():
                    triples.append(Triple(filter, distance, position))
                else:
                    needsGenericTuning = True
            if needsGenericTuning:
                triples.append(Triple(filter, distance, None))
    return triples
    
def GetProcessorPairs() -> list[Pair]:
    pairs:list[Pair] = []
    for distance in Distances().GetProcessors():
        for position in Positions().GetProcessors():
            pairs.append(Pair(distance, position))
    return pairs

def GetTunableProcessorPairs() -> list[Pair]:
    pairs:list[Pair] = []
    for distance in Distances().GetProcessors():
        needsGenericTuning = False
        for position in Positions().GetProcessors():
            if position.HasTunableParameters():
                pairs.append(Pair(distance, position))
            else:
                needsGenericTuning = True
        if needsGenericTuning:
            pairs.append(Pair(distance, None))
    return pairs

class DynamicRun:
    def __init__(self, triple:Triple, badCRCs:bool, session:Session.DynamicSession):
        self.Triple  = triple
        self.BadCRCs = badCRCs
        self.Session = session

        self.Addresses:list[MACAddress] = []
        self.Params:np.ndarray          = None # Flattened parameters (T,)
        self.ParamMins:np.ndarray       = None # Flattened parameter minimums (T,)
        self.ParamMaxs:np.ndarray       = None # Flattened parameter maximums (T,)

        self.FilterDevParams:np.ndarray   = None # (N,I1)
        self.FilterParams:np.ndarray      = None # (J1,)
        self.DistanceDevParams:np.ndarray = None # (N,I2)
        self.DistanceParams:np.ndarray    = None # (J2,)
        self.PositionDevParams:np.ndarray = None # (N,I3)
        self.PositionParams:np.ndarray    = None # (J3,)

        self.OptimizePosition:bool = self.Triple is not None and self.Triple.Position is not None and self.Triple.Position.HasTunableParameters()

    def GetName(self) -> str:
        return self.Triple.GetName() if self.Triple is not None else "Raw"
    
    def GetPath(self) -> str:
        return self.Triple.GetPath() if self.Triple is not None else "Raw"
    
    def GetCalibrationPath(self) -> str:
        return self.Triple.GetCalibrationPath()

    def _RunDistanceError(self, period:Session.StaticPeriod, device:int) -> float:
        return self.Triple.RunDistanceError(
            period.TrueDistances[device:device+1] if device is not None else period.TrueDistances,
            [ period.Graphs[device] ] if device is not None else period.Graphs,
            self.FilterDevParams[device:device+1,:] if device is not None else self.FilterDevParams, self.FilterParams,
            self.DistanceDevParams[device:device+1,:] if device is not None else self.DistanceDevParams, self.DistanceParams)

    def _RunPositionError(self, period:Session.StaticPeriod, sampleCount:int) -> float:
        return self.Triple.RunPositionError(period.TruePos, period.Graphs,
            self.FilterDevParams, self.FilterParams,
            self.DistanceDevParams, self.DistanceParams,
            self.PositionDevParams, self.PositionParams)
    
    def _RunFullDistanceError(self, device:int) -> float:
        totalError = 0.0
        count      = 0
        for pos, period in self.Session:
            if period.TruePos is None or period.TrueDistances is None:
                continue
            error = self._RunDistanceError(period, device)
            if error is not None:
                totalError += error
                count      += 1
        return totalError / count
    
    def _RunFullPositionError(self) -> float:
        totalError = 0.0
        count      = 0
        for pos, period in self.Session:
            if period.TruePos is None or period.TrueDistances is None:
                continue
            error = self._RunPositionError(period)
            if error is not None:
                totalError += error
                count      += 1
        return totalError / count

    def RunError(self, *,
                 device:int = None) -> float:
        """
        Performs this run, and returns the total error.
        """
        if self.OptimizePosition:
            return self._RunFullPositionError()
        return self._RunFullDistanceError(device)

    def Run(self) -> tuple[Session.DynamicSession, Session.DynamicSession, Session.DynamicSession]:
        """
        Performs this run.
        """
        outFilteredSession = Session.DynamicSession()
        outDistanceSession = Session.DynamicSession()
        outPositionSession = Session.DynamicSession()
        outFilteredSession.Addresses = self.Session.Addresses.copy()
        outDistanceSession.Addresses = self.Session.Addresses.copy()
        for index, period in self.Session:
            filterGraphs, distanceGraphs, positionGraph = self.Triple.Run(period.Graphs, self.FilterDevParams, self.FilterParams, self.DistanceDevParams, self.DistanceParams, self.PositionDevParams, self.PositionParams)
            
            outFilteredPeriod         = Session.DynamicPeriod(period)
            outFilteredPeriod.Graphs  = filterGraphs
            outFilteredSession[index] = outFilteredPeriod

            outDistancePeriod         = Session.DynamicPeriod(period)
            outDistancePeriod.Graphs  = distanceGraphs
            outDistanceSession[index] = outDistancePeriod

            outPositionPeriod         = Session.DynamicPeriod(period)
            outPositionPeriod.Graphs  = [ positionGraph ]
            outPositionSession[index] = outPositionPeriod
        return (outFilteredSession, outDistanceSession, outPositionSession)

    def ParamsToDict(self) -> dict[str, dict[str, float]]:
        """
        Creates a dictionary of all the tunable parameters.
        """
        params:dict[str, dict[str, float]] = {}
        params["Filter"]   = self.Triple.Filter.ParamsToDict([ self.Addresses, self.FilterDevParams, self.FilterParams ])
        params["Distance"] = self.Triple.Distance.ParamsToDict([ self.Addresses, self.DistanceDevParams, self.DistanceParams ])
        if self.Triple.Position is not None:
            params["Position"] = self.Triple.Position.ParamsToDict([ self.Addresses, self.PositionDevParams, self.PositionParams ])
        return params
    
    def RemapParams(self):
        cur  = 0
        next = cur + self.FilterDevParams.size
        self.FilterDevParams = self.Params[cur:next].reshape(self.FilterDevParams.shape)
        cur  = next
        next = cur + self.FilterParams.size
        self.FilterParams = self.Params[cur:next].reshape(self.FilterParams.shape)
        cur  = next
        next = cur + self.DistanceDevParams.size
        self.DistanceDevParams = self.Params[cur:next].reshape(self.DistanceDevParams.shape)
        cur  = next
        next = cur + self.DistanceParams.size
        self.DistanceParams = self.Params[cur:next].reshape(self.DistanceParams.shape)
        if self.Triple.Position is not None:
            cur  = next
            next = cur + self.PositionDevParams.size
            self.PositionDevParams = self.Params[cur:next].reshape(self.PositionDevParams.shape)
            cur  = next
            next = cur + self.PositionParams.size
            self.PositionParams = self.Params[cur:next].reshape(self.PositionParams.shape)

    def SetupParams(self, sessionParameters:SP.Parameters, parameters:dict[str, dict[str, float]] = {}):
        """
        Sets up parameters for this run.
        If given a parameters dictionary, this will reload whatever parameters were provided.
        """
        filterParams   = self.Triple.Filter.SetupParams(sessionParameters, parameters["Filter"] if "Filter" in parameters else {})
        distanceParams = self.Triple.Distance.SetupParams(sessionParameters, parameters["Distance"] if "Distance" in parameters else {})
        self.Addresses = filterParams[0]
        if self.Triple.Position is not None:
            positionParams = self.Triple.Position.SetupParams(sessionParameters, parameters["Position"] if "Position" in parameters else {})
            self.Params    = np.concatenate((
                filterParams[1].reshape((filterParams[1].size,)),
                filterParams[4].reshape((filterParams[4].size,)),
                distanceParams[1].reshape((distanceParams[1].size,)),
                distanceParams[4].reshape((distanceParams[4].size,)),
                positionParams[1].reshape((positionParams[1].size,)),
                positionParams[4].reshape((positionParams[4].size,))), axis=0)
            self.ParamMins = np.concatenate((
                filterParams[2].reshape((filterParams[2].size,)),
                filterParams[5].reshape((filterParams[5].size,)),
                distanceParams[2].reshape((distanceParams[2].size,)),
                distanceParams[5].reshape((distanceParams[5].size,)),
                positionParams[2].reshape((positionParams[2].size,)),
                positionParams[5].reshape((positionParams[5].size,))), axis=0)
            self.ParamMaxs = np.concatenate((
                filterParams[3].reshape((filterParams[3].size,)),
                filterParams[6].reshape((filterParams[6].size,)),
                distanceParams[3].reshape((distanceParams[3].size,)),
                distanceParams[6].reshape((distanceParams[6].size,)),
                positionParams[3].reshape((positionParams[3].size,)),
                positionParams[6].reshape((positionParams[6].size,))), axis=0)
        else:
            self.Params    = np.concatenate((
                filterParams[1].reshape((filterParams[1].size,)),
                filterParams[4].reshape((filterParams[4].size,)),
                distanceParams[1].reshape((distanceParams[1].size,)),
                distanceParams[4].reshape((distanceParams[4].size,))), axis=0)
            self.ParamMins = np.concatenate((
                filterParams[2].reshape((filterParams[2].size,)),
                filterParams[5].reshape((filterParams[5].size,)),
                distanceParams[2].reshape((distanceParams[2].size,)),
                distanceParams[5].reshape((distanceParams[5].size,))), axis=0)
            self.ParamMaxs = np.concatenate((
                filterParams[3].reshape((filterParams[3].size,)),
                filterParams[6].reshape((filterParams[6].size,)),
                distanceParams[3].reshape((distanceParams[3].size,)),
                distanceParams[6].reshape((distanceParams[6].size,))), axis=0)
            
        cur  = 0
        next = cur + filterParams[1].size
        self.FilterDevParams = self.Params[cur:next].reshape(filterParams[1].shape)
        cur  = next
        next = cur + filterParams[4].size
        self.FilterParams = self.Params[cur:next].reshape(filterParams[4].shape)
        cur  = next
        next = cur + distanceParams[1].size
        self.DistanceDevParams = self.Params[cur:next].reshape(distanceParams[1].shape)
        cur  = next
        next = cur + distanceParams[4].size
        self.DistanceParams = self.Params[cur:next].reshape(distanceParams[4].shape)
        if self.Triple.Position is not None:
            cur  = next
            next = cur + positionParams[1].size
            self.PositionDevParams = self.Params[cur:next].reshape(positionParams[1].shape)
            cur  = next
            next = cur + positionParams[4].size
            self.PositionParams = self.Params[cur:next].reshape(positionParams[4].shape)
    
class StaticRun:
    def __init__(self, pair:Pair, badCRCs:bool, windowSize:float, windowIndex:int, session:Session.StaticSession):
        self.Pair        = pair
        self.BadCRCs     = badCRCs
        self.WindowSize  = windowSize
        self.WindowIndex = windowIndex
        self.Session     = session

        self.Addresses:list[MACAddress] = []
        self.Params:np.ndarray          = None # Flattened parameters (T,)
        self.ParamMins:np.ndarray       = None # Flattened parameter minimums (T,)
        self.ParamMaxs:np.ndarray       = None # Flattened parameter maximums (T,)

        self.DistanceDevParams:np.ndarray = None # (N,I1)
        self.DistanceParams:np.ndarray    = None # (J1,)
        self.PositionDevParams:np.ndarray = None # (N,I2)
        self.PositionParams:np.ndarray    = None # (J2,)

        self.OptimizePosition:bool = self.Pair is not None and self.Pair.Position is not None and self.Pair.Position.HasTunableParameters()

    def GetName(self) -> str:
        return self.Pair.GetName() if self.Pair is not None else "Raw"
    
    def GetPath(self) -> str:
        return self.Pair.GetPath() if self.Pair is not None else "Raw"
    
    def GetCalibrationPath(self) -> str:
        return self.Pair.GetCalibrationPath()

    def SelectRandomPeriods(self, count:int) -> list[int]:
        return np.random.choice(len(self.Session), size=(count,), replace=True).tolist()

    def _RunDistanceError(self, period:Session.StaticPeriod, device:int) -> float:
        return self.Pair.RunDistanceError(
            period.TrueDistances[device:device+1] if device is not None else period.TrueDistances,
            [ period.Graphs[device] ] if device is not None else period.Graphs,
            self.DistanceDevParams[device:device+1,:] if device is not None else self.DistanceDevParams, self.DistanceParams)

    def _RunPositionError(self, period:Session.StaticPeriod, sampleCount:int) -> float:
        return self.Pair.RunPositionError(period.TruePos, period.Graphs,
            self.DistanceDevParams, self.DistanceParams,
            self.PositionDevParams, self.PositionParams,
            sampleCount=sampleCount)

    def _RunIndexedDistanceError(self, indices:list[int], device:int) -> float:
        periodItems = list(self.Session.Periods.items())
        totalError  = 0.0
        count       = 0
        for i in indices:
            error = self._RunDistanceError(periodItems[i][1], device)
            if error is not None:
                totalError += error
                count      += 1
        return totalError / count
    
    def _RunFullDistanceError(self, device:int) -> float:
        totalError = 0.0
        count      = 0
        for pos, period in self.Session:
            error = self._RunDistanceError(period, device)
            if error is not None:
                totalError += error
                count      += 1
        return totalError / count
    
    def _RunIndexedPositionError(self, indices:list[int], sampleCount:int) -> float:
        periodItems = list(self.Session.Periods.items())
        totalError  = 0.0
        count       = 0
        for i in indices:
            error = self._RunPositionError(periodItems[i][1], sampleCount)
            if error is not None:
                totalError += error
                count      += 1
        return totalError / count
    
    def _RunFullPositionError(self, sampleCount:int) -> float:
        totalError = 0.0
        count      = 0
        for pos, period in self.Session:
            error = self._RunPositionError(period, sampleCount)
            if error is not None:
                totalError += error
                count      += 1
        return totalError / count

    def RunError(self, *,
                 indices:list[int] = None,
                 device:int = None,
                 sampleCount:int = 1_000) -> float:
        """
        Performs this run, and returns the total error.
        """
        if indices is not None:
            if self.OptimizePosition:
                return self._RunIndexedPositionError(indices, sampleCount)
            return self._RunIndexedDistanceError(indices, device)
        if self.OptimizePosition:
            return self._RunFullPositionError(sampleCount)
        return self._RunFullDistanceError(device)

    def Run(self, *,
            sampleCount:int = 100_000) -> tuple[Session.StaticSession, Session.StaticSession]:
        """
        Performs this run.
        """
        outDistanceSession = Session.StaticSession({})
        outPositionSession = Session.StaticSession({})
        outDistanceSession.Addresses = self.Session.Addresses.copy()
        for pos, period in self.Session:
            distanceGraphs, positionGraph = self.Pair.Run(period.Graphs, self.DistanceDevParams, self.DistanceParams, self.PositionDevParams, self.PositionParams,
                          sampleCount=sampleCount)
            
            outDistancePeriod        = Session.StaticPeriod(period)
            outDistancePeriod.Graphs = distanceGraphs
            outDistanceSession[pos]  = outDistancePeriod

            outPositionPeriod        = Session.StaticPeriod(period)
            outPositionPeriod.Graphs = [ positionGraph ]
            outPositionSession[pos]  = outPositionPeriod
        return (outDistanceSession, outPositionSession)

    def ParamsToDict(self) -> dict[str, dict[str, float]]:
        """
        Creates a dictionary of all the tunable parameters.
        """
        params:dict[str, dict[str, float]] = {}
        params["Distance"] = self.Pair.Distance.ParamsToDict([ self.Addresses, self.DistanceDevParams, self.DistanceParams ])
        if self.Pair.Position is not None:
            params["Position"] = self.Pair.Position.ParamsToDict([ self.Addresses, self.PositionDevParams, self.PositionParams ])
        return params
    
    def RemapParams(self):
        cur  = 0
        next = cur + self.DistanceDevParams.size
        self.DistanceDevParams = self.Params[cur:next].reshape(self.DistanceDevParams.shape)
        cur  = next
        next = cur + self.DistanceParams.size
        self.DistanceParams = self.Params[cur:next].reshape(self.DistanceParams.shape)
        if self.Pair.Position is not None:
            cur  = next
            next = cur + self.PositionDevParams.size
            self.PositionDevParams = self.Params[cur:next].reshape(self.PositionDevParams.shape)
            cur  = next
            next = cur + self.PositionParams.size
            self.PositionParams = self.Params[cur:next].reshape(self.PositionParams.shape)

    def SetupParams(self, sessionParameters:SP.Parameters, parameters:dict[str, dict[str, float]] = {}):
        """
        Sets up parameters for this run.
        If given a parameters dictionary, this will reload whatever parameters were provided.
        """
        distanceParams = self.Pair.Distance.SetupParams(sessionParameters, parameters["Distance"] if "Distance" in parameters else {})
        self.Addresses = distanceParams[0]
        if self.Pair.Position is not None:
            positionParams = self.Pair.Position.SetupParams(sessionParameters, parameters["Position"] if "Position" in parameters else {})
            self.Params    = np.concatenate((
                distanceParams[1].reshape((distanceParams[1].size,)),
                distanceParams[4].reshape((distanceParams[4].size,)),
                positionParams[1].reshape((positionParams[1].size,)),
                positionParams[4].reshape((positionParams[4].size,))), axis=0)
            self.ParamMins = np.concatenate((
                distanceParams[2].reshape((distanceParams[2].size,)),
                distanceParams[5].reshape((distanceParams[5].size,)),
                positionParams[2].reshape((positionParams[2].size,)),
                positionParams[5].reshape((positionParams[5].size,))), axis=0)
            self.ParamMaxs = np.concatenate((
                distanceParams[3].reshape((distanceParams[3].size,)),
                distanceParams[6].reshape((distanceParams[6].size,)),
                positionParams[3].reshape((positionParams[3].size,)),
                positionParams[6].reshape((positionParams[6].size,))), axis=0)
        else:
            self.Params    = np.concatenate((
                distanceParams[1].reshape((distanceParams[1].size,)),
                distanceParams[4].reshape((distanceParams[4].size,))), axis=0)
            self.ParamMins = np.concatenate((
                distanceParams[2].reshape((distanceParams[2].size,)),
                distanceParams[5].reshape((distanceParams[5].size,))), axis=0)
            self.ParamMaxs = np.concatenate((
                distanceParams[3].reshape((distanceParams[3].size,)),
                distanceParams[6].reshape((distanceParams[6].size,))), axis=0)
    
        cur  = 0
        next = cur + distanceParams[1].size
        self.DistanceDevParams = self.Params[cur:next].reshape(distanceParams[1].shape)
        cur  = next
        next = cur + distanceParams[4].size
        self.DistanceParams = self.Params[cur:next].reshape(distanceParams[4].shape)
        if self.Pair.Position is not None:
            cur  = next
            next = cur + positionParams[1].size
            self.PositionDevParams = self.Params[cur:next].reshape(positionParams[1].shape)
            cur  = next
            next = cur + positionParams[4].size
            self.PositionParams = self.Params[cur:next].reshape(positionParams[4].shape)

def DynamicInitErrorFunc(userdata:dict):
    run:DynamicRun = userdata["Run"]
    run.Params     = run.Params.copy()
    run.RemapParams()

def DynamicErrorFunc(userdata:dict, params:np.ndarray):
    device:int      = userdata["Device"]
    run:DynamicRun  = userdata["Run"]
    run.Params[...] = params
    return run.RunError(device=device)

def StaticInitErrorFunc(userdata:dict):
    run:StaticRun = userdata["Run"]
    run.Params    = run.Params.copy()
    run.RemapParams()

def StaticErrorFunc(userdata:dict, params:np.ndarray):
    device:int      = userdata["Device"]
    run:StaticRun   = userdata["Run"]
    run.Params[...] = params
    return run.RunError(device=device)