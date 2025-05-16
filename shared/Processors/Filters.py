from IProcessor import IFilter, ParameterSpec, Filters

import numpy as np

class SMA(IFilter):
    def __init__(self):
        super().__init__("SMA", fullname="Simple Moving Average")
    
    def DefineDevParams(self):
        return [ ParameterSpec("Time", default=1.0, min=0.25, max=4.0) ]
    
    def Run(self, graph, devParams, params):
        # devParams[0] => "Time"
        n      = graph.shape[0]
        times  = graph[:,0]
        values = graph[:,1]

        starts = np.searchsorted(times, times - devParams[0], side="left") # (T,)
        cumsum = np.cumsum(np.insert(values, 0, 0.0)) # (T+1,)

        sums   = cumsum[1:] - cumsum[starts]  # (T,)
        counts = np.arange(1, n + 1) - starts # (T,)
        sma    = sums / counts
        return np.stack((times, sma), axis=1)
Filters().Define(SMA)

class EMA(IFilter):
    def __init__(self):
        super().__init__("EMA", fullname="Exponential Moving Average")
    
    def DefineDevParams(self):
        return [ ParameterSpec("Alpha", default=0.9, min=0.0, max=1.0) ]
    
    def Run(self, graph, devParams, params):
        # Not vectorizable
        # devParams[0] => "Alpha"
        alpha  = devParams[0]
        alpha2 = 1.0 - alpha

        weightedValues = alpha2 * graph[:,1]
        outGraph       = np.zeros_like(graph)
        outGraph[0,:]  = graph[0,:]
        for i in range(1, graph.shape[0]):
            outGraph[i,0] = graph[i,0]
            outGraph[i,1] = alpha * outGraph[i - 1,1] + weightedValues[i]
        return outGraph
Filters().Define(EMA)

class TEMA(IFilter):
    # Works identically to EMA with the only difference of how alpha and beta are calculated.
    # In this case we make the assumption that original gets exponentially smaller with time, I.e. original * alpha**t.
    # And t = now - previous
    def __init__(self):
        super().__init__("TEMA", fullname="Time-based Exponential Moving Average")
    
    def DefineDevParams(self):
        return [
            ParameterSpec("Beta", default=0.9, min=0.0, max=1.0),
            ParameterSpec("C", default=0.2, min=0.0, max=1.0)
        ]
    
    def Run(self, graph, devParams, params):
        # Not vectorizable
        # devParams[0] => "Beta"
        # devParams[1] => "C"
        beta = devParams[0]
        c    = devParams[1]

        times  = graph[:,0]
        values = graph[:,1]
        dt     = times[1:] - times[:-1]
        alpha  = beta**(dt * c)

        weightedValues = (1.0 - alpha) * values[1:]
        outGraph       = np.zeros_like(graph)
        outGraph[0,:]  = graph[0,:]
        for i in range(1, graph.shape[0]):
            outGraph[i,0] = graph[i,0]
            outGraph[i,1] = alpha[i - 1] * outGraph[i - 1,1] + weightedValues[i - 1]
        return outGraph
Filters().Define(TEMA)

class SC(IFilter):
    def __init__(self):
        super().__init__("SC", fullname="Sigma Clipping")
    
    def DefineDevParams(self):
        return [
            ParameterSpec("Time", default=1.0, min=0.25, max=4.0),
            ParameterSpec("k", default=2.0, min=0.0, max=5.0)
        ]
    
    def Run(self, graph, devParams, params):
        # Not vectorizable
        # devParams[0] => "Time"
        # devParams[1] => "k"
        k = devParams[1]

        n      = graph.shape[0]
        times  = graph[:,0]
        values = graph[:,1]

        starts  = np.searchsorted(times, times - devParams[0], side="left")
        cumsum  = np.cumsum(np.insert(values, 0, 0.0))
        cumsum2 = np.cumsum(np.insert(values**2, 0, 0.0))

        sums      = cumsum[1:] - cumsum[starts]
        sums2     = cumsum2[1:] - cumsum2[starts]
        counts    = np.arange(1, n + 1) - starts
        means     = sums / np.clip(counts, min=1, max=None)
        variances = (sums2 / np.clip(counts, min=1, max=None)) - (means**2)
        stddevs   = np.sqrt(np.maximum(variances, 0.0))

        m = counts.max()

        windowValues = np.full((n, m), np.nan)
        windowTimes  = np.full((n, m), np.nan)
        for i in range(n):
            start = starts[i]
            count = counts[i]
            windowValues[i,:count] = values[start:i+1]
            windowTimes[i,:count]  = times[start:i+1]
        
        absDiff    = np.abs(windowValues - means[:,None])
        inlierMask = absDiff <= (k * stddevs[:,None])

        maskedVals   = np.where(inlierMask, windowValues, 0.0)
        inlierCounts = inlierMask.sum(axis=1)
        inlierSums   = maskedVals.sum(axis=1)

        result = np.where(inlierCounts == 0, means, inlierSums / np.clip(inlierCounts, min=1, max=None))
        return np.stack((times, result), axis=1)
Filters().Define(SC)

class OSC(IFilter):
    def __init__(self):
        super().__init__("OSC", fullname="One-sided Sigma Clipping")
    
    def DefineDevParams(self):
        return [
            ParameterSpec("Time", default=1.0, min=0.25, max=4.0),
            ParameterSpec("k", default=2.0, min=0.0, max=5.0)
        ]
    
    def Run(self, graph, devParams, params):
        # Not vectorizable
        # devParams[0] => "Time"
        # devParams[1] => "k"
        k = devParams[1]

        n      = graph.shape[0]
        times  = graph[:,0]
        values = graph[:,1]

        starts  = np.searchsorted(times, times - devParams[0], side="left")
        cumsum  = np.cumsum(np.insert(values, 0, 0.0))
        cumsum2 = np.cumsum(np.insert(values**2, 0, 0.0))

        sums      = cumsum[1:] - cumsum[starts]
        sums2     = cumsum2[1:] - cumsum2[starts]
        counts    = np.arange(1, n + 1) - starts
        means     = sums / np.clip(counts, min=1, max=None)
        variances = (sums2 / np.clip(counts, min=1, max=None)) - (means**2)
        stddevs   = np.sqrt(np.maximum(variances, 0.0))

        m = counts.max()

        windowValues = np.full((n, m), np.nan)
        windowTimes  = np.full((n, m), np.nan)
        for i in range(n):
            start = starts[i]
            count = counts[i]
            windowValues[i,:count] = values[start:i+1]
            windowTimes[i,:count]  = times[start:i+1]
        
        diff       = windowValues - means[:,None]
        inlierMask = diff <= (k * stddevs[:,None])

        maskedVals   = np.where(inlierMask, windowValues, 0.0)
        inlierCounts = inlierMask.sum(axis=1)
        inlierSums   = maskedVals.sum(axis=1)

        result = np.where(inlierCounts == 0, means, inlierSums / np.clip(inlierCounts, min=1, max=None))
        return np.stack((times, result), axis=1)
Filters().Define(OSC)

class Kalman(IFilter):
    def __init__(self):
        super().__init__("Kalman", fullname="Kalman Filter")
    
    def DefineDevParams(self):
        return [
            ParameterSpec("ProcessNoise", default=0.1, min=0.0, max=1.0),
            ParameterSpec("MeasuredNoise", default=3.0, min=0.0, max=10.0),
            ParameterSpec("ProcessScaling", default=0.05, min=0.0, max=2.0)
        ]
    
    def Run(self, graph, devParams, params):
        # Not vectorizable
        # devParams[0] => "ProcessNoise"
        # devParams[1] => "MeasuredNoise"
        # devParams[2] => "ProcessScaling"
        processNoise   = devParams[0]
        measuredNoise  = devParams[1]
        processScaling = devParams[2]
        
        errorCovariance:float = 1.0

        outGraph = np.zeros_like(graph)
        outGraph[0,:] = graph[0,:]
        for i in range(1, graph.shape[0]):
            estimatePrev        = outGraph[i - 1,1]
            errorCovariancePrev = errorCovariance + processNoise

            kalmanGain      = errorCovariancePrev / (errorCovariancePrev + measuredNoise)
            estimate        = estimatePrev + kalmanGain * (graph[i,1] - estimatePrev)
            errorCovariance = (1.0 - kalmanGain) * errorCovariancePrev

            processNoise  = processScaling * abs(estimate - estimatePrev)
            outGraph[i,0] = graph[i,0]
            outGraph[i,1] = estimate
        return outGraph
Filters().Define(Kalman)

class AB(IFilter):
    def __init__(self):
        super().__init__("AB", fullname="Alpha-Beta Filter")
    
    def DefineDevParams(self):
        return [
            ParameterSpec("Alpha", default=0.85, min=0.0, max=1.0),
            ParameterSpec("Beta", default=0.005, min=0.0, max=1.0)
        ]
    
    def Run(self, graph, devParams, params):
        # Not vectorizable
        # devParams[0] => "Alpha"
        # devParams[1] => "Beta"
        alpha = devParams[0]
        beta  = devParams[1]
        
        times    = graph[:,0]
        dt       = times[1:] - times[:-1]
        prevRate = 0.0

        outGraph = np.zeros_like(graph)
        outGraph[0,:] = graph[0,:]
        for i in range(1, graph.shape[0]):
            estimate = outGraph[i - 1,1] + prevRate * dt[i - 1]
            r        = graph[i,1] - estimate

            estimate += alpha * r
            prevRate += (beta * r) / dt[i - 1]

            outGraph[i,0] = graph[i,0]
            outGraph[i,1] = estimate
        return outGraph
Filters().Define(AB)