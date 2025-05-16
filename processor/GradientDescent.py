from collections.abc import Callable
from multiprocessing import Pool, Manager, Queue
from threading import Thread

import numpy as np

from timer import time_ns

_SigmoidK = 1.0

def _InvSigmoid(parameters:np.ndarray, minValues:np.ndarray, maxValues:np.ndarray) -> np.ndarray:
    """
    Converts parameter space into optimization space (For making boundaries become infinities)
    """
    minBounded = np.isfinite(minValues)
    maxBounded = np.isfinite(maxValues)
    minValues  = np.where(minBounded, minValues, 0.0)
    maxValues  = np.where(maxBounded, maxValues, 0.0)

    s   = np.where(minBounded & maxBounded, maxValues - minValues, 2 * _SigmoidK)
    f2  = 0.5 * (maxValues + minValues)
    f3  = maxValues - _SigmoidK
    f4  = minValues + _SigmoidK
    f   = np.where(minBounded & maxBounded, f2, np.where(minBounded, f4, np.where(maxBounded, f3, 0.0)))
    En  = (s / np.pi) * np.tan(np.pi * (parameters - f) / np.where(s == 0.0, 1e-9, s))
    f1n = parameters
    f2n = En
    f3n = np.where(parameters <= f3, parameters - f3, En)
    f4n = np.where(parameters >= f4, parameters - f4, En)
    return np.where(minBounded & maxBounded, np.where(minValues == maxValues, minValues, f2n), np.where(minBounded, f4n, np.where(maxBounded, f3n, f1n)))

def _Sigmoid(parameters:np.ndarray, minValues:np.ndarray, maxValues:np.ndarray) -> np.ndarray:
    """
    Converts optimization space into parameter space (For making boundaries become infinities)
    """
    minBounded = np.isfinite(minValues)
    maxBounded = np.isfinite(maxValues)
    minValues  = np.where(minBounded, minValues, 0.0)
    maxValues  = np.where(maxBounded, maxValues, 0.0)

    s  = np.where(minBounded & maxBounded, maxValues - minValues, 2 * _SigmoidK)
    E  = (s / np.pi) * np.arctan(np.pi * parameters / np.where(s == 0.0, 1e-9, s))
    f1 = parameters
    f2 = 0.5 * (maxValues + minValues) + E
    f3 = maxValues - _SigmoidK + np.where(parameters <= 0.0, parameters, E)
    f4 = minValues + _SigmoidK + np.where(parameters >= 0.0, parameters, E)
    return np.where(minBounded & maxBounded, np.where(minValues == maxValues, minValues, f2), np.where(minBounded, f4, np.where(maxBounded, f3, f1)))

def _Gradient(parameters:np.ndarray, minValues:np.ndarray, maxValues:np.ndarray, learningRate:float, errorFunc:Callable[[any, np.ndarray], float], userdata) -> np.ndarray:
    """
    Calculates the Gradient of `errorFunc` using Center Difference in optimization space.
    """
    gradient = np.zeros_like(parameters)
    for index in np.ndindex(parameters.shape):
        # Skip fixed parameters
        if minValues[index] == maxValues[index]:
            continue

        origValue = parameters[index]
        
        parameters[index] = origValue - learningRate
        minErr = errorFunc(userdata, _Sigmoid(parameters, minValues, maxValues))
        parameters[index] = origValue + learningRate
        maxErr = errorFunc(userdata, _Sigmoid(parameters, minValues, maxValues))
        parameters[index] = origValue
        
        gradient[index] = (maxErr - minErr) / (2 * learningRate)
    denom = np.linalg.norm(gradient)
    return (gradient / denom) if denom != 0.0 else np.zeros_like(parameters)

def _Descent(params:np.ndarray, errorFunc:Callable[[any, np.ndarray], float], userdata,
             minValues:np.ndarray, maxValues:np.ndarray,
             learningRate:float, decay:float,
             minLearningRate:float, maxIterations:int,
             progress:Callable[[float, float], None]):
    lastGradient = None
    iteration    = 0
    while iteration < maxIterations:
        gradient = _Gradient(params, minValues, maxValues, learningRate, errorFunc, userdata)
        if np.all(gradient == 0.0):
            break

        dp = np.dot(lastGradient, gradient) if lastGradient is not None else 1.0
        if dp >= 0.0:
            params      -= gradient * learningRate
            lastGradient = gradient
        learningRate *= decay**(np.where(dp >= 0.0, -1.0, 1.0))
        if learningRate <= minLearningRate:
            break

        err = errorFunc(userdata, _Sigmoid(params, minValues, maxValues))
        progress(err, learningRate)
        iteration += 1

def _DescentMP(parameters:np.ndarray, errorFunc:Callable[[any, np.ndarray], float], initFunc:Callable[[any],None], userdata,
               minValues:np.ndarray, maxValues:np.ndarray,
               learningRate:float, decay:float,
               minLearningRate:float, maxIterations:int, fastSearch:int, diversity:np.ndarray,
               sharedBest:dict, lock, queue:Queue):
    params = parameters.copy()
    initFunc(userdata)

    minBounded  = np.isfinite(minValues)
    maxBounded  = np.isfinite(maxValues)
    minValues2  = np.where(minBounded, minValues, 0.0)
    maxValues2  = np.where(maxBounded, maxValues, 0.0)
    boundDeltas = np.where(~minBounded & ~maxBounded, 10.0 * 0.25, (maxValues2 - minValues2) * 0.25) * diversity
    bestError   = errorFunc(userdata, _Sigmoid(params, minValues, maxValues))
    bestParams  = params
    with lock:
        if bestError < sharedBest["error"]:
            sharedBest["error"]  = bestError
            sharedBest["params"] = bestParams.copy()
            queue.put((bestError, learningRate))
    for j in range(fastSearch):
        rand       = np.random.uniform(-0.5, 0.5, size=params.shape)
        randParams = params + rand * boundDeltas
        error      = errorFunc(userdata, _Sigmoid(randParams, minValues, maxValues))
        if error < bestError:
            bestError  = error
            bestParams = randParams
            with lock:
                if bestError < sharedBest["error"]:
                    sharedBest["error"]  = bestError
                    sharedBest["params"] = bestParams.copy()
                    queue.put((bestError, learningRate))
    params = bestParams

    def progress(error, learningRate):
        with lock:
            if error < sharedBest["error"]:
                sharedBest["error"]  = error
                sharedBest["params"] = params.copy()
                queue.put((error, learningRate))

    _Descent(params, errorFunc, userdata, minValues, maxValues, learningRate, decay, minLearningRate, maxIterations, progress)

def GradientDescent(parameters:np.ndarray, errorFunc:Callable[[any, np.ndarray], float], initFunc:Callable[[any],None], userdata, *,
                    minValues:np.ndarray = None, maxValues:np.ndarray = None,
                    learningRate:float = 1e-6, decay:float = 0.8,
                    minLearningRate:float = 1e-64,
                    maxIterations:int = 1_000,
                    progress:Callable[[float], None] = None, **kwargs) -> tuple[np.ndarray, float]:
    """
    Perform Gradient Descent following the formula below:
    ```
    a_i+1            = a_i - learningRate_i * gradient F(a_i), i >= 0
    learningRate_i+1 = learningRate_i * decay^(-sgn(a_i+1 dot a_i))
    ```
    Returns the optimized parameters and the error for them.
    """
    if minValues is not None and minValues.shape != parameters.shape:
        raise ValueError("'GradientDescent' requires 'minValues' and 'parameters' to be equally sized")
    if maxValues is not None and maxValues.shape != parameters.shape:
        raise ValueError("'GradientDescent' requires 'maxValues' and 'parameters' to be equally sized")
    
    if minValues is None:
        minValues = float("-inf") * np.ones_like(parameters)
    if maxValues is None:
        maxValues = float("inf") * np.ones_like(parameters)

    initFunc(userdata)
    params = parameters.copy()
    _Descent(params, errorFunc, userdata, minValues, maxValues, learningRate, decay, minLearningRate, maxIterations, progress)
    current = _Sigmoid(params, minValues, maxValues)
    return (current, errorFunc(userdata, current))

def _ProgressListener(queue:Queue, progress:Callable[[float], None]):
    while True:
        err, lr = queue.get()
        if err is None:
            break
        progress(err, lr)

def GradientDescentMP(parameters:np.ndarray, errorFunc:Callable[[any, np.ndarray], float], initFunc:Callable[[any],None], userdata, *,
                      minValues:np.ndarray = None, maxValues:np.ndarray = None,
                      learningRate:float = 1e-6, decay:float = 0.8,
                      minLearningRate:float = 1e-64,
                      maxIterations:int = 1_000, numThreads:int = 1, fastSearch:int = 512, diversity:np.ndarray|float = 1.0,
                      progress:Callable[[float], None] = None) -> tuple[np.ndarray, float]:
    """
    Perform Gradient Descent following the formula below:
    ```
    a_i+1            = a_i - learningRate_i * gradient F(a_i), i >= 0
    learningRate_i+1 = learningRate_i * decay^(-sgn(a_i+1 dot a_i))
    ```
    Returns the optimized parameters and the error for them.
    """
    if minValues is not None and minValues.shape != parameters.shape:
        raise ValueError("'GradientDescent' requires 'minValues' and 'parameters' to be equally sized")
    if maxValues is not None and maxValues.shape != parameters.shape:
        raise ValueError("'GradientDescent' requires 'maxValues' and 'parameters' to be equally sized")
    
    if minValues is None:
        minValues = float("-inf") * np.ones_like(parameters)
    if maxValues is None:
        maxValues = float("inf") * np.ones_like(parameters)

    params      = _InvSigmoid(parameters, minValues, maxValues)
    minBounded  = np.isfinite(minValues)
    maxBounded  = np.isfinite(maxValues)
    minValues2  = np.where(minBounded, minValues, 0.0)
    maxValues2  = np.where(maxBounded, maxValues, 0.0)
    boundDeltas = np.where(~minBounded & ~maxBounded, 15.0, (maxValues2 - minValues2)) * diversity
    with Manager() as manager:
        sharedBest           = manager.dict()
        sharedBest["error"]  = float("inf")
        sharedBest["params"] = params
        lock                 = manager.Lock()
        queue                = manager.Queue()

        procArgs:list[list] = []
        for i in range(numThreads):
            rand       = np.random.uniform(-0.5, 0.5, size=params.shape)
            randParams = params + rand * boundDeltas
            procArgs.append((randParams, errorFunc, initFunc, userdata, minValues, maxValues, learningRate, decay, minLearningRate, maxIterations, fastSearch, diversity, sharedBest, lock, queue))

        if progress:
            listener = Thread(target=_ProgressListener, args=(queue, progress), daemon=True)
            listener.start()

        with Pool() as pool:
            pool.starmap(_DescentMP, procArgs)
        
        if progress:
            queue.put((None, None))
            listener.join()

        return _Sigmoid(sharedBest["params"], minValues, maxValues), sharedBest["error"]