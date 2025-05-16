import bisect
import math
import sys
import itertools
import numpy as np

import time as ptime
from collections.abc import Callable

_TIMER_NS:Callable[[], int] = None

def _select_time_source():
	global _TIMER_NS
	timers = [
		(ptime.perf_counter_ns, ptime.get_clock_info("perf_counter")),
		(ptime.process_time_ns, ptime.get_clock_info("process_time")),
		(ptime.thread_time_ns, ptime.get_clock_info("thread_time")),
		(ptime.monotonic_ns, ptime.get_clock_info("monotonic")),
		(ptime.time_ns, ptime.get_clock_info("time"))
	]
	timers = sorted([ timer for timer in timers if timer[1].monotonic ], key=lambda x: x[1].resolution)
	_TIMER_NS = timers[0][0]
if _TIMER_NS is None: _select_time_source()

def time_ns() -> int:
	"""
	Provides a monotonic time value in nanoseconds with the highest resolution.
	"""
	return _TIMER_NS()

def time() -> float:
	"""
	Provides a monotonic time value in seconds with the highest resolution.
	"""
	return _TIMER_NS() * 1e-9

def GetDistribution(fullDistribution:list[float]|tuple[list[list[float]], list[float]], rssis:list[int]) -> list[tuple[float, float]]:
	"""
	Combine the full precomputed distributions based on the set of rssi values perceived.  
	Returns the distribution of distances and their weights.
	"""
	values, counts = np.unique(np.array(rssis), return_counts=True)
	if type(fullDistribution) == tuple:
		distances    = np.array(fullDistribution[0])[-values]
		weights      = counts[:,None] * np.array(fullDistribution[1])[None,:]
		allDistances = np.concatenate(distances)
		allWeights   = np.concatenate(weights)
		
		uniqueDistances, inverseIndices = np.unique(allDistances, return_inverse=True)
		summedWeights = np.bincount(inverseIndices, weights=allWeights)
		return list(zip(uniqueDistances.tolist(), (summedWeights / summedWeights.sum()).tolist()))
	else:
		distances      = np.array(fullDistribution)[-values]
		weights        = counts / len(rssis)
		return list(zip(distances.tolist(), weights.tolist()))

def GetStartIndex(graph:list[tuple[float, float]], sample:int, timeWindow:float) -> int:
	return min(bisect.bisect_left(graph, graph[sample][0] - timeWindow, key=lambda x:x[0]), len(graph) - 1)

# Filters:
def RunSMA(graph:list[tuple[float, float]], sample:int) -> float:
	start        = GetStartIndex(graph, sample, 10.0)
	avg:float    = 0.0
	factor:float = 1.0 / (1 + sample - start)
	for i in range(start, 1 + sample):
		avg += factor * graph[i][1]
	return avg

def RunEMA(graph:list[tuple[float, float]], sample:int) -> float:
	alpha:float = 0.9
	yn:float    = graph[0][1]
	for i in range(1, 1 + sample):
		yn = yn * alpha + graph[i][1] * (1.0 - alpha)
	return yn

def RunTEMA(graph:list[tuple[float, float]], sample:int) -> float:
	yn:float   = graph[0][1]
	beta:float = 0.9
	c:float    = 10.0
	for i in range(1, 1 + sample):
		alpha = beta**((graph[i][0] - graph[i - 1][0]) * c)
		yn    = yn * alpha + graph[i][1] * (1.0 - alpha)
	return yn

def RunSC(graph:list[tuple[float, float]], sample:int) -> float:
	start = GetStartIndex(graph, sample, 10.0)
	scK   = 2.0

	mean:float   = 0.0
	factor:float = 1.0 / (1 + sample - start)
	for i in range(start, 1 + sample):
		mean += factor * graph[i][1]
	stddev:float = 0.0
	for i in range(start, 1 + sample):
		stddev += factor * (graph[i][1] - mean)**2
	stddev = math.sqrt(stddev)

	avg:float = 0.0
	count:int = 0
	for i in range(start, 1 + sample):
		if abs(mean - graph[i][1]) <= scK * stddev:
			avg   += graph[i][1]
			count += 1
	return avg / count if count > 0 else None

def RunOSC(graph:list[tuple[float, float]], sample:int) -> float:
	start = GetStartIndex(graph, sample, 10.0)
	oscK  = 2.0

	mean:float   = 0.0
	factor:float = 1.0 / (1 + sample - start)
	for i in range(start, 1 + sample):
		mean += factor * graph[i][1]
	stddev:float = 0.0
	for i in range(start, 1 + sample):
		stddev += factor * (graph[i][1] - mean)**2
	stddev = math.sqrt(stddev)

	avg:float = 0.0
	count:int = 0
	for i in range(start, 1 + sample):
		if mean - graph[i][1] <= oscK * stddev:
			avg   += graph[i][1]
			count += 1
	return avg / count if count > 0 else None

def RunKalman(graph:list[tuple[float, float]], sample:int) -> float:
	processNoise:float    = 0.1
	measuredNoise:float   = 3.0
	processScaling:float  = 0.05
	estimate:float        = graph[0][1]
	errorCovariance:float = 1.0

	for i in range(1, 1 + sample):
		estimatePrev        = estimate
		errorCovariancePrev = errorCovariance + processNoise

		kalmanGain      = errorCovariancePrev / (errorCovariancePrev + measuredNoise)
		estimate        = estimatePrev + kalmanGain * (graph[i][1] - estimatePrev)
		errorCovariance = (1.0 - kalmanGain) * errorCovariancePrev

		processNoise = processScaling * abs(estimate - estimatePrev)

	return estimate

def RunAB(graph:list[tuple[float, float]], sample:int) -> float:
	alpha:float  = 0.85
	beta:float   = 0.005
	prevEstimate = graph[0][1]
	prevRate     = 0.0

	for i in range(1, 1 + sample):
		dt = graph[i][0] - graph[i - 1][0]
		
		estimate = prevEstimate + prevRate * dt

		r = graph[i][1] - estimate

		prevEstimate = estimate + alpha * r
		prevRate    += (beta * r) / dt
	return prevEstimate

# Filters Vectorized:
def RunVectorizedSMA(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	times  = np.array([ sample[0] for sample in graph ])
	values = np.array([ sample[1] for sample in graph ])
	n      = len(values)

	starts = np.searchsorted(times, times - 10.0, side="left")
	cumsum = np.cumsum(np.insert(values, 0, 0.0))

	sums   = cumsum[1:] - cumsum[starts]
	counts = np.arange(1, n + 1) - starts
	sma    = sums / counts

	return list(zip(times.tolist(), sma.tolist()))

def RunVectorizedEMA(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	# Not vectorizable
	alpha:float = 0.9
	
	alpha2 = 1.0 - alpha
	
	results = [ graph[0] ]
	for i in range(1, len(graph)):
		value = alpha * results[i - 1][1] + alpha2 * graph[i][1]
		results.append((graph[i][0], value))
	return results

def RunVectorizedTEMA(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	# Not vectorizable
	beta:float = 0.9
	c:float    = 10.0

	times  = np.array([ sample[0] for sample in graph ])
	values = np.array([ sample[1] for sample in graph[1:] ])
	dt     = times[1:] - times[:-1]
	alpha  = beta**(dt * c)
	
	weightedValues = (1.0 - alpha) * values

	results = [ graph[0] ]
	for i in range(0, len(graph) - 1):
		value = alpha[i] * results[i][1] + weightedValues[i]
		results.append((graph[i + 1][0], float(value)))
	return results

def RunVectorizedSC(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	scK = 2.0

	times  = np.array([ sample[0] for sample in graph ])
	values = np.array([ sample[1] for sample in graph ])
	n      = len(values)

	starts  = np.searchsorted(times, times - 10.0, side="left")
	cumsum  = np.cumsum(np.insert(values, 0, 0.0))
	cumsum2 = np.cumsum(np.insert(values**2, 0, 0.0))

	sums      = cumsum[1:] - cumsum[starts]
	sums2     = cumsum2[1:] - cumsum2[starts]
	counts    = np.arange(1, n + 1) - starts
	means     = sums / counts
	variances = (sums2 / counts) - (means**2)
	stddevs   = np.sqrt(np.maximum(variances, 0.0))

	m = counts.max()
	
	windowValues = np.full((n, m), np.nan)
	windowTimes  = np.full((n, m), np.nan)
	for i in range(n):
		start = starts[i]
		count = counts[i]
		windowValues[i,:count] = values[start:i+1]
		windowTimes[i,:count] = times[start:i+1]
	
	absDiff    = np.abs(windowValues - means[:, None])
	inlierMask = absDiff <= (scK * stddevs[:, None])

	maskedVals   = np.where(inlierMask, windowValues, 0.0)
	inlierCounts = inlierMask.sum(axis=1)
	inlierSums   = maskedVals.sum(axis=1)

	result = np.divide(inlierSums, inlierCounts, out=np.full_like(inlierSums, np.nan), where=inlierCounts > 0)
	
	return list(zip(times.tolist(), result.tolist()))

def RunVectorizedOSC(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	oscK = 2.0

	times  = np.array([ sample[0] for sample in graph ])
	values = np.array([ sample[1] for sample in graph ])
	n      = len(values)

	starts  = np.searchsorted(times, times - 10.0, side="left")
	cumsum  = np.cumsum(np.insert(values, 0, 0.0))
	cumsum2 = np.cumsum(np.insert(values**2, 0, 0.0))

	sums      = cumsum[1:] - cumsum[starts]
	sums2     = cumsum2[1:] - cumsum2[starts]
	counts    = np.arange(1, n + 1) - starts
	means     = sums / counts
	variances = (sums2 / counts) - (means**2)
	stddevs   = np.sqrt(np.maximum(variances, 0.0))

	m = counts.max()
	
	windowValues = np.full((n, m), np.nan)
	windowTimes  = np.full((n, m), np.nan)
	for i in range(n):
		start = starts[i]
		count = counts[i]
		windowValues[i,:count] = values[start:i+1]
		windowTimes[i,:count] = times[start:i+1]
	
	diff       = means[:, None] - windowValues
	inlierMask = diff <= (oscK * stddevs[:, None])

	maskedVals   = np.where(inlierMask, windowValues, 0.0)
	inlierCounts = inlierMask.sum(axis=1)
	inlierSums   = maskedVals.sum(axis=1)

	result = np.divide(inlierSums, inlierCounts, out=np.full_like(inlierSums, np.nan), where=inlierCounts > 0)

	return list(zip(times.tolist(), result.tolist()))

def RunVectorizedKalman(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	# Not Vectorizable
	processNoise:float    = 0.1
	measuredNoise:float   = 3.0
	processScaling:float  = 0.05
	errorCovariance:float = 1.0

	outGraph = []
	outGraph.append(graph[0])
	for i in range(1, len(graph)):
		estimatePrev        = outGraph[i - 1][1]
		errorCovariancePrev = errorCovariance + processNoise

		kalmanGain      = errorCovariancePrev / (errorCovariancePrev + measuredNoise)
		estimate        = estimatePrev + kalmanGain * (graph[i][1] - estimatePrev)
		errorCovariance = (1.0 - kalmanGain) * errorCovariancePrev

		processNoise = processScaling * abs(estimate - estimatePrev)
		outGraph.append((graph[i][0], float(estimate)))
	return outGraph

def RunVectorizedAB(graph:list[tuple[float, float]]) -> list[tuple[float, float]]:
	# Not Vectorizable
	alpha:float = 0.85
	beta:float  = 0.005
	times       = np.array([ sample[0] for sample in graph ])
	dt          = times[1:] - times[:-1]
	prevRate    = 0.0

	outGraph = []
	outGraph.append(graph[0])
	for i in range(1, len(graph)):
		estimate = outGraph[i - 1][1] + prevRate * dt[i - 1]
		r        = graph[i][1] - estimate

		estimate += alpha * r
		prevRate += (beta * r) / dt[i - 1]

		outGraph.append((graph[i][0], float(estimate)))
	return outGraph

filters = [
	("SMA", RunSMA, RunVectorizedSMA),
	("EMA", RunEMA, RunVectorizedEMA),
	("TEMA", RunTEMA, RunVectorizedTEMA),
	("SC", RunSC, RunVectorizedSC),
	("OSC", RunOSC, RunVectorizedOSC),
	("Kalman", RunKalman, RunVectorizedKalman),
	("AB", RunAB, RunVectorizedAB)
]

# Distance Estimators:
def RunEmpirical(rssi:float, rssi1m:float, rssiK:float, gain:float) -> float:
	return 10**((rssi1m - rssi) / (10.0 * rssiK))

def RunLDPL(rssi:float, rssi1m:float, rssiK:float, gain:float) -> tuple[float, float, float, float]:
	lowerDist = 10**((rssi1m - rssi - 0.8) / (10.0 * rssiK))
	midDist   = 10**((rssi1m - rssi) / (10.0 * rssiK))
	upperDist = 10**((rssi1m - rssi + 0.8) / (10.0 * rssiK))
	randDist  = 10**((rssi1m - rssi + np.random.normal(loc=0.0, scale=0.8)) / (10.0 * rssiK))
	return (lowerDist, midDist, upperDist, randDist)

def RunFSPL(rssi:float, rssi1m:float, rssiK:float, gain:float) -> float:
	return 10**((4.0 + gain - rssi - 20 * math.log10(2.402) - 32.4477832219) / 20)

def RunFriis(rssi:float, rssi1m:float, rssiK:float, gain:float) -> float:
	return 0.023856725796 / 2.402 * 10**((4.0 + gain - rssi) / 20)

def RunITUIPM(rssi:float, rssi1m:float, rssiK:float, gain:float) -> float:
	return 10**((rssi1m - rssi - 20 * math.log10(2402) + 28) / 30)

# Distance Estimators Vectorized base:
def RunBaseEmpirical(rssis:np.ndarray, rssi1m:float, rssiK:float) -> np.ndarray:
	f = 1.0 / (10.0 * rssiK)
	return 10**((rssi1m - rssis) * f)

def RunBaseLDPL(rssis:np.ndarray, dev:np.ndarray, rssi1m:float, rssiK:float) -> np.ndarray:
	f = 1.0 / (10.0 * rssiK)
	return 10**((rssi1m - rssis + dev) * f)

def RunBaseFSPL(rssis:np.ndarray, gain:float) -> np.ndarray:
	Tx   = 4.0
	freq = 2.402
	pl   = Tx + gain - 20 * math.log10(freq) - 32.4477832219
	return 10**((pl - rssis) / 20)

def RunBaseFriis(rssis:np.ndarray, gain:float) -> np.ndarray:
	Tx    = 4.0
	freq  = 2.402
	const = 0.02385672579618471129444449166887 # c / (4 * math.pi * 10^9)
	f     = const / freq
	pl    = Tx + gain
	return f * 10**((pl - rssis) / 20)

def RunBaseITUIPM(rssis:np.ndarray, rssi1m:float) -> np.ndarray:
	freq = 2.402
	pl   = rssi1m - 20 * math.log10(freq * 1000) + 28
	return 10**((pl - rssis) / 30)

# Distance Estimators Vectorized:
def RunVectorizedEmpirical(graph:list[tuple[float, float]], rssi1m:float, rssiK:float, gain:float) -> list[tuple[float, float]]:
	times   = np.array([ sample[0] for sample in graph ])
	values  = np.array([ sample[1] for sample in graph ])
	results = RunBaseEmpirical(values, rssi1m, rssiK)
	return list(zip(times.tolist(), results.tolist()))

def RunVectorizedLDPL(graph:list[tuple[float, float]], rssi1m:float, rssiK:float, gain:float) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]], list[tuple[float, float]]]:
	rssiD      = 0.8
	times      = np.array([ sample[0] for sample in graph ])
	values     = np.array([ sample[1] for sample in graph ])
	lowerDists = RunBaseLDPL(values, -rssiD, rssi1m, rssiK)
	midDists   = RunBaseLDPL(values, 0.0, rssi1m, rssiK)
	upperDists = RunBaseLDPL(values, rssiD, rssi1m, rssiK)
	randDists  = RunBaseLDPL(values, np.random.normal(loc=0.0, scale=rssiD, size=values.size), rssi1m, rssiK)

	outLowerGraph = list(zip(times.tolist(), lowerDists.tolist()))
	outMidGraph   = list(zip(times.tolist(), midDists.tolist()))
	outUpperGraph = list(zip(times.tolist(), upperDists.tolist()))
	outRandGraph  = list(zip(times.tolist(), randDists.tolist()))
	return (outLowerGraph, outMidGraph, outUpperGraph, outRandGraph)

def RunVectorizedFSPL(graph:list[tuple[float, float]], rssi1m:float, rssiK:float, gain:float) -> list[tuple[float, float]]:
	times   = np.array([ sample[0] for sample in graph ])
	values  = np.array([ sample[1] for sample in graph ])
	results = RunBaseFSPL(values, gain)
	return list(zip(times.tolist(), results.tolist()))

def RunVectorizedFriis(graph:list[tuple[float, float]], rssi1m:float, rssiK:float, gain:float) -> list[tuple[float, float]]:
	times   = np.array([ sample[0] for sample in graph ])
	values  = np.array([ sample[1] for sample in graph ])
	results = RunBaseFriis(values, gain)
	return list(zip(times.tolist(), results.tolist()))

def RunVectorizedITUIPM(graph:list[tuple[float, float]], rssi1m:float, rssiK:float, gain:float) -> list[tuple[float, float]]:
	times   = np.array([ sample[0] for sample in graph ])
	values  = np.array([ sample[1] for sample in graph ])
	results = RunBaseITUIPM(values, rssi1m)
	return list(zip(times.tolist(), results.tolist()))

# Distance Estimators Distribution:
def RunDistributionEmpirical(rssi1m:float, rssiK:float, gain:float) -> list[float]:
	return RunBaseEmpirical(-np.arange(128), rssi1m, rssiK).tolist()

def RunDistributionLDPL(rssi1m:float, rssiK:float, gain:float) -> tuple[list[list[float]], list[float]]:
	rssiD    = 0.8
	offsets  = np.arange(-30, 31) * 0.1 * rssiD # -3 to 3
	gaussian = np.exp(-0.5 * (offsets / rssiD)**2) / (rssiD * np.sqrt(2 * np.pi))
	dists    = RunBaseLDPL(-np.arange(128)[:,None], offsets[None,:], rssi1m, rssiK)
	return (dists.tolist(), gaussian.tolist())

def RunDistributionFSPL(rssi1m:float, rssiK:float, gain:float) -> list[float]:
	return RunBaseFSPL(-np.arange(128), gain).tolist()

def RunDistributionFriis(rssi1m:float, rssiK:float, gain:float) -> list[float]:
	return RunBaseFriis(-np.arange(128), gain).tolist()

def RunDistributionITUIPM(rssi1m:float, rssiK:float, gain:float) -> list[float]:
	return RunBaseITUIPM(-np.arange(128), rssi1m).tolist()

distanceEstimators = [
	("Empirical", RunEmpirical, RunVectorizedEmpirical, RunDistributionEmpirical, False),
	("LDPL", RunLDPL, RunVectorizedLDPL, RunDistributionLDPL, True),
	("FSPL", RunFSPL, RunVectorizedFSPL, RunDistributionFSPL, False),
	("Friis", RunFriis, RunVectorizedFriis, RunDistributionFriis, False),
	("ITUIPM", RunITUIPM, RunVectorizedITUIPM, RunDistributionITUIPM, False)
]

# Position Estimators:
def RunBL(distances:list[float], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> tuple[float, float]|None:
	if len(distances) < 2:
		return None
	
	positions:list[tuple[float, float]] = []
	errors:list[float]                  = []
	for i, j in itertools.combinations(range(len(distances)), 2):
		posA  = (posXs[i], posYs[i])
		posB  = (posXs[j], posYs[j])
		normA = (normXs[i], normYs[i])
		normB = (normXs[j], normYs[j])
		distA = distances[i]
		distB = distances[j]

		d  = math.sqrt((posB[0] - posA[0])**2 + (posB[1] - posA[1])**2)
		dd = max(d, 1e-9)
		dx = (posB[0] - posA[0]) / dd
		dy = (posB[1] - posA[1]) / dd

		a = (distA**2 - distB**2 + dd**2) / (2.0 * dd)
		h = math.sqrt(max(distA**2 - a**2, 0.0))

		Px = posA[0] + a * dx
		Py = posA[1] + a * dy
		fx = h * dy
		fy = h * dx
		P1 = (Px + fx, Py - fy)
		P2 = (Px - fx, Py + fy)

		avgNormX = 0.5 * (normA[0] + normB[0])
		avgNormY = 0.5 * (normA[1] + normB[1])
		dot      = P1[0] * avgNormX + P1[1] * avgNormY
		bestPos  = P1 if dot >= 0.0 else P2

		valid = (d != 0.0) and (h > 0.0)
		if not valid:
			totalErr = float("inf")
		else:
			totalErr = 0.0
			for k in range(len(distances)):
				dist      = math.sqrt((bestPos[0] - posXs[k])**2 + (bestPos[1] - posYs[k])**2)
				totalErr += abs(dist - distances[k])
		positions.append(bestPos)
		errors.append(totalErr)

	minIdx = 0
	minErr = errors[0]
	for i in range(1, len(errors)):
		if errors[i] < minErr:
			minIdx = i
			minErr = errors[i]
	return positions[minIdx] if minErr != float("inf") else None

def RunTLG(distances:list[float], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> tuple[float, float]|None:
	if len(distances) < 3:
		return None
	
	positions:list[tuple[float, float]] = []
	errors:list[float]                  = []
	for i, j, k in itertools.combinations(range(len(distances)), 3):
		posA  = (posXs[i], posYs[i])
		posB  = (posXs[j], posYs[j])
		posC  = (posXs[k], posYs[k])
		distA = distances[i]
		distB = distances[j]
		distC = distances[k]
		
		xba = posB[0] - posA[0]
		yba = posB[1] - posA[1]
		xca = posC[0] - posA[0]
		yca = posC[1] - posA[1]

		lba = math.sqrt(xba**2 + yba**2)
		lca = math.sqrt(xca**2 + yca**2)

		xba = xba / max(lba, 1e-9)
		yba = yba / max(lba, 1e-9)
		xca = xca / max(lca, 1e-9)
		yca = yca / max(lca, 1e-9)

		P1x = 0.5 * (posA[0] + posB[0] + xba * (distA - distB))
		P1y = 0.5 * (posA[1] + posB[1] + yba * (distA - distB))
		n1x = -yba
		n1y = xba
		
		P2x = 0.5 * (posA[0] + posC[0] + xca * (distA - distC))
		P2y = 0.5 * (posA[1] + posC[1] + yca * (distA - distC))
		n2x = -yca
		n2y = xca

		div = n2y - n1y / n1x * n2x
		t   = (P1y - P2y + n1y / n1x * (P2x - P1x)) / max(div, 1e-9)

		pos = (P2x + n2x * t, P2y + n2y * t)

		valid = div != 0.0
		if not valid:
			totalErr = float("inf")
		else:
			totalErr = 0.0
			for k in range(len(distances)):
				dist      = math.sqrt((pos[0] - posXs[k])**2 + (pos[1] - posYs[k])**2)
				totalErr += abs(dist - distances[k])
		positions.append(pos)
		errors.append(totalErr)

	minIdx = 0
	minErr = errors[0]
	for i in range(1, len(errors)):
		if errors[i] < minErr:
			minIdx = i
			minErr = errors[i]
	return positions[minIdx] if minErr != float("inf") else None

def RunTLMT(distances:list[float], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> tuple[float, float]|None:
	if len(distances) < 3:
		return None
	
	positions:list[tuple[float, float]] = []
	errors:list[float]                  = []
	for i, j, k in itertools.combinations(range(len(distances)), 3):
		posA  = (posXs[i], posYs[i])
		posB  = (posXs[j], posYs[j])
		posC  = (posXs[k], posYs[k])
		distA = distances[i]
		distB = distances[j]
		distC = distances[k]

		adota = posA[0]**2 + posA[1]**2
		bdotb = posB[0]**2 + posB[1]**2
		cdotc = posC[0]**2 + posC[1]**2
		A1    = 2 * (posB[0] - posA[0])
		A2    = 2 * (posC[0] - posA[0])
		A3    = 2 * (posC[0] - posB[0])
		B1    = 2 * (posB[1] - posA[1])
		B2    = 2 * (posC[1] - posA[1])
		B3    = 2 * (posC[1] - posB[1])
		C1    = distA**2 - distB**2 + bdotb - adota
		C2    = distA**2 - distC**2 + cdotc - adota
		C3    = distB**2 - distC**2 + cdotc - bdotb

		a1 = A1**2 + A2**2 + A3**2
		a2 = A1 * B1 + A2 * B2 + A3 * B3
		a3 = B1**2 + B2**2 + B3**2

		det = a1 * a3 - a2**2

		c1 = a3 / max(det, 1e-9)
		c2 = -a2 / max(det, 1e-9)
		c3 = a1 / max(det, 1e-9)

		b1 = A1 * C1 + A2 * C2 + A3 * C3
		b2 = B1 * C1 + B2 * C2 + B3 * C3

		pos = (c1 * b1 + c2 * b2, c2 * b1 + c3 * b2)

		valid = det != 0.0
		if not valid:
			totalErr = float("inf")
		else:
			totalErr = 0.0
			for k in range(len(distances)):
				dist      = math.sqrt((pos[0] - posXs[k])**2 + (pos[1] - posYs[k])**2)
				totalErr += abs(dist - distances[k])
		positions.append(pos)
		errors.append(totalErr)

	minIdx = 0
	minErr = errors[0]
	for i in range(1, len(errors)):
		if errors[i] < minErr:
			minIdx = i
			minErr = errors[i]
	return positions[minIdx] if minErr != float("inf") else None

def RunTLMD(distances:list[float], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> tuple[float, float]|None:
	if len(distances) < 3:
		return None
	
	positions:list[tuple[float, float]] = []
	errors:list[float]                  = []
	for i, j, k in itertools.combinations(range(len(distances)), 3):
		posA  = (posXs[i], posYs[i])
		posB  = (posXs[j], posYs[j])
		posC  = (posXs[k], posYs[k])
		distA = distances[i]
		distB = distances[j]
		distC = distances[k]

		a1 = 2 * (posB[0] - posA[0])
		a2 = 2 * (posB[1] - posA[1])
		a3 = 2 * (posC[0] - posA[0])
		a4 = 2 * (posC[1] - posA[1])
		ad = a1 * a4 - a2 * a3

		adota = posA[0]**2 + posA[1]**2
		bdotb = posB[0]**2 + posB[1]**2
		cdotc = posC[0]**2 + posC[1]**2
		b1    = distA**2 - distB**2 + bdotb - adota
		b2    = 2 * (posB[1] - posA[1])
		b3    = distA**2 - distC**2 + cdotc - adota
		b4    = 2 * (posC[1] - posA[1])
		bd    = b1 * b4 - b2 * b3

		c1    = 2 * (posB[1] - posA[1])
		c2    = distA**2 - distB**2 + bdotb - adota
		c3    = 2 * (posC[1] - posA[1])
		c4    = distA**2 - distC**2 + cdotc - adota
		cd    = c1 * c4 - c2 * c3

		pos = (bd / max(ad, 1e-9), cd / max(ad, 1e-9))

		valid = ad != 0.0
		if not valid:
			totalErr = float("inf")
		else:
			totalErr = 0.0
			for k in range(len(distances)):
				dist      = math.sqrt((pos[0] - posXs[k])**2 + (pos[1] - posYs[k])**2)
				totalErr += abs(dist - distances[k])
		positions.append(pos)
		errors.append(totalErr)

	minIdx = 0
	minErr = errors[0]
	for i in range(1, len(errors)):
		if errors[i] < minErr:
			minIdx = i
			minErr = errors[i]
	return positions[minIdx] if minErr != float("inf") else None

def RunTLSE(distances:list[float], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> tuple[float, float]|None:
	if len(distances) < 3:
		return None
	
	positions:list[tuple[float, float]] = []
	errors:list[float]                  = []
	for i, j, k in itertools.combinations(range(len(distances)), 3):
		posA  = (posXs[i], posYs[i])
		posB  = (posXs[j], posYs[j])
		posC  = (posXs[k], posYs[k])
		distA = distances[i]
		distB = distances[j]
		distC = distances[k]

		adota = posA[0]**2 + posA[1]**2
		bdotb = posB[0]**2 + posB[1]**2
		cdotc = posC[0]**2 + posC[1]**2
		A = 2 * (posB[0] - posA[0])
		B = 2 * (posB[1] - posA[1])
		C = distA**2 - distB**2 + bdotb - adota
		D = 2 * (posC[0] - posB[0])
		E = 2 * (posC[1] - posB[1])
		F = distB**2 - distC**2 + cdotc - bdotb

		d1 = E * A
		d2 = B * D
		a  = C * E - F * B

		pos = (a / max(d1 - d2, 1e-9), a / max(d2 - d1, 1e-9))

		valid = d1 != d2
		if not valid:
			totalErr = float("inf")
		else:
			totalErr = 0.0
			for k in range(len(distances)):
				dist      = math.sqrt((pos[0] - posXs[k])**2 + (pos[1] - posYs[k])**2)
				totalErr += abs(dist - distances[k])
		positions.append(pos)
		errors.append(totalErr)

	minIdx = 0
	minErr = errors[0]
	for i in range(1, len(errors)):
		if errors[i] < minErr:
			minIdx = i
			minErr = errors[i]
	return positions[minIdx] if minErr != float("inf") else None

# Distance Estimators Vectorized base:
def RunBaseBL(distances:np.ndarray, devPos:np.ndarray, devNorm:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	pairs = list(itertools.combinations(range(len(distances)), 2))
	Iidx  = np.array([ i for i,j in pairs ]) # (P,)
	Jidx  = np.array([ j for i,j in pairs ]) # (P,)
	distA = distances[Iidx]                  # (P,T)
	distB = distances[Jidx]                  # (P,T)
	posA  = devPos[Iidx]                     # (P,2)
	posB  = devPos[Jidx]                     # (P,2)
	normA = devNorm[Iidx]                    # (P,2)
	normB = devNorm[Jidx]                    # (P,2)

	dVec = posB - posA                    # (P,2)
	d    = np.linalg.norm(dVec, axis=1)   # (P,)
	dd   = np.clip(d, min=1e-9, max=None) # (P,)
	dxdy = dVec / dd[:,None]              # (P,2)

	a = (distA**2 - distB**2 + (dd**2)[:,None]) / (2 * dd[:,None]) # (P,T)
	h = np.sqrt(np.clip(distA**2 - a**2, min=0.0, max=None))       # (P,T)

	dx = dxdy[:,0] # (P,)
	dy = dxdy[:,1] # (P,)

	Px = posA[:,0][:,None] + a * dx[:,None]      # (P,T)
	Py = posA[:,1][:,None] + a * dy[:,None]      # (P,T)
	fx = h * dy[:,None]                          # (P,T)
	fy = h * dx[:,None]                          # (P,T)
	P1 = np.stack([ Px + fx, Py - fy ], axis=-1) # (P,T,2)
	P2 = np.stack([ Px - fx, Py + fy ], axis=-1) # (P,T,2)

	avgNorm = 0.5 * (normA + normB)[:,None,:]      # (P,1,2)
	dot     = np.sum(P1 * avgNorm, axis=-1)        # (P,T)
	bestPos = np.where(dot[...,None] >= 0, P1, P2) # (P,T,2)

	valid    = (d != 0)[:,None] & (h > 0)                                         # (P,T)
	dists    = np.linalg.norm(bestPos[None,...] - devPos[:,None,None,:], axis=-1) # (N,P,T)
	errs     = np.abs(dists - distances[:,None,:])                                # (N,P,T)
	totalErr = np.where(valid, np.sum(errs, axis=0), np.inf)                      # (P,T)

	bestPairIdx  = np.argmin(totalErr, axis=0)               # (T,)
	timeIdx      = np.arange(bestPairIdx.shape[0])           # (T,)
	bestTotalErr = totalErr[bestPairIdx,timeIdx]             # (T,)
	validMask    = np.isfinite(bestTotalErr)                 # (T,) 
	finalPos     = bestPos[bestPairIdx,timeIdx,:][validMask] # (U,2)
	return (finalPos, validMask)

def RunBaseTLG(distances:np.ndarray, devPos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	triples = list(itertools.combinations(range(len(distances)), 3))
	Iidx    = np.array([ i for i,j,k in triples ]) # (P,)
	Jidx    = np.array([ j for i,j,k in triples ]) # (P,)
	Kidx    = np.array([ k for i,j,k in triples ]) # (P,)
	distA   = distances[Iidx]                      # (P,T)
	distB   = distances[Jidx]                      # (P,T)
	distC   = distances[Kidx]                      # (P,T)
	posA    = devPos[Iidx]                         # (P,2)
	posB    = devPos[Jidx]                         # (P,2)
	posC    = devPos[Kidx]                         # (P,2)

	ba = posB - posA                  # (P,2)
	ca = posC - posA                  # (P,2)
	lba = np.linalg.norm(ba, axis=-1) # (P,)
	lca = np.linalg.norm(ca, axis=-1) # (P,)

	ba = ba / np.clip(lba, min=1e-9, max=None)[:,None] # (P,2)
	ca = ca / np.clip(lca, min=1e-9, max=None)[:,None] # (P,2)

	P1 = 0.5 * ((posA + posB)[:,None,:] + ba[:,None,:] * (distA - distB)[:,:,None]) # (P,T,2)
	P2 = 0.5 * ((posA + posC)[:,None,:] + ca[:,None,:] * (distA - distC)[:,:,None]) # (P,T,2)
	n1 = np.stack((-ba[:,1], ba[:,0]), axis=-1)                                     # (P,2)
	n2 = np.stack((-ca[:,1], ca[:,0]), axis=-1)                                     # (P,2)

	div = n2[:,1] - n1[:,1] / n1[:,0] * n2[:,0]                                                                                      # (P,)
	t   = (P1[:,:,1] - P2[:,:,1] + (n1[:,1] / n1[:,0])[:,None] * (P2[:,:,0] - P1[:,:,0])) / np.clip(div, min=1e-9, max=None)[:,None] # (P,T)

	pos = P2 + n2[:,None,:] * t[...,None] # (P,T,2)

	valid    = div != 0.0                                                     # (P,)
	dists    = np.linalg.norm(pos[None,...] - devPos[:,None,None,:], axis=-1) # (N,P,T)
	errs     = np.abs(dists - distances[:,None,:])                            # (N,P,T)
	totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)          # (P,T)

	bestPairIdx  = np.argmin(totalErr, axis=0)           # (T,)
	timeIdx      = np.arange(bestPairIdx.shape[0])       # (T,)
	bestTotalErr = totalErr[bestPairIdx,timeIdx]         # (T,)
	validMask    = np.isfinite(bestTotalErr)             # (T,) 
	finalPos     = pos[bestPairIdx,timeIdx,:][validMask] # (U,2)
	return (finalPos, validMask)

def RunBaseTLMT(distances:np.ndarray, devPos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	triples = list(itertools.combinations(range(len(distances)), 3))
	Iidx    = np.array([ i for i,j,k in triples ]) # (P,)
	Jidx    = np.array([ j for i,j,k in triples ]) # (P,)
	Kidx    = np.array([ k for i,j,k in triples ]) # (P,)
	distA   = distances[Iidx]                      # (P,T)
	distB   = distances[Jidx]                      # (P,T)
	distC   = distances[Kidx]                      # (P,T)
	posA    = devPos[Iidx]                         # (P,2)
	posB    = devPos[Jidx]                         # (P,2)
	posC    = devPos[Kidx]                         # (P,2)

	adota = np.sum(posA * posA, axis=-1)                  # (P,)
	bdotb = np.sum(posB * posB, axis=-1)                  # (P,)
	cdotc = np.sum(posC * posC, axis=-1)                  # (P,)
	AB1   = 2 * (posB - posA)                             # (P,2)
	AB2   = 2 * (posC - posA)                             # (P,2)
	AB3   = 2 * (posC - posB)                             # (P,2)
	C1    = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
	C2    = distA**2 - distC**2 + (cdotc - adota)[:,None] # (P,T)
	C3    = distB**2 - distC**2 + (cdotc - bdotb)[:,None] # (P,T)

	a13 = AB1**2 + AB2**2 + AB3**2                                        # (P,2)
	a2  = AB1[:,0] * AB1[:,1] + AB2[:,0] * AB2[:,1] + AB3[:,0] * AB3[:,1] # (P,)
	
	det = a13[:,0] * a13[:,1] - a2**2 # (P,)
	
	c31 = a13 / np.clip(det, min=1e-9, max=None)[:,None] # (P,2)
	c2  = -a2 / np.clip(det, min=1e-9, max=None)         # (P,)

	b = AB1[:,None,:] * C1[...,None] + AB2[:,None,:] * C2[...,None] + AB3[:,None,:] * C3[...,None] # (P,T,2)

	pos = np.stack((c31[:,None,1] * b[:,:,0] + c2[:,None] * b[:,:,1], c2[:,None] * b[:,:,0] + c31[:,None,0] * b[:,:,1]), axis=-1) # (P,T,2)
	
	valid    = det != 0.0                                                     # (P,)
	dists    = np.linalg.norm(pos[None,...] - devPos[:,None,None,:], axis=-1) # (N,P,T)
	errs     = np.abs(dists - distances[:,None,:])                            # (N,P,T)
	totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)          # (P,T)

	bestPairIdx  = np.argmin(totalErr, axis=0)           # (T,)
	timeIdx      = np.arange(bestPairIdx.shape[0])       # (T,)
	bestTotalErr = totalErr[bestPairIdx,timeIdx]         # (T,)
	validMask    = np.isfinite(bestTotalErr)             # (T,) 
	finalPos     = pos[bestPairIdx,timeIdx,:][validMask] # (U,2)
	return (finalPos, validMask)

def RunBaseTLMD(distances:np.ndarray, devPos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	triples = list(itertools.combinations(range(len(distances)), 3))
	Iidx    = np.array([ i for i,j,k in triples ]) # (P,)
	Jidx    = np.array([ j for i,j,k in triples ]) # (P,)
	Kidx    = np.array([ k for i,j,k in triples ]) # (P,)
	distA   = distances[Iidx]                      # (P,T)
	distB   = distances[Jidx]                      # (P,T)
	distC   = distances[Kidx]                      # (P,T)
	posA    = devPos[Iidx]                         # (P,2)
	posB    = devPos[Jidx]                         # (P,2)
	posC    = devPos[Kidx]                         # (P,2)

	a12 = 2 * (posB - posA)                         # (P,2)
	a34 = 2 * (posC - posA)                         # (P,2)
	ad  = a12[:,0] * a34[:,1] - a12[:,1] * a34[:,0] # (P,)

	adota = np.sum(posA * posA, axis=-1)                  # (P,)
	bdotb = np.sum(posB * posB, axis=-1)                  # (P,)
	cdotc = np.sum(posC * posC, axis=-1)                  # (P,)
	b1    = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
	b2    = 2 * (posB[:,1] - posA[:,1])                   # (P,)
	b3    = distA**2 - distC**2 + (cdotc - adota)[:,None] # (P,T)
	b4    = 2 * (posC[:,1] - posA[:,1])                   # (P,)
	bd    = b1 * b4[:,None] - b2[:,None] * b3             # (P,T)

	c1    = 2 * (posB[:,1] - posA[:,1])                   # (P,)
	c2    = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
	c3    = 2 * (posC[:,1] - posA[:,1])                   # (P,)
	c4    = distA**2 - distC**2 + (cdotc - adota)[:,None] # (P,T)
	cd    = c1[:,None] * c4 - c2 * c3[:,None]             # (P,T)

	pos = np.stack((bd, cd), axis=-1) / np.clip(ad, min=1e-9, max=None)[:,None,None] # (P,T,2)
	
	valid    = ad != 0.0                                                      # (P,)
	dists    = np.linalg.norm(pos[None,...] - devPos[:,None,None,:], axis=-1) # (N,P,T)
	errs     = np.abs(dists - distances[:,None,:])                            # (N,P,T)
	totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)          # (P,T)

	bestPairIdx  = np.argmin(totalErr, axis=0)           # (T,)
	timeIdx      = np.arange(bestPairIdx.shape[0])       # (T,)
	bestTotalErr = totalErr[bestPairIdx,timeIdx]         # (T,)
	validMask    = np.isfinite(bestTotalErr)             # (T,) 
	finalPos     = pos[bestPairIdx,timeIdx,:][validMask] # (U,2)
	return (finalPos, validMask)

def RunBaseTLSE(distances:np.ndarray, devPos:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	triples = list(itertools.combinations(range(len(distances)), 3))
	Iidx    = np.array([ i for i,j,k in triples ]) # (P,)
	Jidx    = np.array([ j for i,j,k in triples ]) # (P,)
	Kidx    = np.array([ k for i,j,k in triples ]) # (P,)
	distA   = distances[Iidx]                      # (P,T)
	distB   = distances[Jidx]                      # (P,T)
	distC   = distances[Kidx]                      # (P,T)
	posA    = devPos[Iidx]                         # (P,2)
	posB    = devPos[Jidx]                         # (P,2)
	posC    = devPos[Kidx]                         # (P,2)

	adota = np.sum(posA * posA, axis=-1)                  # (P,)
	bdotb = np.sum(posB * posB, axis=-1)                  # (P,)
	cdotc = np.sum(posC * posC, axis=-1)                  # (P,)
	AB    = 2 * (posB - posA)                             # (P,2)
	C     = distA**2 - distB**2 + (bdotb - adota)[:,None] # (P,T)
	DE    = 2 * (posC - posB)                             # (P,2)
	F     = distB**2 - distC**2 + (cdotc - bdotb)[:,None] # (P,T)

	d1 = DE[:,1] * AB[:,0]                         # (P,)
	d2 = AB[:,1] * DE[:,0]                         # (P,)
	a  = C * DE[:,1][:,None] - F * AB[:,1][:,None] # (P,T)

	div = np.stack((d1 - d2, d2 - d1), axis=-1) # (P,2)

	pos = np.stack((a, a), axis=-1) / np.clip(div, min=1e-9, max=None)[:,None,:] # (P,T,2)

	valid    = d1 != d2                                                       # (P,)
	dists    = np.linalg.norm(pos[None,...] - devPos[:,None,None,:], axis=-1) # (N,P,T)
	errs     = np.abs(dists - distances[:,None,:])                            # (N,P,T)
	totalErr = np.where(valid[:,None], np.sum(errs, axis=0), np.inf)          # (P,T)

	bestPairIdx  = np.argmin(totalErr, axis=0)           # (T,)
	timeIdx      = np.arange(bestPairIdx.shape[0])       # (T,)
	bestTotalErr = totalErr[bestPairIdx,timeIdx]         # (T,)
	validMask    = np.isfinite(bestTotalErr)             # (T,) 
	finalPos     = pos[bestPairIdx,timeIdx,:][validMask] # (U,2)
	return (finalPos, validMask)

# Position Estimators Vectorized:
def VectorizedHeader(distances:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	devPos             = np.array([ [ posXs[i], posYs[i] ] for i in range(len(distances)) ])   # (N,2)
	devNorm            = np.array([ [ normXs[i], normYs[i] ] for i in range(len(distances)) ]) # (N,2)
	longestSampleCount = max([ len(graph) for graph in distances ]) # = T
	values             = np.array([ np.pad(np.array(graph), (longestSampleCount - len(graph),), mode="edge") for graph in distances ]) # (N,T,2)
	uniqueTimes        = np.unique(np.concatenate(values[:,:,0]))                                                                      # (T,)
	indices            = np.array([ np.searchsorted(values[i,:,0], uniqueTimes, side="right") - 1 for i in range(len(distances))])     # (N,T)
	recentDistances    = values[np.arange(values.shape[0])[:,None], indices,1]                                                         # (N,T)
	return (devPos, devNorm, uniqueTimes, recentDistances)

def RunVectorizedBL(distances:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> list[tuple[float, float, float]]:
	devPos, devNorm, uniqueTimes, recentDistances = VectorizedHeader(distances, posXs, posYs, normXs, normYs)
	finalPos, validMask = RunBaseBL(recentDistances, devPos, devNorm)
	return list(zip(uniqueTimes[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunVectorizedTLG(distances:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> list[tuple[float, float, float]]:
	devPos, _, uniqueTimes, recentDistances = VectorizedHeader(distances, posXs, posYs, normXs, normYs)
	finalPos, validMask = RunBaseTLG(recentDistances, devPos)
	return list(zip(uniqueTimes[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunVectorizedTLMT(distances:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> list[tuple[float, float, float]]:
	devPos, _, uniqueTimes, recentDistances = VectorizedHeader(distances, posXs, posYs, normXs, normYs)
	finalPos, validMask = RunBaseTLMT(recentDistances, devPos)
	return list(zip(uniqueTimes[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunVectorizedTLMD(distances:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> list[tuple[float, float, float]]:
	devPos, _, uniqueTimes, recentDistances = VectorizedHeader(distances, posXs, posYs, normXs, normYs)
	finalPos, validMask = RunBaseTLMD(recentDistances, devPos)
	return list(zip(uniqueTimes[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunVectorizedTLSE(distances:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float]) -> list[tuple[float, float, float]]:
	devPos, _, uniqueTimes, recentDistances = VectorizedHeader(distances, posXs, posYs, normXs, normYs)
	finalPos, validMask = RunBaseTLSE(recentDistances, devPos)
	return list(zip(uniqueTimes[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

# Position Estimators Distribution:
def DistributionHeader(distributions:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float], samples:int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	maxSamples  = np.prod([ len(graph) for graph in distributions ])
	numSamples  = min(samples, maxSamples)
	flatIndices = np.unique(np.random.choice(maxSamples, numSamples))                                             # (T,)
	indices     = np.array(np.unravel_index(flatIndices, shape=tuple([ len(graph) for graph in distributions ]))) # (N,T)
	
	devPos  = np.array([ [ posXs[i], posYs[i] ] for i in range(len(distributions)) ])                        # (N,2)
	devNorm = np.array([ [ normXs[i], normYs[i] ] for i in range(len(distributions)) ])                      # (N,2)
	values  = np.array([ [ distributions[i][j] for j in indices[i,:] ] for i in range(len(distributions)) ]) # (N,T,2)

	distances = values[:,:,0]             # (N,T)
	weights   = values[:,:,1]             # (N,T)
	weights   = np.prod(weights, axis=0)  # (T,)
	weights   = weights / np.sum(weights) # (T,)
	return (devPos, devNorm, weights, distances)

def RunDistributionBL(distributions:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float], samples:int) -> list[tuple[float, float, float]]:
	devPos, devNorm, weights, distances = DistributionHeader(distributions, posXs, posYs, normXs, normYs, samples)
	finalPos, validMask = RunBaseBL(distances, devPos, devNorm)
	return list(zip(weights[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunDistributionTLG(distributions:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float], samples:int) -> list[tuple[float, float, float]]:
	devPos, _, weights, distances = DistributionHeader(distributions, posXs, posYs, normXs, normYs, samples)
	finalPos, validMask = RunBaseTLG(distances, devPos)
	return list(zip(weights[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunDistributionTLMT(distributions:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float], samples:int) -> list[tuple[float, float, float]]:
	devPos, _, weights, distances = DistributionHeader(distributions, posXs, posYs, normXs, normYs, samples)
	finalPos, validMask = RunBaseTLMT(distances, devPos)
	return list(zip(weights[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunDistributionTLMD(distributions:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float], samples:int) -> list[tuple[float, float, float]]:
	devPos, _, weights, distances = DistributionHeader(distributions, posXs, posYs, normXs, normYs, samples)
	finalPos, validMask = RunBaseTLMD(distances, devPos)
	return list(zip(weights[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

def RunDistributionTLSE(distributions:list[list[tuple[float, float]]], posXs:list[float], posYs:list[float], normXs:list[float], normYs:list[float], samples:int) -> list[tuple[float, float, float]]:
	devPos, _, weights, distances = DistributionHeader(distributions, posXs, posYs, normXs, normYs, samples)
	finalPos, validMask = RunBaseTLSE(distances, devPos)
	return list(zip(weights[validMask].tolist(), finalPos[:,0].tolist(), finalPos[:,1].tolist()))

positionEstimators = [
	("BL", RunBL, RunVectorizedBL, RunDistributionBL),
	("TLG", RunTLG, RunVectorizedTLG, RunDistributionTLG),
	("TLMT", RunTLMT, RunVectorizedTLMT, RunDistributionTLMT),
	("TLMD", RunTLMD, RunVectorizedTLMD, RunDistributionTLMD),
	("TLSE", RunTLSE, RunVectorizedTLSE, RunDistributionTLSE)
]

#

def FilterPointDistribution(distribution:list[tuple[float, float, float]], sampleCount:int) -> tuple[list[tuple[float, float, float]], tuple[float, float]]:
	if len(distribution) == 0:
		return ([], [])
	
	allPoints = np.array(distribution) # (T,3)
	avgPos    = np.sum(allPoints[:,0][:,None] * allPoints[:,1:3], axis=0)
	avgPos   /= np.sum(allPoints[:,0])
	if len(distribution) < sampleCount:
		return (distribution, (float(avgPos[0]), float(avgPos[1])))
	
	mask       = allPoints[:,0] >= np.partition(allPoints[:,0], -sampleCount)[-sampleCount]
	points     = allPoints[mask,:].copy()
	exclPoints = allPoints[~mask,:]

	sigma      = 1.0
	distances  = np.linalg.norm(points[:,None,1:3] - exclPoints[None,:,1:3], axis=-1) # (U,V)
	influences = np.exp(-0.5 * (distances / sigma)**2) # (U,V)

	influencedWeights = influences * exclPoints[None,:,0] # (U,V)
	newWeights        = points[:,0] + np.sum(influencedWeights, axis=1) # (U,)

	origWeightedPos = points[:,0:1] * points[:,1:3] # (U,2)
	inflWeightedPos = np.sum(influencedWeights[:,:,None] * exclPoints[None,:,1:3], axis=1) # (U,2)
	newPos          = (origWeightedPos + inflWeightedPos) / newWeights[:,None] # (U,2)

	points[:,0]   = newWeights
	points[:,1:3] = newPos
	points[:,0]   /= np.sum(points[:,0])

	return (points.tolist(), (float(avgPos[0]), float(avgPos[1])))

totalBeacons        = 5
totalTime           = 400.0
distributionSamples = 100_000
totalSamples        = math.floor(totalTime / 0.1)
print(f"{totalSamples} over {totalTime} seconds for {totalBeacons} beacons")

np.random.seed(0)

#npBeaconPosXs  = 100.0 * np.random.random((totalBeacons,))
#npBeaconPosYs  = 100.0 * np.random.random((totalBeacons,))
npBeaconPosXs  = np.zeros((totalBeacons,), dtype=float)
npBeaconPosYs  = np.linspace(0.0, 20.0 * totalBeacons, totalBeacons)
npBeaconNormXs = np.ones((totalBeacons,), dtype=float)
npBeaconNormYs = np.zeros((totalBeacons,), dtype=float)
npBeaconRSSI1m = -55 - 4 * np.random.random((totalBeacons,))
npBeaconRSSIk  = 2.0 + 0.2 * np.random.random((totalBeacons,))
npBeaconGains  = npBeaconRSSI1m + 40.0

beaconPosXs  = npBeaconPosXs.tolist()
beaconPosYs  = npBeaconPosYs.tolist()
beaconNormXs = npBeaconNormXs.tolist()
beaconNormYs = npBeaconNormYs.tolist()
beaconRSSI1m = npBeaconRSSI1m.tolist()
beaconRSSIk  = npBeaconRSSIk.tolist()
beaconGains  = npBeaconGains.tolist()

expectedPosX = 10.0 * np.random.random()
expectedPosY = 20.0 * totalBeacons * np.random.random()

print(np.stack((npBeaconPosXs, npBeaconPosYs), axis=-1))
print(f"Expected position: {float(expectedPosX):7.3f}, {float(expectedPosY):7.3f}")

distances     = np.sqrt((npBeaconPosXs - expectedPosX)**2 + (npBeaconPosYs - expectedPosY)**2)
expectedRSSIs = npBeaconRSSI1m - 10.0 * npBeaconRSSIk * np.log10(distances)
print(f"Expected distances: {distances}")

times  = np.array([ np.linspace(0.0, totalTime, totalSamples) + 0.1 * np.random.random((totalSamples,)) for i in range(totalBeacons) ])
values = np.array([ np.clip(np.random.normal(loc=expectedRSSIs[i], scale=0.1, size=totalSamples), min=-127.0, max=0.0).round() for i in range(totalBeacons) ])

graphs = [ list(zip(times[i,:].tolist(), values[i,:].tolist())) for i in range(totalBeacons) ]

skipFilters            = False
skipDistanceEstimators = False
skipPositionEstimators = False

if not skipFilters:
	print("Filters:")
	for name, standard, vectorized in filters:
		sys.stdout.write(f"  {name:<9}: Standard ")

		sourceGraph = graphs[0]

		standardStart = time_ns()
		standardGraph = []
		for i in range(len(sourceGraph)):
			value = standard(sourceGraph, i)
			standardGraph.append((sourceGraph[i][0], value))
		standardEnd  = time_ns()
		standardTime = (standardEnd - standardStart)

		sys.stdout.write(f"{standardTime * 1e-3:14.3f} us, Vectorized ")

		vectorizedStart = time_ns()
		vectorizedGraph = vectorized(sourceGraph)
		vectorizedEnd   = time_ns()
		vectorizedTime  = vectorizedEnd - vectorizedStart

		sys.stdout.write(f"{vectorizedTime * 1e-3:14.3f} us, {standardTime / vectorizedTime:8.3f}x Speedup")

		if len(standardGraph) != len(vectorizedGraph):
			sys.stdout.write(": LENGTHS DIFFER\n")
		else:
			for i in range(len(standardGraph)):
				if standardGraph[i][0] != vectorizedGraph[i][0]:
					sys.stdout.write(": TIMESTAMPS DIFFER\n")
					break
				if abs(standardGraph[i][1] - vectorizedGraph[i][1]) > 1e-3:
					sys.stdout.write(f": VALUES DIFFER BY {abs(standardGraph[i][1] - vectorizedGraph[i][1])}\n")
					break
			else:
				sys.stdout.write("\n")

if not skipDistanceEstimators:
	print("Distance Estimators:")
	for name, standard, vectorized, distribution, gaussian in distanceEstimators:
		sys.stdout.write(f"  {name:<9}: Standard ")

		sourceGraph = graphs[0]

		if gaussian:
			standardStart      = time_ns()
			standardLowerGraph = []
			standardMidGraph   = []
			standardUpperGraph = []
			standardRandGraph  = []
			for i in range(len(sourceGraph)):
				value = standard(sourceGraph[i][1], beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
				standardLowerGraph.append((sourceGraph[i][0], value[0]))
				standardMidGraph.append((sourceGraph[i][0], value[1]))
				standardUpperGraph.append((sourceGraph[i][0], value[2]))
				standardRandGraph.append((sourceGraph[i][0], value[3]))
			standardEnd  = time_ns()
			standardTime = (standardEnd - standardStart)

			sys.stdout.write(f"{standardTime * 1e-3:14.3f} us, Vectorized ")

			vectorizedStart = time_ns()
			vectorizedLowerGraph, vectorizedMidGraph, vectorizedUpperGraph, vectorizedRandGraph = vectorized(sourceGraph, beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
			vectorizedEnd   = time_ns()
			vectorizedTime  = vectorizedEnd - vectorizedStart

			sys.stdout.write(f"{vectorizedTime * 1e-3:14.3f} us, {standardTime / vectorizedTime:8.3f}x Speedup")

			if len(standardLowerGraph) != len(vectorizedLowerGraph) or len(standardMidGraph) != len(vectorizedMidGraph) or len(standardUpperGraph) != len(vectorizedUpperGraph) or len(vectorizedRandGraph) != len(standardRandGraph):
				sys.stdout.write(": LENGTHS DIFFER\n")
			else:
				for i in range(len(standardLowerGraph)):
					if standardLowerGraph[i][0] != vectorizedLowerGraph[i][0] or standardMidGraph[i][0] != vectorizedMidGraph[i][0] or standardUpperGraph[i][0] != vectorizedUpperGraph[i][0] or standardRandGraph[i][0] != vectorizedRandGraph[i][0]:
						sys.stdout.write(": TIMESTAMPS DIFFER\n")
						break
					if abs(standardLowerGraph[i][1] - vectorizedLowerGraph[i][1]) > 1e-3 or abs(standardMidGraph[i][1] - vectorizedMidGraph[i][1]) > 1e-3 or abs(standardUpperGraph[i][1] - vectorizedUpperGraph[i][1]) > 1e-3:
						sys.stdout.write(": VALUES DIFFER GREATLY\n")
						break
				else:
					sys.stdout.write("\n")
		else:
			standardStart = time_ns()
			standardGraph = []
			for i in range(len(sourceGraph)):
				value = standard(sourceGraph[i][1], beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
				standardGraph.append((sourceGraph[i][0], value))
			standardEnd  = time_ns()
			standardTime = (standardEnd - standardStart)

			sys.stdout.write(f"{standardTime * 1e-3:14.3f} us, Vectorized ")

			vectorizedStart = time_ns()
			vectorizedGraph = vectorized(sourceGraph, beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
			vectorizedEnd   = time_ns()
			vectorizedTime  = vectorizedEnd - vectorizedStart

			sys.stdout.write(f"{vectorizedTime * 1e-3:14.3f} us, {standardTime / vectorizedTime:8.3f}x Speedup")

			if len(standardGraph) != len(vectorizedGraph):
				sys.stdout.write(": LENGTHS DIFFER\n")
			else:
				for i in range(len(standardGraph)):
					if standardGraph[i][0] != vectorizedGraph[i][0]:
						sys.stdout.write(": TIMESTAMPS DIFFER\n")
						break
					if abs(standardGraph[i][1] - vectorizedGraph[i][1]) > 1e-3:
						sys.stdout.write(f": VALUES DIFFER BY {abs(standardGraph[i][1] - vectorizedGraph[i][1])}\n")
						break
				else:
					sys.stdout.write("\n")

		sys.stdout.write(f"  {' Dist':<9}: Standard ")
		if gaussian:
			standardStart        = time_ns()
			standardDistributions = []
			for i in range(0, -128, -1):
				lower, mid, upper, rand = standard(i, beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
				standardDistribution.append(mid) # Not exactly indicative of the real performance we would've spent here...
			standardEnd          = time_ns()
			standardTime         = standardEnd - standardStart

			sys.stdout.write(f"{standardTime * 1e-3:14.3f} us, Vectorized ")

			vectorizedStart        = time_ns()
			vectorizedDistribution = distribution(beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
			vectorizedEnd          = time_ns()
			vectorizedTime         = vectorizedEnd - vectorizedStart

			sys.stdout.write(f"{vectorizedTime * 1e-3:14.3f} us, {standardTime / vectorizedTime:8.3f}x Speedup\n")
		else:
			standardStart        = time_ns()
			standardDistribution = []
			for i in range(0, -128, -1):
				standardDistribution.append(standard(i, beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0]))
			standardEnd          = time_ns()
			standardTime         = standardEnd - standardStart

			sys.stdout.write(f"{standardTime * 1e-3:14.3f} us, Vectorized ")

			vectorizedStart        = time_ns()
			vectorizedDistribution = distribution(beaconRSSI1m[0], beaconRSSIk[0], beaconGains[0])
			vectorizedEnd          = time_ns()
			vectorizedTime         = vectorizedEnd - vectorizedStart

			sys.stdout.write(f"{vectorizedTime * 1e-3:14.3f} us, {standardTime / vectorizedTime:8.3f}x Speedup")

			if len(standardDistribution) != len(vectorizedDistribution):
				sys.stdout.write(": LENGTHS DIFFER\n")
			else:
				for i in range(len(standardDistribution)):
					if abs(standardDistribution[i] - vectorizedDistribution[i]) > 1e-3:
						sys.stdout.write(": VALUES DIFFER GREATLY\n")
						break
				else:
					sys.stdout.write("\n")

filteredGraphs = [ RunVectorizedSMA(graphs[i]) for i in range(totalBeacons) ]
distanceGraphs = [ RunVectorizedEmpirical(filteredGraphs[i], beaconRSSI1m[i], beaconRSSIk[i], beaconGains[i]) for i in range(totalBeacons) ]

distanceDistribution  = [ RunDistributionEmpirical(beaconRSSI1m[i], beaconRSSIk[i], beaconGains[i]) for i in range(totalBeacons) ]
distanceDistributionG = [ RunDistributionLDPL(beaconRSSI1m[i], beaconRSSIk[i], beaconGains[i]) for i in range(totalBeacons) ]

distanceDistributions  = [ GetDistribution(distanceDistribution[i], [ int(sample[1]) for sample in graphs[i] ]) for i in range(totalBeacons) ]
distanceDistributionsG = [ GetDistribution(distanceDistributionG[i], [ int(sample[1]) for sample in graphs[i] ]) for i in range(totalBeacons) ]

if not skipPositionEstimators:
	print("Position Estimators:")
	for name, standard, vectorized, distribution in positionEstimators:
		sys.stdout.write(f"  {name:<9}: Standard ")

		standardStart = time_ns()
		standardGraph = []

		uniqueTimestamps = []
		for graph in distanceGraphs:
			uniqueTimestamps.extend([ sample[0] for sample in graph ])
		uniqueTimestamps.sort()
		uniqueTimestamps = np.unique(uniqueTimestamps).tolist()
		
		for timestamp in uniqueTimestamps:
			distances = []
			for i in range(totalBeacons):
				idx = bisect.bisect_right(distanceGraphs[i], timestamp, key=lambda x:x[0]) - 1
				distances.append(distanceGraphs[i][idx][1])

			value = standard(distances, beaconPosXs, beaconPosYs, beaconNormXs, beaconNormYs)
			if value is not None:
				standardGraph.append((timestamp, value[0], value[1]))
		standardEnd  = time_ns()
		standardTime = (standardEnd - standardStart)

		sys.stdout.write(f"{standardTime * 1e-3:14.3f} us, Vectorized ")

		vectorizedStart = time_ns()
		vectorizedGraph = vectorized(distanceGraphs, beaconPosXs, beaconPosYs, beaconNormXs, beaconNormYs)
		vectorizedEnd   = time_ns()
		vectorizedTime  = vectorizedEnd - vectorizedStart

		sys.stdout.write(f"{vectorizedTime * 1e-3:14.3f} us, {standardTime / vectorizedTime:8.3f}x Speedup")

		if len(standardGraph) != len(vectorizedGraph):
			sys.stdout.write(f": LENGTHS DIFFER {len(standardGraph)} vs {len(vectorizedGraph)}\n")
		else:
			for i in range(len(standardGraph)):
				if standardGraph[i][0] != vectorizedGraph[i][0]:
					sys.stdout.write(": TIMESTAMPS DIFFER\n")
					break
				dx = abs(standardGraph[i][1] - vectorizedGraph[i][1])
				dy = abs(standardGraph[i][2] - vectorizedGraph[i][2])
				if (dx != 0 and (math.log10(dx) > (math.log10(abs(standardGraph[i][1])) if standardGraph[i][1] != 0.0 else 0) - 3)) or (dy != 0 and (math.log10(dy) > (math.log10(abs(standardGraph[i][2])) if standardGraph[i][2] != 0.0 else 0) - 3)):
					sys.stdout.write(f": VALUES DIFFER BY {(math.log10(dx) if dx != 0 else 0) + (math.log10(dy) if dx != 0 else 0)}\n")
					break
			else:
				sys.stdout.write("\n")

		sys.stdout.write(f"  {' Dist':<9}: Linear   ")

		linearStart        = time_ns()
		linearDistribution = distribution(distanceDistributions, beaconPosXs, beaconPosYs, beaconNormXs, beaconNormYs, distributionSamples)
		linearEnd          = time_ns()
		linearTime         = (linearEnd - linearStart)

		sys.stdout.write(f"{linearTime * 1e-3:14.3f} us, Gaussian   ")
		gaussianStart        = time_ns()
		gaussianDistribution = distribution(distanceDistributionsG, beaconPosXs, beaconPosYs, beaconNormXs, beaconNormYs, distributionSamples)
		gaussianEnd          = time_ns()
		gaussianTime         = (gaussianEnd - gaussianStart)
		sys.stdout.write(f"{gaussianTime * 1e-3:14.3f} us\n")

		linearPoints, avgLinearPos     = FilterPointDistribution(linearDistribution, 1000)
		gaussianPoints, avgGaussianPos = FilterPointDistribution(gaussianDistribution, 1000)
		
		import matplotlib.pyplot as plt

		fig, axs = plt.subplots(2, 2, layout="constrained", sharex=True, sharey=True)
		axs[0,0].set_xlabel("X Position")
		axs[0,1].set_xlabel("X Position")
		axs[1,0].set_xlabel("X Position")
		axs[1,1].set_xlabel("X Position")
		axs[0,0].set_ylabel("Y Position")
		axs[0,1].set_ylabel("Y Position")
		axs[1,0].set_ylabel("Y Position")
		axs[1,1].set_ylabel("Y Position")

		if len(linearPoints) > 0:
			linearWeights, linearXs, linearYs = zip(*linearPoints)
			c = axs[0,0].scatter(linearXs, linearYs, c=linearWeights, cmap="viridis", s=40 / np.max(linearWeights) * np.array(linearWeights), alpha=0.7)
			fig.colorbar(c, ax=axs[0,0], label="Weight")
			_,_,_,c = axs[0,1].hist2d(linearXs, linearYs, weights=linearWeights, bins=50, cmap="viridis")
			fig.colorbar(c, ax=axs[0,1], label="Weight")
			axs[0,0].scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
			axs[0,1].scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
			if len(avgLinearPos) > 0:
				axs[0,0].scatter([ avgLinearPos[0] ], [ avgLinearPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")
				axs[0,1].scatter([ avgLinearPos[0] ], [ avgLinearPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")

		if len(gaussianPoints) > 0:
			gaussianWeights, gaussianXs, gaussianYs = zip(*gaussianPoints)
			c = axs[1,0].scatter(gaussianXs, gaussianYs, c=gaussianWeights, cmap="viridis", s=40 / np.max(gaussianWeights) * np.array(gaussianWeights), alpha=0.7)
			fig.colorbar(c, ax=axs[1,0], label="Weight")
			_,_,_,c = axs[1,1].hist2d(gaussianXs, gaussianYs, weights=gaussianWeights, bins=50, cmap="viridis")
			fig.colorbar(c, ax=axs[1,1], label="Weight")
			axs[1,0].scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
			axs[1,1].scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
			if len(avgGaussianPos) > 0:
				axs[1,0].scatter([ avgGaussianPos[0] ], [ avgGaussianPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")
				axs[1,1].scatter([ avgGaussianPos[0] ], [ avgGaussianPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")

		axs[0,0].grid(True)
		axs[0,1].grid(True)
		axs[1,0].grid(True)
		axs[1,1].grid(True)

		plt.show()
# sc = plt.scatter(xs, ys, c=weights, cmap="viridis", s=40 / np.max(weights) * np.array(weights), alpha=0.7)
# plt.colorbar(sc, label="Weight")
# plt.scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
# plt.scatter([ avgPos[0] ], [ avgPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Position Distribution (Weighted)")
# plt.grid(True)
# plt.show()

# Plotting test:

# import matplotlib.pyplot as plt

# allPoints  = np.array(positionDistributions) # (T,3)
# mask       = allPoints[:,0] >= np.partition(allPoints[:,0], -1000)[-1000]
# points     = allPoints[mask,:].copy()
# exclPoints = allPoints[~mask,:]

# sigma      = 1.0
# distances  = np.linalg.norm(points[:,None,1:3] - exclPoints[None,:,1:3], axis=-1) # (U,V)
# influences = np.exp(-0.5 * (distances / sigma)**2) # (U,V)

# influencedWeights = influences * exclPoints[None,:,0] # (U,V)
# newWeights        = points[:,0] + np.sum(influencedWeights, axis=1) # (U,)

# origWeightedPos = points[:,0:1] * points[:,1:3] # (U,2)
# inflWeightedPos = np.sum(influencedWeights[:,:,None] * exclPoints[None,:,1:3], axis=1) # (U,2)
# newPos          = (origWeightedPos + inflWeightedPos) / newWeights[:,None] # (U,2)

# points[:,0]       = newWeights
# points[:,1:3]     = newPos
# points[:,0]      /= np.sum(points[:,0])

# avgPos = np.sum(allPoints[:,0][:,None] * allPoints[:,1:3], axis=0)

# weights, xs, ys = points[:,0], points[:,1], points[:,2]
# plt.figure(figsize=(8,6))
# sc = plt.scatter(xs, ys, c=weights, cmap="viridis", s=40 / np.max(weights) * np.array(weights), alpha=0.7)
# plt.colorbar(sc, label="Weight")
# plt.scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
# plt.scatter([ avgPos[0] ], [ avgPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Position Distribution (Weighted)")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))
# plt.hist2d(xs, ys, weights=weights, bins=50, cmap="viridis")
# plt.scatter([ expectedPosX ], [ expectedPosY ], s=50, color=(0,0,0,0), edgecolor="r")
# plt.scatter([ avgPos[0] ], [ avgPos[1] ], s=50, color=(0,0,0,0), edgecolor="darkorange")
# plt.colorbar(label="Total Weight")
# plt.xlabel("X Position")
# plt.ylabel("Y Position")
# plt.title("Position Distribution (Weighted)")
# plt.grid(True)
# plt.show()