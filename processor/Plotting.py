import math

import numpy as np

from matplotlib.axes import Axes

from MACAddress import MACAddress
import SessionParameters as SP

def TunnelGeometry(ax:Axes, sessionParmaeters:SP.Parameters, **kwargs):
	leftWall  = np.array([ segment.Left for segment in sessionParmaeters.Tunnel.Walls ])
	rightWall = np.array([ segment.Right for segment in sessionParmaeters.Tunnel.Walls ])

	l1 = ax.plot(leftWall[:,0], leftWall[:,1], **kwargs)
	l2 = ax.plot(rightWall[:,0], rightWall[:,1], **kwargs)
	return (l1, l2)

def Beacons(ax:Axes, sessionParameters:SP.Parameters, addr:MACAddress, defaultColor="r", selectedColor="g", labels:bool=True, **kwargs):
	beaconItems = list(sessionParameters.Beacons.items())
	xs = [ beacon.RealPosX for _, beacon in beaconItems ]
	ys = [ beacon.RealPosY for _, beacon in beaconItems ]
	cs = [ selectedColor if addr2 == addr else defaultColor for addr2, beacon in beaconItems ]
	if labels:
		for addr2, beacon in beaconItems:
			ax.text(beacon.RealPosX - 1.0, beacon.RealPosY, beacon.Name, horizontalalignment="right", verticalalignment="center", transform=ax.transData)
	return ax.scatter(x=xs, y=ys, c=cs, **kwargs)

def Heatmap(ax:Axes, xy:np.ndarray, zvalues:np.ndarray, resX:float=0.1, resY:float=0.1, w:float=1.5, maxDist:float=1.5, *,
			minX:float = None, minY:float = None,
			maxX:float = None, maxY:float = None,
			**kwargs):
	mask2   = np.isfinite(zvalues)
	xyValid = xy[mask2]
	zValid  = zvalues[mask2]

	minX = (np.min(xyValid[:,0]) - 1.0) if minX is None else minX
	maxX = (np.max(xyValid[:,0]) + 1.0) if maxX is None else maxX
	minY = (np.min(xyValid[:,1]) - 1.0) if minY is None else minY
	maxY = (np.max(xyValid[:,1]) + 1.0) if maxY is None else maxY

	vw     = int(math.ceil((maxX - minX) / resX))
	vh     = int(math.ceil((maxY - minY) / resY))
	xs, ys = np.meshgrid(np.linspace(minX, maxX, vw), np.linspace(minY, maxY, vh))

	gridPoints   = np.stack([ xs.ravel(), ys.ravel() ], axis=1)   # (M*N,2)
	deltaSq      = (gridPoints[:,None,:] - xyValid[None,:,:])**2  # (M*N,O,2)
	distSq       = np.sum(deltaSq, axis=-1)                       # (M*N,O)
	minDistSq    = np.min(distSq, axis=1)                         # (M*N,)
	mask         = (minDistSq <= maxDist**2)                      # (M*N,)
	validDistSq  = distSq[mask]
	validWeights = 1.0 / (np.clip(validDistSq**w, min=1e-9, max=None))
	validZs      = (np.sum(validWeights * zValid[None,:], axis=-1) /  np.sum(validWeights, axis=-1))
	
	zsFull = np.full(xs.size, np.nan)
	zsFull[mask] = validZs
	zsFull = zsFull.reshape(xs.shape)

	return ax.pcolormesh(xs, ys, zsFull, cmap="plasma", rasterized=True, **kwargs)