from typing import overload

from MACAddress import MACAddress

import numpy as np

BLE_CHANNEL_FREQUENCY = [
    2.404,
    2.406,
    2.408,
    2.410,
    2.412,
    2.414,
    2.416,
    2.418,
    2.420,
    2.422,
    2.424,
    2.428,
    2.430,
    2.432,
    2.434,
    2.436,
    2.438,
    2.440,
    2.442,
    2.444,
    2.446,
    2.448,
    2.450,
    2.452,
    2.454,
    2.456,
    2.458,
    2.460,
    2.462,
    2.464,
    2.466,
    2.468,
    2.470,
    2.472,
    2.474,
    2.476,
    2.478,
    2.402, # 37
    2.426, # 38
    2.480  # 39
]

class Beacon:
    def __init__(self, name:str, Tx:float, height:float, posX:float, posY:float, normX:float, normY:float):
        self.Name      = name
        self.Tx        = Tx
        self.Height    = height
        self.PosX      = posX
        self.PosY      = posY
        self.NormX     = normX
        self.NormY     = normY
        self.RealPosX  = self.PosX
        self.RealPosY  = self.PosY
        self.RealNormX = self.NormX
        self.RealNormY = self.NormY

class Sniffer:
    def __init__(self, BLEChannel:int, height:float, dX:float, dY:float):
        self.BLEChannel = BLEChannel
        self.Frequency  = BLE_CHANNEL_FREQUENCY[self.BLEChannel]
        self.Height     = height
        self.dX         = dX
        self.dY         = dY

class StaticGrid:
    def __init__(self, startX:float, startY:float, deltaX:float, deltaY:float):
        self.StartX = startX
        self.StartY = startY
        self.DeltaX = deltaX
        self.DeltaY = deltaY

class Segment:
    def __init__(self, yVal:float, left:tuple[float, float], right:tuple[float, float]):
        self.YVal  = yVal
        self.Left  = left
        self.Right = right

class Tunnel:
    def __init__(self, width:float, depth:float, walls:list[Segment]):
        self.Width = width
        self.Depth = depth
        self.Walls = walls

class Parameters:
    def __init__(self, beacons:dict[MACAddress, Beacon], sniffer:Sniffer, staticGrid:StaticGrid, tunnel:Tunnel, windowSizes:list[float]):
        self.Beacons     = beacons
        self.Sniffer     = sniffer
        self.StaticGrid  = staticGrid
        self.Tunnel      = tunnel
        self.WindowSizes = windowSizes

        self.TunnelYVals:np.ndarray = None
        
    def _TransformPoint(self, point:tuple[float, float], snifferPoint:bool) -> tuple[float, float]:
        if snifferPoint:
            point = (point[0] * self.StaticGrid.DeltaX + self.StaticGrid.StartX + self.Sniffer.dX, point[1] * self.StaticGrid.DeltaY + self.StaticGrid.StartY + self.Sniffer.dY)
        x, y = point

        idx      = min(np.searchsorted(self.TunnelYVals, y), len(self.TunnelYVals) - 2)
        segment1 = self.Tunnel.Walls[idx]
        segment2 = self.Tunnel.Walls[idx + 1]

        yfac = (y - segment1.YVal) / (segment2.YVal - segment1.YVal)
        xfac = x / self.Tunnel.Width

        left1, right1 = segment1.Left, segment1.Right
        left2, right2 = segment2.Left, segment2.Right

        left3  = (left1[0] + (left2[0] - left1[0]) * yfac, left1[1] + (left2[1] - left1[1]) * yfac)
        right3 = (right1[0] + (right2[0] - right1[0]) * yfac, right1[1] + (right2[1] - right1[1]) * yfac)
        return (left3[0] + (right3[0] - left3[0]) * xfac, left3[1] + (right3[1] - left3[1]) * xfac)

    @overload
    def Transform(self, point:tuple[float, float], *, snifferPoint:bool) -> tuple[float, float]: ...
    @overload
    def Transform(self, points:list[tuple[float, float]], *, snifferPoint:bool) -> list[tuple[float, float]]: ...

    def Transform(self, pos:tuple[float, float]|list[tuple[float, float]], *, snifferPoint:bool = False) -> tuple[float, float]|list[tuple[float, float]]:
        if type(pos) == list:
            outPoints:list[tuple[float, float]] = []
            for point in pos:
                outPoints.append(self._TransformPoint(point, snifferPoint))
            return outPoints
        return self._TransformPoint(pos, snifferPoint)
    
    def GetNormal(self, pos:tuple[float, float], *, snifferPoint:bool = False) -> tuple[float, float]:
        if snifferPoint:
            point = (point[0] * self.StaticGrid.DeltaX + self.StaticGrid.StartX + self.Sniffer.dX, point[1] * self.StaticGrid.DeltaY + self.StaticGrid.StartY + self.Sniffer.dY)
        x, y = pos

        idx      = min(np.searchsorted(self.TunnelYVals, y), len(self.TunnelYVals) - 2)
        segment1 = self.Tunnel.Walls[idx]
        segment2 = self.Tunnel.Walls[idx + 1]

        yfac = (y - segment1.YVal) / (segment2.YVal - segment1.YVal)

        left1, right1 = segment1.Left, segment1.Right
        left2, right2 = segment2.Left, segment2.Right

        left3  = (left1[0] + (left2[0] - left1[0]) * yfac, left1[1] + (left2[1] - left1[1]) * yfac)
        right3 = (right1[0] + (right2[0] - right1[0]) * yfac, right1[1] + (right2[1] - right1[1]) * yfac)
        return ((right3[0] - left3[0]) / self.Tunnel.Width, (right3[1] - left3[1]) / self.Tunnel.Width)