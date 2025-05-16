import os
import json
import csv
from typing import overload

from MACAddress import MACAddress
import SessionParameters as SP
from IProcessor import DynamicRun, StaticRun
import SourceSession as SS
import Session

import numpy as np

def GetTestRootDir(testname:str) -> str:
    return f"tests/{testname}"

@overload
def GetTestProcessedDir(testname:str, run:StaticRun) -> str: ...
@overload
def GetTestProcessedDir(testname:str, run:DynamicRun) -> str: ...

def GetTestProcessedDir(testname:str, run:StaticRun|DynamicRun) -> str:
    kind    = f"{'static' if type(run) == StaticRun else 'dynamic'}_{'full' if run.BadCRCs else 'good'}"
    runName = f"{run.WindowSize if run.WindowSize >= 0.0 else ''}/{run.WindowIndex if run.WindowIndex >= 0 else ''}" if type(run) == StaticRun else ""
    return f"{GetTestRootDir(testname)}/processed/{kind}/{run.GetPath()}/{runName}"

def GetTestAnalysisRootDir(testname:str, static:bool, badCRCs:bool) -> str:
    kind = f"{'static' if static else 'dynamic'}_{'full' if badCRCs else 'good'}_analysis"
    return f"{GetTestRootDir(testname)}/processed/{kind}/"

@overload
def GetTestAnalysisDir(testname:str, run:StaticRun) -> str: ...
@overload
def GetTestAnalysisDir(testname:str, run:DynamicRun) -> str: ...

def GetTestAnalysisDir(testname:str, run:StaticRun|DynamicRun) -> str:
    kind    = f"{'static' if type(run) == StaticRun else 'dynamic'}_{'full' if run.BadCRCs else 'good'}_analysis"
    runName = f"{run.WindowSize if run.WindowSize >= 0.0 else ''}/{run.WindowIndex if run.WindowIndex >= 0 else ''}" if type(run) == StaticRun else ""
    return f"{GetTestRootDir(testname)}/processed/{kind}/{run.GetPath()}/{runName}"

@overload
def GetTestPlotsDir(testname:str, run:StaticRun) -> str: ...
@overload
def GetTestPlotsDir(testname:str, run:DynamicRun) -> str: ...

def GetTestPlotsDir(testname:str, run:StaticRun|DynamicRun) -> str:
    kind    = f"{'static' if type(run) == StaticRun else 'dynamic'}_{'full' if run.BadCRCs else 'good'}_plots"
    runName = f"{run.WindowSize if run.WindowSize >= 0.0 else ''}/{run.WindowIndex if run.WindowIndex >= 0 else ''}" if type(run) == StaticRun else ""
    return f"{GetTestRootDir(testname)}/processed/{kind}/{run.GetPath()}/{runName}"

@overload
def GetTestCalibrationDir(testname:str, run:StaticRun) -> str: ...
@overload
def GetTestCalibrationDir(testname:str, run:DynamicRun) -> str: ...

def GetTestCalibrationDir(testname:str, run:StaticRun|DynamicRun) -> str:
    kind = f"{'static' if type(run) == StaticRun else 'dynamic'}_{'full' if run.BadCRCs else 'good'}"
    return f"{GetTestRootDir(testname)}/processed/{kind}"

def LoadSessionParameters(testname:str) -> SP.Parameters:
    """
    Loads Session Parameters for the given test.
    """
    testRootDir = GetTestRootDir(testname)
    if not os.path.exists(f"{testRootDir}/parameters.json"):
        raise RuntimeError(f"Test '{testname}' missing 'parameters.json' file")
    
    with open(f"{testRootDir}/parameters.json", "r") as inFile:
        obj = json.load(inFile)

        if "Beacons" not in obj or type(obj["Beacons"]) != dict:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Beacons\" JSONObject")
        if "Sniffer" not in obj or type(obj["Sniffer"]) != dict:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Sniffer\" JSONObject")
        if "StaticGrid" not in obj or type(obj["StaticGrid"]) != dict:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"StaticGrid\" JSONObject")
        if "Tunnel" not in obj or type(obj["Tunnel"]) != dict:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Tunnel\" JSONObject")
        if "WindowSizes" not in obj or type(obj["WindowSizes"]) != list:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"WindowSizes\" JSONArray")
        
        beaconsObj     = obj["Beacons"]
        snifferObj     = obj["Sniffer"]
        staticGridObj  = obj["StaticGrid"]
        tunnelObj      = obj["Tunnel"]
        windowSizesArr = obj["WindowSizes"]

        beacons:dict[MACAddress, SP.Beacon] = {}
        for address, beaconObj in beaconsObj.items():
            try:
                address = MACAddress(address)
            except:
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" with an invalid MAC Address")
                continue

            if "Name" not in beaconObj or type(beaconObj["Name"]) != str:
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"Name\" string")
                continue
            if "Tx" not in beaconObj or type(beaconObj["Tx"]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"Tx\" number")
                continue
            if "Height" not in beaconObj or type(beaconObj["Height"]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"Height\" number")
                continue
            if "PosX" not in beaconObj or type(beaconObj["PosX"]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"PosX\" number")
                continue
            if "PosY" not in beaconObj or type(beaconObj["PosY"]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"PosY\" number")
                continue
            if "NormX" not in beaconObj or type(beaconObj["NormX"]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"NormX\" number")
                continue
            if "NormY" not in beaconObj or type(beaconObj["NormY"]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes a \"Beacon\" without a \"NormY\" number")
                continue

            beacon = SP.Beacon(
                name=str(beaconObj["Name"]),
                Tx=float(beaconObj["Tx"]),
                height=float(beaconObj["Height"]),
                posX=float(beaconObj["PosX"]),
                posY=float(beaconObj["PosY"]),
                normX=float(beaconObj["NormX"]),
                normY=float(beaconObj["NormY"]))
            beacons[address] = beacon
        
        if "BLEChannel" not in snifferObj or type(snifferObj["BLEChannel"]) != int:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"BLEChannel\" integer in \"Sniffer\"")
        if "Height" not in snifferObj or type(snifferObj["Height"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Height\" number in \"Sniffer\"")
        if "dX" not in snifferObj or type(snifferObj["dX"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"dX\" number in \"Sniffer\"")
        if "dY" not in snifferObj or type(snifferObj["dY"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"dY\" number in \"Sniffer\"")
        
        sniffer = SP.Sniffer(
            BLEChannel=int(snifferObj["BLEChannel"]),
            height=float(snifferObj["Height"]),
            dX=float(snifferObj["dX"]),
            dY=float(snifferObj["dY"]))
        
        if "StartX" not in staticGridObj or type(staticGridObj["StartX"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"StartX\" number in \"StaticGrid\"")
        if "StartY" not in staticGridObj or type(staticGridObj["StartY"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"StartY\" number in \"StaticGrid\"")
        if "DeltaX" not in staticGridObj or type(staticGridObj["DeltaX"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"DeltaX\" number in \"StaticGrid\"")
        if "DeltaY" not in staticGridObj or type(staticGridObj["DeltaY"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"DeltaY\" number in \"StaticGrid\"")
        
        staticGrid = SP.StaticGrid(
            startX=float(staticGridObj["StartX"]),
            startY=float(staticGridObj["StartY"]),
            deltaX=float(staticGridObj["DeltaX"]),
            deltaY=float(staticGridObj["DeltaY"]))
        
        if "Width" not in tunnelObj or type(tunnelObj["Width"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Width\" number in \"Tunnel\"")
        if "Depth" not in tunnelObj or type(tunnelObj["Depth"]) not in (float, int):
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Depth\" number in \"Tunnel\"")
        if "Walls" not in tunnelObj or type(tunnelObj["Walls"]) != list:
            raise RuntimeError(f"'parameters.json' file in test '{testname}' does not include \"Walls\" JSONArray in \"Tunnel\"")

        walls:list[SP.Segment] = []
        wallsObj = tunnelObj["Walls"]
        for segmentObj in wallsObj:
            if type(segmentObj) != list or len(segmentObj) != 3:
                print(f"WARN: 'parameters.json' file in test '{testname}' includes an invalid \"Wall\" segment, should be a list of y value and two lists of two numbers each, I.e. [ y, [x1,y1], [x2,y2] ]")
                continue

            yVal     = segmentObj[0]
            leftObj  = segmentObj[1]
            rightObj = segmentObj[2]
            if type(yVal) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes an invalid \"Wall\" segment yVal, should be a number")
                continue
            if type(leftObj) != list or len(leftObj) != 2 or type(leftObj[0]) not in (float, int) or type(leftObj[1]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes an invalid \"Wall\" segment left point, should be a list of two numbers")
                continue
            if type(rightObj) != list or len(rightObj) != 2 or type(rightObj[0]) not in (float, int) or type(rightObj[1]) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes an invalid \"Wall\" segment right point, should be a list of two numbers")
                continue

            walls.append(SP.Segment(float(yVal), (float(leftObj[0]), float(leftObj[1])), (float(rightObj[0]), float(rightObj[1]))))

        tunnel = SP.Tunnel(
            width=float(tunnelObj["Width"]),
            depth=float(tunnelObj["Depth"]),
            walls=walls
        )

        windowSizes:list[float] = []
        for windowSizeObj in windowSizesArr:
            if type(windowSizeObj) not in (float, int):
                print(f"WARN: 'parameters.json' file in test '{testname}' includes an invalid \"WindowSizes\" window size, should be a number")
            windowSizes.append(float(windowSizeObj))

        parameters             = SP.Parameters(beacons, sniffer, staticGrid, tunnel, windowSizes)
        parameters.TunnelYVals = np.array([ segment.YVal for segment in walls ])
        for addr, beacon in beacons.items():
            beacon.RealPosX,  beacon.RealPosY  = parameters.Transform((beacon.PosX, beacon.PosY))
            beacon.RealNormX, beacon.RealNormY = parameters.GetNormal((beacon.PosX, beacon.PosY))
        return parameters

def StoreSessionParameters(testname:str, sessionParameters:SP.Parameters):
    testRootDir = GetTestRootDir(testname)

    beaconsObj = {}
    for address, beacon in sessionParameters.Beacons.items():
        beaconObj = {
            "Name": beacon.Name,
            "Tx": beacon.Tx,
            "Height": beacon.Height,
            "PosX": beacon.PosX,
            "PosY": beacon.PosY,
            "NormX": beacon.NormX,
            "NormY": beacon.NormY
        }
        beaconsObj[str(address)] = beaconObj
    snifferObj = {
        "BLEChannel": sessionParameters.Sniffer.BLEChannel,
        "Height": sessionParameters.Sniffer.Height,
        "dX": sessionParameters.Sniffer.dX,
        "dY": sessionParameters.Sniffer.dY
    }
    staticGridObj = {
        "StartX": sessionParameters.StaticGrid.StartX,
        "StartY": sessionParameters.StaticGrid.StartY,
        "DeltaX": sessionParameters.StaticGrid.DeltaX,
        "DeltaY": sessionParameters.StaticGrid.DeltaY
    }
    tunnelObj = {
        "Width": sessionParameters.Tunnel.Width,
        "Depth": sessionParameters.Tunnel.Depth,
        "Walls": [ [ segment.Left, segment.Right ] for segment in sessionParameters.Tunnel.Walls ]
    }

    obj = {
        "Beacons": beaconsObj,
        "Sniffer": snifferObj,
        "StaticGrid": staticGridObj,
        "Tunnel": tunnelObj,
        "WindowSizes": sessionParameters.WindowSizes
    }
    os.makedirs(testRootDir, exist_ok=True)
    with open(f"{testRootDir}/parameters.json", "w") as outFile:
        json.dump(obj, outFile, indent=4)

@overload
def LoadParameters(testname:str, run:StaticRun) -> dict[str, dict[str, float]]|None: ...
@overload
def LoadParameters(testname:str, run:DynamicRun) -> dict[str, dict[str, float]]|None: ...

def LoadParameters(testname:str, run:StaticRun|DynamicRun) -> dict[str, dict[str, float]]|None:
    testDir = GetTestCalibrationDir(testname, run)
    if not os.path.isfile(f"{testDir}/{run.GetCalibrationPath()}_parameters.json"):
        return None
    
    with open(f"{testDir}/{run.GetCalibrationPath()}_parameters.json", "r") as inFile:
        obj = json.load(inFile)

        outParams:dict[str, dict[str, float]] = {}
        if type(run) == DynamicRun:
            if "Filter" in obj and type(obj["Filter"]) == dict:
                filterParams:dict[str, float] = {}

                for key, value in obj["Filter"].items():
                    if type(value) not in (float, int):
                        continue
                    filterParams[key] = float(value)
                outParams["Filter"] = filterParams

        if "Distance" in obj and type(obj["Distance"]) == dict:
            distanceParams:dict[str, float] = {}

            for key, value in obj["Distance"].items():
                if type(value) not in (float, int):
                    continue
                distanceParams[key] = float(value)
            outParams["Distance"] = distanceParams

        if "Position" in obj and type(obj["Position"]) == dict:
            positionParams:dict[str, float] = {}

            for key, value in obj["Position"].items():
                if type(value) not in (float, int):
                    continue
                positionParams[key] = float(value)
            outParams["Position"] = positionParams
        
        return outParams

@overload
def StoreParameters(testname:str, run:StaticRun, parameters:dict[str, dict[str, float]]): ...
@overload
def StoreParameters(testname:str, run:DynamicRun, parameters:dict[str, dict[str, float]]): ...

def StoreParameters(testname:str, run:StaticRun|DynamicRun, parameters:dict[str, dict[str, float]]):
    testDir = GetTestCalibrationDir(testname, run)
    os.makedirs(testDir, exist_ok=True)
    with open(f"{testDir}/{run.GetCalibrationPath()}_parameters.json", "w") as outFile:
        json.dump(parameters, outFile, indent=4)

def LoadSourceSession(testname:str, static:bool) -> SS.Session:
    testDir = f"{GetTestRootDir(testname)}/{'static' if static else 'dynamic'}"
    if not os.path.isdir(testDir):
        return None
    
    session = SS.Session()
    for periodName in [ d for d in os.listdir(testDir) if os.path.isdir(f"{testDir}/{d}") ]:
        periodName = periodName.removesuffix("/").removesuffix("\\")
        periodDir  = f"{testDir}/{periodName}"

        period  = SS.Period()
        minTime = None
        for address in [ f for f in os.listdir(periodDir) if os.path.isfile(f"{periodDir}/{f}") and f.endswith(".csv") ]:
            try:
                address = address.removesuffix(".csv")
                addr    = MACAddress(address)
            except:
                continue

            graph = SS.Graph()
            with open(f"{periodDir}/{address}.csv", "r") as inFile:
                reader = csv.DictReader(inFile, delimiter=",")
                for row in reader:
                    graph.Samples.append(SS.Sample(
                        timestamp=int(row["Timestamp"]),
                        cpuTimestamp=int(row["CPUTimestamp"]),
                        rssi=int(row["RSSI"]),
                        crcOK=row["CRCOK"].casefold() == "True".casefold()
                    ))
            if len(graph) == 0:
                continue
            if minTime is None or graph[0].Time < minTime:
                minTime = graph[0].Time
            period[addr] = graph
        if len(period) > 0:
            for _, graph in period:
                for sample in graph:
                    sample.Time -= minTime
            session[periodName] = period
    return session

def LoadDynamicPaths(testname:str) -> dict[str, list[tuple[float, float, float]]]:
    testDir = f"{GetTestRootDir(testname)}/dynamic"
    if not os.path.isdir(testDir):
        return None
    
    session:dict[str, list[tuple[float, float, float]]] = {}
    for periodName in [ d for d in os.listdir(testDir) if os.path.isdir(f"{testDir}/{d}") ]:
        periodName = periodName.removesuffix("/").removesuffix("\\")
        periodDir  = f"{testDir}/{periodName}"

        if not os.path.isfile(f"{periodDir}/path.json"):
            continue

        points:list[tuple[float, float, float]] = []
        with open(f"{periodDir}/path.json", "r") as inFile:
            obj = json.load(inFile)
            if "Points" not in obj or type(obj["Points"]) != list:
                print(f"WARN: 'path.json' file in test '{testname}' dynamic period '{periodName}' is missing \"Points\" JSONArray.")
                continue

            pointsArr = obj["Points"]
            for pointArr in pointsArr:
                if type(pointArr) != list or len(pointArr) != 3 or type(pointArr[0]) not in (float, int) or type(pointArr[1]) not in (float, int) or type(pointArr[2]) not in (float, int):
                    print(f"WARN: 'path.json' file in test '{testname}' dynamic period '{periodName}' contains invalid point in \"Points\", should be list of 3 numbers, [ Time, X, Y ].")
                    continue
                points.append((float(pointArr[0]), float(pointArr[1]), float(pointArr[2])))
        if len(points) < 2:
            print(f"WARN: 'path.json' file in test '{testname}' dynamic period '{periodName}' needs a \"Points\" JOSNArray with at least 2 points.")
            continue
        session[periodName] = points
    return session

def StoreDynamicSession(testname:str, run:DynamicRun, kind:str, session:Session.DynamicSession, *,
                        labels:list[str] = [ "Time", "Value" ]):
    testDir = GetTestProcessedDir(testname, run)
    for index, period in session:
        if len(period.Graphs) > 1:
            outDir = f"{testDir}/{kind}/{index}"
            os.makedirs(outDir, exist_ok=True)
            for i in range(len(session.Addresses)):
                if period.Graphs[i] is None or len(labels) != period.Graphs[i].shape[1]:
                    continue
                with open(f"{outDir}/{session.Addresses[i].filename()}.csv", "w") as outFile:
                    writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
                    writer.writerow(labels)
                    writer.writerows(period.Graphs[i].tolist())
        else:
            if len(labels) != period.Graphs[0].shape[1]:
                continue
            outDir = f"{testDir}/{kind}/"
            os.makedirs(outDir, exist_ok=True)
            with open(f"{outDir}/{index}.csv", "w") as outFile:
                writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
                writer.writerow(labels)
                writer.writerows(period.Graphs[0].tolist())
    
def LoadDynamicSession(testname:str, run:DynamicRun, kind:str, dynamicPaths:dict[str, list[tuple[float, float, float]]], sessionParameters:SP.Parameters, *,
                       singular:bool = False,
                       labels:list[str] = [ "Time", "Value" ]) -> Session.DynamicSession:
    testDir = GetTestProcessedDir(testname, run)
    inDir   = f"{testDir}/{kind}"
    if not os.path.isdir(inDir):
        return None
    
    source:dict[str, dict[MACAddress, np.ndarray]] = {}
    for periodName in [ d for d in os.listdir(inDir) if ((os.path.isfile(f"{inDir}/{d}") and d.endswith(".csv")) if singular else os.path.isdir(f"{inDir}/{d}")) ]:
        periodName = periodName.removesuffix(".csv") if singular else periodName.removesuffix("/").removesuffix("\\")
        periodDir  = f"{inDir}/{periodName}"

        period:dict[MACAddress, np.ndarray] = {}
        if singular:
            columns:dict[str, list[float]] = { label: [] for label in labels }
            with open(f"{periodDir}.csv", "r") as inFile:
                reader = csv.DictReader(inFile, delimiter=",")
                for row in reader:
                    for key, value in columns.items():
                        value.append(float(row[key]) if key in row else float("nan"))
            period[None] = np.stack(tuple([ columns[label] for label in labels ]), axis=1)
        else:
            for address in [ f for f in os.listdir(periodDir) if os.path.isfile(f"{periodDir}/{f}") and f.endswith(".csv") ]:
                try:
                    address = address.removesuffix(".csv")
                    addr    = MACAddress(address)
                except:
                    continue

                columns:dict[str, list[float]] = { label: [] for label in labels }
                with open(f"{periodDir}/{address}.csv", "r") as inFile:
                    reader = csv.DictReader(inFile, delimiter=",")
                    for row in reader:
                        for key, value in columns.items():
                            value.append(float(row[key]) if key in row else float("nan"))
                period[addr] = np.stack(tuple([ columns[label] for label in labels ]), axis=1)
        source[periodName] = period
    return Session.DynamicSession(dynamicPaths, source, sessionParameters, singular)

def StoreStaticSession(testname:str, run:StaticRun, kind:str, session:Session.StaticSession, *,
                       labels:list[str] = [ "Weight", "Value" ]):
    testDir = GetTestProcessedDir(testname, run)
    for pos, period in session:
        if len(period.Graphs) > 1:
            outDir = f"{testDir}/{kind}/{pos[0]}_{pos[1]}"
            os.makedirs(outDir, exist_ok=True)
            for i in range(len(session.Addresses)):
                if period.Graphs[i] is None or len(labels) != period.Graphs[i].shape[1]:
                    continue
                with open(f"{outDir}/{session.Addresses[i].filename()}.csv", "w") as outFile:
                    writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
                    writer.writerow(labels)
                    writer.writerows(period.Graphs[i].tolist())
        else:
            if len(labels) != period.Graphs[0].shape[1]:
                continue
            outDir = f"{testDir}/{kind}/"
            os.makedirs(outDir, exist_ok=True)
            with open(f"{outDir}/{pos[0]}_{pos[1]}.csv", "w") as outFile:
                writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
                writer.writerow(labels)
                writer.writerows(period.Graphs[0].tolist())

def LoadStaticSession(testname:str, run:StaticRun, kind:str, sessionParameters:SP.Parameters, *,
                       singular:bool = False,
                       labels:list[str] = [ "Weight", "Value" ]) -> Session.StaticSession:
    testDir = GetTestProcessedDir(testname, run)
    inDir   = f"{testDir}/{kind}"
    if not os.path.isdir(inDir):
        return None
    
    source:dict[str, dict[MACAddress, np.ndarray]] = {}
    for periodName in [ d for d in os.listdir(inDir) if ((os.path.isfile(f"{inDir}/{d}") and d.endswith(".csv")) if singular else os.path.isdir(f"{inDir}/{d}")) ]:
        periodName = periodName.removesuffix(".csv") if singular else periodName.removesuffix("/").removesuffix("\\")
        periodDir  = f"{inDir}/{periodName}"

        period:dict[MACAddress, np.ndarray] = {}
        if singular:
            columns:dict[str, list[float]] = { label: [] for label in labels }
            with open(f"{periodDir}.csv", "r") as inFile:
                reader = csv.DictReader(inFile, delimiter=",")
                for row in reader:
                    for key, value in columns.items():
                        value.append(float(row[key]) if key in row else float("nan"))
            period[None] = np.stack(tuple([ columns[label] for label in labels ]), axis=1)
        else:
            for address in [ f for f in os.listdir(periodDir) if os.path.isfile(f"{periodDir}/{f}") and f.endswith(".csv") ]:
                try:
                    address = address.removesuffix(".csv")
                    addr    = MACAddress(address)
                except:
                    continue

                columns:dict[str, list[float]] = { label: [] for label in labels }
                with open(f"{periodDir}/{address}.csv", "r") as inFile:
                    reader = csv.DictReader(inFile, delimiter=",")
                    for row in reader:
                        for key, value in columns.items():
                            value.append(float(row[key]) if key in row else float("nan"))
                period[addr] = np.stack(tuple([ columns[label] for label in labels ]), axis=1)
        source[periodName] = period
    return Session.StaticSession(source, sessionParameters, singular)