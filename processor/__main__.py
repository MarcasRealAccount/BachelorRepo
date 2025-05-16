from collections.abc import Callable
import argparse
import os
import csv
import sys
from pathlib import Path
# Hacky solution to re-use functionality
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

import GradientDescent as GD
import DataParser as DP
import SessionParameters as SP
import Session
from IProcessor import GetProcessorPairs, GetTunableProcessorPairs, GetProcessorTriples, GetTunableProcessorTriples, StaticRun, DynamicRun, Filters, Distances, Positions, StaticErrorFunc, StaticInitErrorFunc, DynamicErrorFunc, DynamicInitErrorFunc
from MACAddress import MACAddress

import Plotting

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import matplotlib.colors
from matplotlib.collections import LineCollection

import time

#
# TODO: Run python3 processor Tunnel2B --skip-dynamic

# TODO: Implement plots
#



# TODO: Calibrating filters seem to produce some bad results. Maybe we shouldn't?
#   Partial fix: Disable parameter limits, albeit it would technically be incorrect.
#   Potential fix: Use sigmoid function to make parameters unbounded during optimization, but bounded in the error function.
# TODO: We could calibrate better by starting at many different random parameter values and seeing which one produces the lowest error.
# TODO: We could process multiple things simultaneously with MultiProcessing. But for now there's no time for that.

def main():
    argParser = argparse.ArgumentParser(
        prog="processor",
        description="Processes and performs all algorithms on the test session")
    argParser.add_argument("--skip-filters",
                           action="extend",
                           nargs="+",
                           default=[],
                           choices=[ filter.name for filter in Filters().GetProcessors()],
                           help="Skip processing of filters")
    argParser.add_argument("--skip-distances",
                           action="extend",
                           nargs="+",
                           default=[],
                           choices=[ distance.name for distance in Distances().GetProcessors() ],
                           help="Skip processing of distance estimators")
    argParser.add_argument("--skip-positions",
                           action="extend",
                           nargs="+",
                           default=[],
                           choices=[ position.name for position in Positions().GetProcessors() ],
                           help="Skip processing of position estimators")
    argParser.add_argument("--skip-bad-crc",
                           action="store_true",
                           default=False,
                           help="Skip processing of bad crc data")
    argParser.add_argument("--skip-only-good-crc",
                           action="store_true",
                           default=False,
                           help="Skip processing of only good crc data")
    argParser.add_argument("--skip-calibrations",
                           action="store_true",
                           default=False,
                           help="Skip calibrations")
    argParser.add_argument("--skip-processing",
                           action="store_true",
                           default=False,
                           help="Skip processing")
    argParser.add_argument("--skip-analysis",
                           action="store_true",
                           default=False,
                           help="Skip analysis")
    argParser.add_argument("--skip-static",
                           action="store_true",
                           default=False,
                           help="Skip processing static data")
    argParser.add_argument("--skip-dynamic",
                           action="store_true",
                           default=False,
                           help="Skip processing dynamic data")
    argParser.add_argument("testname")
    args = argParser.parse_args()

    if (args.skip_dynamic and args.skip_static) or (args.skip_calibrations and args.skip_processing and args.skip_analysis) or (args.skip_bad_crc and args.skip_only_good_crc):
        print("You don't want to process anything? How funny")
        return
    
    testname:str = args.testname
    dir          = f"tests/{testname}/"
    if not os.path.isdir(dir):
        print(f"Test '{testname}' does not exist")
        return
    
    Filters().Skip(args.skip_filters)
    Distances().Skip(args.skip_distances)
    Positions().Skip(args.skip_positions)
    
    sys.stdout.write("Loading Session Parameters")
    sessionParameters = DP.LoadSessionParameters(testname)
    sys.stdout.write(" DONE\n")

    if not args.skip_static:
        ProcessStaticData(testname, args, sessionParameters)

    if not args.skip_dynamic:
        ProcessDynamicData(testname, args, sessionParameters)

def ProcessStaticData(testname:str, args, sessionParameters:SP.Parameters):
    processorPairs        = GetProcessorPairs()
    tunableProcessorPairs = GetTunableProcessorPairs()

    # -----------------------
    #  Step 1. Load raw data
    # -----------------------
    sys.stdout.write("Loading Raw Static Sessions")
    sourceStaticSession = DP.LoadSourceSession(testname, True)
    sys.stdout.write(" DONE\n")

    # ----------------------------
    #  Step 2. Cut up source data
    #     Produce all the runs
    # ----------------------------
    calibrationRuns:list[StaticRun]    = []
    processingRuns:list[StaticRun]     = []
    rawAnalysisRuns:list[StaticRun]    = []
    rawAvgAnalysisRuns:list[StaticRun] = []
    baseAnalysisRuns:list[StaticRun]   = []
    avgAnalysisRuns:list[StaticRun]    = []
    sys.stdout.write("Cut up source data")
    if not args.skip_bad_crc:
        fullSession = Session.StaticSession(sourceStaticSession.WithBadCRC(), sessionParameters)
        if not args.skip_calibrations:
            for pair in tunableProcessorPairs:
                calibrationRuns.append(StaticRun(pair, True, -1.0, 0, fullSession))
        for window in sessionParameters.WindowSizes:
            windows = fullSession.CutUp(window)
            if not args.skip_analysis:
                rawAvgAnalysisRuns.append(StaticRun(None, True, window, -1, windows))
                for pair in processorPairs:
                    avgAnalysisRuns.append(StaticRun(pair, True, window, -1, windows))
            for windowIndex, session in windows:
                # TODO: Producing plots for each individual window is a bit much. So we're just not doing that...
                #if not args.skip_analysis:
                #    rawAnalysisRuns.append(StaticRun(None, True, window, windowIndex, session))
                for pair in processorPairs:
                    if not args.skip_processing:
                        processingRuns.append(StaticRun(pair, True, window, windowIndex, session))
                    #if not args.skip_analysis:
                    #    baseAnalysisRuns.append(StaticRun(pair, True, window, windowIndex, session))

    if not args.skip_only_good_crc:
        fullSession = Session.StaticSession(sourceStaticSession.WithGoodCRC(), sessionParameters)
        if not args.skip_calibrations:
            for pair in tunableProcessorPairs:
                calibrationRuns.append(StaticRun(pair, False, -1.0, 0, fullSession))
        for window in sessionParameters.WindowSizes:
            windows = fullSession.CutUp(window)
            if not args.skip_analysis:
                rawAvgAnalysisRuns.append(StaticRun(None, False, window, -1, windows))
                for pair in processorPairs:
                    avgAnalysisRuns.append(StaticRun(pair, False, window, -1, windows))
            for windowIndex, session in windows:
                # TODO: Producing plots for each individual window is a bit much. So we're just not doing that...
                #if not args.skip_analysis:
                #    rawAnalysisRuns.append(StaticRun(None, False, window, windowIndex, session))
                for pair in processorPairs:
                    if not args.skip_processing:
                        processingRuns.append(StaticRun(pair, False, window, windowIndex, session))
                    #if not args.skip_analysis:
                    #    baseAnalysisRuns.append(StaticRun(pair, False, window, windowIndex, session))
    sys.stdout.write(" DONE\n")

    # --------------------------
    #  Step 3. Run Calibrations
    # --------------------------
    for run in calibrationRuns:
        sys.stdout.write(f"Calibrating {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}'\n")

        prefix:str = ""
        def ProgressFunc(error:float, learningRate:float):
            sys.stdout.write(f"\r{prefix} = {error:14.3f} Error, Learning rate {learningRate:7.3e}         ")

        curDevice = 0
        run.SetupParams(sessionParameters)
        for i in range(len(run.Addresses)):
            curDevice = i
            prefix = f"  {f'{sessionParameters.Beacons[run.Addresses[i]].Name:>6}' if not run.OptimizePosition else ''}"
            sys.stdout.write(f"\r{prefix}")
            params, error = GD.GradientDescentMP(run.Params, StaticErrorFunc, StaticInitErrorFunc,
                                               { "Device": curDevice, "Run": run },
                                               minValues=run.ParamMins, maxValues=run.ParamMaxs,
                                               progress=ProgressFunc, learningRate=1e-9, decay=0.85,
                                               numThreads=32, fastSearch=2048)
            run.Params[...] = params
            sys.stdout.write(f"\r{prefix} = {error:14.3f} Error, Done                          \n")
            if run.OptimizePosition:
                break

        sys.stdout.write(f"  Saving")
        DP.StoreParameters(testname, run, run.ParamsToDict())
        sys.stdout.write(f" DONE\n")

    # ------------------------
    #  Step 4. Run Processors
    # ------------------------
    for run in processingRuns:
        sys.stdout.write(f"Processing {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}', {run.WindowSize} second window size, window {run.WindowIndex}\n")

        sys.stdout.write(f"  Loading Parameters")
        params = DP.LoadParameters(testname, run)
        if len(params) == 0:
            sys.stdout.write(f" FAILED, SKIPPING\n")
            continue
        run.SetupParams(sessionParameters, params)
        sys.stdout.write(f" DONE\n")

        sys.stdout.write(f"  Running Processors")
        distanceResult, positionResult = run.Run(sampleCount=100_000)
        sys.stdout.write(f" DONE\n")

        sys.stdout.write(f"  Storing Results")
        DP.StoreStaticSession(testname, run, "distance", distanceResult, labels=["Weight", "Distance"])
        DP.StoreStaticSession(testname, run, "position", positionResult, labels=["Weight", "X", "Y"])
        sys.stdout.write(f" DONE\n")
    
    heatmapMinX   = min([ min(segment.Left[0], segment.Right[0]) for segment in sessionParameters.Tunnel.Walls ])
    heatmapMinY   = min([ min(segment.Left[1], segment.Right[1]) for segment in sessionParameters.Tunnel.Walls ])
    heatmapMaxX   = max([ max(segment.Left[0], segment.Right[0]) for segment in sessionParameters.Tunnel.Walls ])
    heatmapMaxY   = max([ max(segment.Left[1], segment.Right[1]) for segment in sessionParameters.Tunnel.Walls ])
    heatmapWidth  = heatmapMaxX - heatmapMinX
    heatmapHeight = heatmapMaxY - heatmapMinY

    def PlotRawHeatmap(run:StaticRun, name:str, title:str, func:Callable[[np.ndarray], float], invalidValue:float = 0.0, minValue:float = None, maxValue:float = None):
        allPositions:set[tuple[float, float]] = set()
        if type(run.Session) == list:
            for _, session in run.Session:
                for pos, _ in session:
                    allPositions.add(pos)
        else:
            for pos, _ in run.Session:
                allPositions.add(pos)
        coords    = list(allPositions)
        addresses = list(sessionParameters.Beacons.keys())

        n = len(addresses)

        gridXY = np.array(sessionParameters.Transform(coords, snifferPoint=True))
        valuesRaw:list[np.ndarray] = []
        for i in range(n):
            addr = addresses[i]
            valuesArr:list[float] = []
            for pos in coords:
                if type(run.Session) == list:
                    avgValue:float = 0.0
                    count:int      = 0
                    for _, session in run.Session:
                        if pos in session.Periods:
                            period = session.Periods[pos]
                            if period.Graphs[i] is not None:
                                avgValue += func(period.Graphs[i])
                                count    += 1
                    valuesArr.append((avgValue / count) if count > 0 else invalidValue)
                else:
                    period = run.Session.Periods[pos]
                    if period.Graphs[i] is not None:
                        valuesArr.append(func(period.Graphs[i]))
                    else:
                        valuesArr.append(invalidValue)
            valuesRaw.append(np.array(valuesArr))
        mask   = [ np.isfinite(valuesRaw[i]) for i in range(n) ]
        minVal = minValue if minValue is not None else min([ np.min(valuesRaw[i][mask[i]]) for i in range(n) ])
        maxVal = maxValue if maxValue is not None else max([ np.max(valuesRaw[i][mask[i]]) for i in range(n) ])

        fig, axes = plt.subplots(ncols=n,
                                 figsize=(0.25 + heatmapWidth / heatmapHeight * 9.0 * n, 7.0),
                                 layout="constrained")
        fig.suptitle(f"{title}\n{'Full' if run.BadCRCs else 'Good'}{f', {run.WindowSize} second window size{f', window {run.WindowIndex}' if run.WindowIndex >= 0 else ''}' if run.WindowSize >= 0.0 else ''}")
        for i in range(n):
            addr = addresses[i]
            ax:Axes = axes[i] if n > 1 else axes
            ax.set_title(sessionParameters.Beacons[addr].Name)
            ax.axis("equal")

            mm = Plotting.Heatmap(ax, gridXY, valuesRaw[i], w=5.0,
                                  minX=heatmapMinX, minY=heatmapMinY,
                                  maxX=heatmapMaxX, maxY=heatmapMaxY,
                                  norm=matplotlib.colors.PowerNorm(0.4, vmin=minVal, vmax=maxVal),
                                  zorder=-1)
            Plotting.TunnelGeometry(ax, sessionParameters, c="black", linewidth=2.0, zorder=0)
            Plotting.Beacons(ax, sessionParameters, addr, defaultColor="aqua", selectedColor="lime", labels=False, s=5.0, zorder=1)
            if i == n - 1:
                fig.colorbar(mm, ax=axes if n > 1 else [ axes ])
        plotsDir = DP.GetTestPlotsDir(testname, run)
        os.makedirs(f"{plotsDir}/", exist_ok=True)
        plt.savefig(f"{plotsDir}/{name}.svg", format="svg")
        plt.close()

    def PlotHeatmap(run:StaticRun, session:Session.StaticSession|list[Session.StaticSession], name:str, title:str, func:Callable[[Session.StaticPeriod, np.ndarray, int], float], invalidValue:float = 0.0, minValue:float = None, maxValue:float = None, singular:bool = False):
        allPositions:set[tuple[float, float]] = set()
        if type(session) == list:
            for sess in session:
                for pos, _ in sess:
                    allPositions.add(pos)
        else:
            for pos, _ in session:
                allPositions.add(pos)
        coords    = list(allPositions)
        addresses = list(sessionParameters.Beacons.keys()) if not singular else [ None ]

        n = len(addresses)

        gridXY = np.array(sessionParameters.Transform(coords, snifferPoint=True))
        valuesRaw:list[np.ndarray] = []
        for i in range(n):
            addr = addresses[i]
            valuesArr:list[float] = []
            if type(session) == list:
                for pos in coords:
                    avgValue:float = 0.0
                    count:int      = 0
                    for sess in session:
                        if pos not in sess.Periods:
                            continue
                        if not singular:
                            found = False
                            for j in range(len(sess.Addresses)):
                                if addr == sess.Addresses[j]:
                                    found = True
                                    break
                            if not found:
                                continue
                        else:
                            j = 0

                        period = sess.Periods[pos]
                        if period.Graphs[j] is not None:
                            avgValue += func(period, period.Graphs[j], j)
                            count    += 1
                    valuesArr.append((avgValue / count) if count > 0 else invalidValue)
            else:
                found = False
                for j in range(len(session.Addresses)):
                    if addr == session.Addresses[j]:
                        found = True
                        break
                if not found:
                    for pos in coords:
                        valuesArr.append(invalidValue)
                    continue
                for pos in coords:
                    period = session.Periods[pos]
                    if period[j] is not None:
                        valuesArr.append(func(period, period.Graphs[j], j))
                    else:
                        valuesArr.append(invalidValue)

            valuesRaw.append(np.array(valuesArr))
        mask   = [ np.isfinite(valuesRaw[i]) for i in range(n) ]
        minVal = minValue if minValue is not None else min([ np.min(valuesRaw[i][mask[i]]) for i in range(n) ])
        maxVal = maxValue if maxValue is not None else max([ np.max(valuesRaw[i][mask[i]]) for i in range(n) ])

        fig, axes = plt.subplots(ncols=n,
                                 figsize=(max(0.25 + heatmapWidth / heatmapHeight * 9.0 * n, 4.5), 7.0),
                                 layout="none" if singular else "constrained")
        fig.suptitle(f"{run.GetName()} | {title}\n{'Full' if run.BadCRCs else 'Good'}{f', {run.WindowSize} second window size{f', window {run.WindowIndex}' if run.WindowIndex >= 0 else ''}' if run.WindowSize >= 0.0 else ''}")
        for i in range(n):
            addr = addresses[i]
            ax:Axes = axes[i] if n > 1 else axes
            if not singular:
                ax.set_title(sessionParameters.Beacons[addr].Name)
            ax.axis("equal")

            mm = Plotting.Heatmap(ax, gridXY, valuesRaw[i], w=5.0,
                                  minX=heatmapMinX, minY=heatmapMinY,
                                  maxX=heatmapMaxX, maxY=heatmapMaxY,
                                  norm=matplotlib.colors.PowerNorm(0.4, vmin=minVal, vmax=maxVal),
                                  zorder=-1)
            Plotting.TunnelGeometry(ax, sessionParameters, c="black", linewidth=2.0, zorder=0)
            Plotting.Beacons(ax, sessionParameters, addr, defaultColor="aqua", selectedColor="lime", labels=False, s=5.0, zorder=1)
            if i == n - 1:
                fig.colorbar(mm, ax=axes if n > 1 else [ axes ])
        plotsDir = DP.GetTestPlotsDir(testname, run)
        os.makedirs(f"{plotsDir}/", exist_ok=True)
        plt.savefig(f"{plotsDir}/{name}.svg", format="svg")
        plt.close()

    # def PlotRawHistogram(run:StaticRun, name:str, title:str, func:Callable[[np.ndarray], np.ndarray], binWidth:float, xlabel:str, ylabel:str):
    #     periodItems = list(run.Session.Periods.items())
    #     n = len(run.Session.Addresses)
    #     for pos, period in periodItems:
    #         fig, axes = plt.subplots(ncols=n,
    #                                  figsize=(0.25 + heatmapWidth / heatmapHeight * 9.0 * n, 7.0),
    #                                  layout="constrained", sharey=True)
    #         fig.suptitle(f"{title}\n{pos[0]}, {pos[1]} | {'Full' if run.BadCRCs else 'Good'}{f', {run.WindowSize} second window size, window {run.WindowIndex}' if run.WindowSize >= 0.0 else ''}")
    #         for i in range(n):
    #             addr = run.Session.Addresses[i]
    #             ax:Axes = axes[i] if n > 0 else axes
    #             ax.set_title(sessionParameters.Beacons[addr].Name)
                
    #             data:np.ndarray = None
    #             found = False
    #             for j, addr2 in enumerate(period.Addresses):
    #                 if addr == addr2:
    #                     found = True
    #                     break
    #             if found:
    #                 data = func(period.Graphs[j])

    #             if data is not None:
    #                 ax.hist(data, bins=np.arange(np.min(data), np.max(data), binWidth), orientation="horizontal")
    #             ax.set_xlabel(xlabel)
    #             ax.set_ylabel(ylabel)

    #         plotsDir = DP.GetTestPlotsDir(testname, run)
    #         os.makedirs(f"{plotsDir}/{name}/", exist_ok=True)
    #         plt.savefig(f"{plotsDir}/{name}/{pos[0]}_{pos[1]}.svg", format="svg")
    #         plt.close()

    # --------------------------
    #  Step 5. Run Raw Analysis
    # --------------------------
    for run in rawAnalysisRuns:
        sys.stdout.write(f"Analysing Raw, Run '{'Full' if run.BadCRCs else 'Good'}'")
        if run.WindowSize >= 0.0:
            sys.stdout.write(f", {run.WindowSize} second window size, window {run.WindowIndex}")
        sys.stdout.write("\n")

        # -------------------------------------------
        #  Step 5a. Heatmap of Received Packet Count
        # -------------------------------------------
        PlotRawHeatmap(run, "packetCount", "Received Packet Count", lambda graph: graph.shape[0], invalidValue=0.0, minValue=0.0)
        # -------------------------------
        #  Step 5b. Heatmap of Mean RSSI
        # -------------------------------
        PlotRawHeatmap(run, "RSSImean", "Mean RSSI", lambda graph: np.mean(graph[:,1]), invalidValue=float("nan"))
        # ---------------------------------
        #  Step 5c. Heatmap of RSSI stddev
        # ---------------------------------
        PlotRawHeatmap(run, "RSSIstddev", "RSSI Standard Deviation", lambda graph: np.sqrt(np.var(graph[:,1])), invalidValue=float("nan"))
        # -----------------------------
        #  Step 5d. Histogram of RSSIs
        # -----------------------------
        # PlotRawHistogram(run, "RSSIhist", "RSSI Histogram", lambda graph:graph[:,1], binWidth=1.0, xlabel="Count", ylabel="RSSI")

    # ----------------------------------
    #  Step 6. Run Raw Average Analysis
    # ----------------------------------
    for run in rawAvgAnalysisRuns:
        sys.stdout.write(f"Analysing Raw Average, Run '{'Full' if run.BadCRCs else 'Good'}', {run.WindowSize} second window size\n")
        
        # -------------------------------------------
        #  Step 6a. Heatmap of Received Packet Count
        # -------------------------------------------
        PlotRawHeatmap(run, "avgPacketCount", "Average Received Packet Count", lambda graph: graph.shape[0], invalidValue=0.0, minValue=0.0)
        # -------------------------------
        #  Step 6b. Heatmap of Mean RSSI
        # -------------------------------
        PlotRawHeatmap(run, "avgRSSImean", "Average Mean RSSI", lambda graph: np.mean(graph[:,1]), invalidValue=float("nan"))
        # ---------------------------------
        #  Step 6c. Heatmap of RSSI stddev
        # ---------------------------------
        PlotRawHeatmap(run, "avgRSSIstddev", "Average RSSI Standard Deviation", lambda graph: np.sqrt(np.var(graph[:,1])), invalidValue=float("nan"))

    # ---------------------------
    #  Step 7. Run Base Analysis
    # ---------------------------
    for run in baseAnalysisRuns:
        sys.stdout.write(f"Analysing {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}', {run.WindowSize} second window size, window {run.WindowIndex}\n")

        distanceSession = DP.LoadStaticSession(testname, run, "distance", sessionParameters, labels=["Weight", "Distance"])
        positionSession = DP.LoadStaticSession(testname, run, "position", sessionParameters, singular=True, labels=["Weight", "X", "Y"])

        # -----------------------------------
        #  Step 7a. Heatmap of Mean Distance
        # -----------------------------------
        PlotHeatmap(run, distanceSession, "distMean", "Mean Distance", lambda period,graph: np.mean(graph[:,1]), invalidValue=float("nan"))
        # -------------------------------------
        #  Step 7b. Heatmap of Distance stddev
        # -------------------------------------
        PlotHeatmap(run, distanceSession, "distStddev", "Distance Standard Deviation", lambda period,graph: np.sqrt(np.var(graph[:,1])), invalidValue=float("nan"))
        # ---------------------------------
        #  Step 7c. Histogram of Distances
        # ---------------------------------
        #PlotHistogram(run, distanceSession, "distHist", "Distance Histogram", lambda graph: graph[:,1], binWidth=0.25, xlabel="Count", ylabel="Distance")

        # -------------------------------------------
        #  Step 7d. Heatmap of Mean Positional Error
        # -------------------------------------------
        PlotHeatmap(run, positionSession, "posErrorMean", "Mean Position Error", lambda period,graph: np.mean(np.linalg.norm(graph[:,1:3] - np.array(period.TruePos)[None,:], axis=1)), invalidValue=float("nan"), singular=True)
        # ---------------------------------------------
        #  Step 7e. Heatmap of Positional Error stddev
        # ---------------------------------------------
        PlotHeatmap(run, positionSession, "posErrorStddev", "Position Error Standard Deviation", lambda period,graph: np.sqrt(np.var(np.linalg.norm(graph[:,1:3] - np.array(period.TruePos)[None,:], axis=1))), invalidValue=float("nan"), singular=True)

    # ------------------------------
    #  Step 8. Run Average Analysis
    # ------------------------------
    for run in avgAnalysisRuns:
        sys.stdout.write(f"Analysing {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}', {run.WindowSize} second window size\n")

        distanceSessions:list[Session.StaticSession] = []
        positionSessions:list[Session.StaticSession] = []

        for i, _ in run.Session:
            run.WindowIndex = i
            distanceSession = DP.LoadStaticSession(testname, run, "distance", sessionParameters, labels=["Weight", "Distance"])
            positionSession = DP.LoadStaticSession(testname, run, "position", sessionParameters, singular=True, labels=["Weight", "X", "Y"])
            if distanceSession is not None:
                distanceSessions.append(distanceSession)
            if positionSession is not None:
                positionSessions.append(positionSession)

        run.WindowIndex = -1

        analysisDir = DP.GetTestAnalysisDir(testname, run)
        os.makedirs(f"{analysisDir}/", exist_ok=True)

        avgDistMses:dict[MACAddress, float] = {}
        for session in distanceSessions:
            for i in range(len(session.Addresses)):
                avgDistMses[session.Addresses[i]] = 0.0
            for i in range(len(session.Addresses)):
                avgMse = 0.0
                for pos, period in session:
                    if period.Graphs[i] is not None:
                        avgMse += np.sum(period.Graphs[i][:,0] * (period.TrueDistances[i] - period.Graphs[i][:,1])**2)
                avgDistMses[session.Addresses[i]] += avgMse
        with open(f"{analysisDir}/distMSE.csv", "w") as outFile:
            writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
            writer.writerow([ "Name", "MSE" ])
            for addr, mse in avgDistMses.items():
                writer.writerow([ sessionParameters.Beacons[addr].Name, mse / len(distanceSessions) ])
            
        avgPosMse:float = 0.0
        for session in positionSessions:
            for pos, period in session:
                if period.Graphs[0] is not None:
                    avgPosMse += np.sum(period.Graphs[0][:,0] * np.linalg.norm(np.array(period.TruePos) - period.Graphs[0][:,1:3])**2)
        with open(f"{analysisDir}/posMSE.csv", "w") as outFile:
            writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
            writer.writerow([ "MSE" ])
            writer.writerow([ avgPosMse / len(positionSessions) ])

        # -----------------------------------
        #  Step 8aa. Heatmap of Mean Distance
        # -----------------------------------
        PlotHeatmap(run, distanceSessions, "avgDistMean", "Average Mean Distance", lambda period,graph,j: np.mean(graph[:,1]), invalidValue=float("nan"))
        # -------------------------------------
        #  Step 8ab. Heatmap of Distance stddev
        # -------------------------------------
        PlotHeatmap(run, distanceSessions, "avgDistStddev", "Average Distance Standard Deviation", lambda period,graph,j: np.sqrt(np.var(graph[:,1])), invalidValue=float("nan"))
        # -----------------------------------
        #  Step 8ac. Heatmap of Mean Distance
        # -----------------------------------
        PlotHeatmap(run, distanceSessions, "avgDistErrorMean", "Average Mean Distance Error", lambda period,graph,j: np.mean(np.abs(period.TrueDistances[j] - graph[:,1])), invalidValue=float("nan"))
        # -------------------------------------
        #  Step 8ad. Heatmap of Distance stddev
        # -------------------------------------
        PlotHeatmap(run, distanceSessions, "avgDistErrorStddev", "Average Distance Error Standard Deviation", lambda period,graph,j: np.sqrt(np.var(np.abs(period.TrueDistances[j] - graph[:,1]))), invalidValue=float("nan"))
        
        # -------------------------------------------
        #  Step 8ba. Heatmap of Mean Positional Error
        # -------------------------------------------
        PlotHeatmap(run, positionSessions, "avgPosErrorMean", "Average Mean Position Error", lambda period,graph,j: np.mean(np.linalg.norm(graph[:,1:3] - np.array(period.TruePos)[None,:], axis=1)), invalidValue=float("nan"), singular=True)
        # ---------------------------------------------
        #  Step 8bb. Heatmap of Positional Error stddev
        # ---------------------------------------------
        PlotHeatmap(run, positionSessions, "avgPosErrorStddev", "Average Position Error Standard Deviation", lambda period,graph,j: np.sqrt(np.var(np.linalg.norm(graph[:,1:3] - np.array(period.TruePos)[None,:], axis=1))), invalidValue=float("nan"), singular=True)

        # ---------------------------------
        #  Step 8ca. Heatmap of Mean DeltaX
        # ---------------------------------
        PlotHeatmap(run, positionSessions, "avgDeltaXMean", "Average Mean Delta X", lambda period,graph,j: np.mean(graph[:,1] - period.TruePos[0]), invalidValue=float("nan"), singular=True)
        # ---------------------------------
        #  Step 8cb. Heatmap of Mean DeltaY
        # ---------------------------------
        PlotHeatmap(run, positionSessions, "avgDeltaYMean", "Average Mean Delta Y", lambda period,graph,j: np.mean(graph[:,2] - period.TruePos[1]), invalidValue=float("nan"), singular=True)
        # -----------------------------------
        #  Step 8cc. Heatmap of DeltaX stddev
        # -----------------------------------
        PlotHeatmap(run, positionSessions, "avgDeltaXStddev", "Average Delta X Standard Deviation", lambda period,graph,j: np.sqrt(np.var(graph[:,1] - period.TruePos[0])), invalidValue=float("nan"), singular=True)
        # -----------------------------------
        #  Step 8cd. Heatmap of DeltaY stddev
        # -----------------------------------
        PlotHeatmap(run, positionSessions, "avgDeltaYStddev", "Average Delta Y Standard Deviation", lambda period,graph,j: np.sqrt(np.var(graph[:,2] - period.TruePos[1])), invalidValue=float("nan"), singular=True)
        
def ProcessDynamicData(testname:str, args, sessionParameters:SP.Parameters):
    processorTriples        = GetProcessorTriples()
    tunableProcessorTriples = GetTunableProcessorTriples()

    # -----------------------
    #  Step 1. Load raw data
    # -----------------------
    sys.stdout.write("Loading Raw Dynamic Sessions")
    sourceDynamicSession = DP.LoadSourceSession(testname, False)
    sourceDynamicPaths   = DP.LoadDynamicPaths(testname)
    sys.stdout.write(" DONE\n")

    # ----------------------------
    #  Step 2. Cut up source data
    #     Produce all the runs
    # ----------------------------
    calibrationRuns:list[DynamicRun]  = []
    processingRuns:list[DynamicRun]   = []
    rawAnalysisRuns:list[DynamicRun]  = []
    baseAnalysisRuns:list[DynamicRun] = []
    sys.stdout.write("Cut up source data")
    if not args.skip_bad_crc:
        fullSession = Session.DynamicSession(sourceDynamicPaths, sourceDynamicSession.WithBadCRC(), sessionParameters)
        if not args.skip_analysis:
            rawAnalysisRuns.append(DynamicRun(None, True, fullSession))
        if not args.skip_calibrations:
            for triple in tunableProcessorTriples:
                calibrationRuns.append(DynamicRun(triple, True, fullSession))
        if not args.skip_processing:
            for triple in processorTriples:
                processingRuns.append(DynamicRun(triple, True, fullSession))
        if not args.skip_analysis:
            for triple in processorTriples:
                baseAnalysisRuns.append(DynamicRun(triple, True, fullSession))
    if not args.skip_only_good_crc:
        fullSession = Session.DynamicSession(sourceDynamicPaths, sourceDynamicSession.WithGoodCRC(), sessionParameters)
        if not args.skip_analysis:
            rawAnalysisRuns.append(DynamicRun(None, False, fullSession))
        if not args.skip_calibrations:
            for triple in tunableProcessorTriples:
                calibrationRuns.append(DynamicRun(triple, False, fullSession))
        if not args.skip_processing:
            for triple in processorTriples:
                processingRuns.append(DynamicRun(triple, False, fullSession))
        if not args.skip_analysis:
            for triple in processorTriples:
                baseAnalysisRuns.append(DynamicRun(triple, False, fullSession))
    sys.stdout.write(" DONE\n")

    # --------------------------
    #  Step 3. Run Calibrations
    # --------------------------
    for run in calibrationRuns:
        sys.stdout.write(f"Calibrating {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}'\n")

        prefix:str = ""
        def ProgressFunc(error:float, learningRate:float):
            sys.stdout.write(f"\r{prefix} = {error:14.3f} Error, Learning rate {learningRate:7.3e}         ")

        curDevice = 0
        
        run.SetupParams(sessionParameters, params)
        for i in range(len(run.Addresses)):
            curDevice = i
            prefix = f"  {f'{sessionParameters.Beacons[run.Addresses[i]].Name:>6}' if not run.OptimizePosition else ''} "
            sys.stdout.write(f"\r{prefix}")
            params, error = GD.GradientDescentMP(run.Params, DynamicErrorFunc, DynamicInitErrorFunc,
                                               { "Device": curDevice, "Run": run },
                                               minValues=run.ParamMins, maxValues=run.ParamMaxs,
                                               progress=ProgressFunc, learningRate=1e-9, decay=0.75,
                                               maxIterations=250,
                                               numThreads=32, fastSearch=4096)
            run.Params[...] = params
            sys.stdout.write(f"\r{prefix} = {error:14.3f} Error, Done                          \n")
            if run.OptimizePosition:
                break

        sys.stdout.write(f"  Saving")
        DP.StoreParameters(testname, run, run.ParamsToDict())
        sys.stdout.write(f" DONE\n")

    # ------------------------
    #  Step 4. Run Processors
    # ------------------------
    for run in processingRuns:
        sys.stdout.write(f"Processing {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}'\n")

        sys.stdout.write(f"  Loading Parameters")
        params = DP.LoadParameters(testname, run)
        if len(params) == 0:
            sys.stdout.write(f" FAILED, SKIPPING\n")
            continue
        run.SetupParams(sessionParameters, params)
        sys.stdout.write(f" DONE\n")

        sys.stdout.write(f"  Running Processors")
        filterResult, distanceResult, positionResult = run.Run()
        sys.stdout.write(f" DONE\n")

        sys.stdout.write(f"  Storing Results")
        DP.StoreDynamicSession(testname, run, "filter", filterResult, labels=["Time", "RSSI"])
        DP.StoreDynamicSession(testname, run, "distance", distanceResult, labels=["Time", "Distance"])
        DP.StoreDynamicSession(testname, run, "position", positionResult, labels=["Time", "X", "Y"])
        sys.stdout.write(f" DONE\n")
    
    tunnelMinX   = min([ min(segment.Left[0], segment.Right[0]) for segment in sessionParameters.Tunnel.Walls ])
    tunnelMinY   = min([ min(segment.Left[1], segment.Right[1]) for segment in sessionParameters.Tunnel.Walls ])
    tunnelMaxX   = max([ max(segment.Left[0], segment.Right[0]) for segment in sessionParameters.Tunnel.Walls ])
    tunnelMaxY   = max([ max(segment.Left[1], segment.Right[1]) for segment in sessionParameters.Tunnel.Walls ])
    tunnelWidth  = tunnelMaxX - tunnelMinX
    tunnelHeight = tunnelMaxY - tunnelMinY

    def PlotGraphs(run:StaticRun, session:Session.DynamicSession, name:str, title:str, func:Callable[[Session.DynamicPeriod, np.ndarray, int], np.ndarray], singular:bool = False, yLabel:str = "Value", showTrueDistance:bool = False):
        plotsDir = DP.GetTestPlotsDir(testname, run)
        os.makedirs(f"{plotsDir}/", exist_ok=True)
        for index, period in session:
            fig, ax = plt.subplots(figsize=(14.0, 7.0),
                                layout="constrained")
            fig.suptitle(f"{run.GetName()} | {title} | Period {index}\n{'Full' if run.BadCRCs else 'Good'}")
            ax.set_xlabel("Time")
            ax.set_ylabel(yLabel)
            
            for i, graph in enumerate(period.Graphs):
                color = ax._get_lines.get_next_color()
                if not singular and showTrueDistance:
                    ax.plot(period.TrueDistances[i][:,0], period.TrueDistances[i][:,1], color=color, linestyle="--", label=f"{sessionParameters.Beacons[session.Addresses[i]].Name} Expected")
                if graph is None:
                    continue

                yValues = func(period, graph, i)
                if yValues is None:
                    continue

                segments:list[list[tuple[float, float]]] = []
                for j in range(graph.shape[0]):
                    if len(segments) == 0 or graph[j,0] >= segments[-1][-1][0] + 1.2:
                        segments.append([ (graph[j,0], yValues[j]) ])
                    else:
                        segments[-1].append((graph[j,0], yValues[j]))

                offsets = [ segment[0] for segment in segments if len(segment) == 1 ]
                lines   = LineCollection([ segment for segment in segments if len(segment) > 1 ], linewidths=1.0, color=color, label=sessionParameters.Beacons[session.Addresses[i]].Name if not singular else "")
                ax.add_collection(lines)
                ax.scatter([ offset[0] for offset in offsets ], [ offset[1] for offset in offsets ], color=color)
            if not singular:
                ax.legend(loc="upper left")

            plt.savefig(f"{plotsDir}/{name}_{index}.svg", format="svg")
            plt.close()

    def PlotPath(run:StaticRun, session:Session.DynamicSession, name:str, title:str, func:Callable[[Session.DynamicPeriod, np.ndarray, int], np.ndarray]):
        plotsDir = DP.GetTestPlotsDir(testname, run)
        os.makedirs(f"{plotsDir}/", exist_ok=True)
        for index, period in session:
            fig, ax = plt.subplots(figsize=(7.0, tunnelHeight / tunnelWidth * 3),
                                layout="constrained")
            ax.axis("equal")
            fig.suptitle(f"{run.GetName()} | {title} | Period {index}\n{'Full' if run.BadCRCs else 'Good'}")

            Plotting.TunnelGeometry(ax, sessionParameters, c="black", linewidth=2.0, zorder=0)
            Plotting.Beacons(ax, sessionParameters, None, defaultColor="aqua", selectedColor="lime", labels=False, s=5.0, zorder=1)

            ax.plot(period.TruePos[:,1], period.TruePos[:,2], color="blue", linestyle="--", label="Expected path")

            if len(period.Graphs) > 0 and period.Graphs[0] is not None:
                positions = func(period, period.Graphs[0], 0)
                if positions is not None:
                    ax.scatter(positions[:,0], positions[:,1], s=3.0, color="red", label="Estimated path")
            
            ax.legend(loc="upper left")

            plt.savefig(f"{plotsDir}/{name}_{index}.svg", format="svg")
            plt.close()

    # --------------------------
    #  Step 5. Run Raw Analysis
    # --------------------------
    for run in rawAnalysisRuns:
        sys.stdout.write(f"Analysing Raw, Run '{'Full' if run.BadCRCs else 'Good'}'\n")

        # -----------------------------------
        #  Step 5a. Graph of Raw RSSI values
        # -----------------------------------
        PlotGraphs(run, run.Session, "rssiRaw", "RSSI", lambda period,graph,i: graph[:,1], yLabel="RSSI")
        # --------------------
        #  Step 5b. Plot Path
        # --------------------
        PlotPath(run, run.Session, "path", "Expected Path", lambda period,graph,i: None)

    # ---------------------------
    #  Step 6. Run Base Analysis
    # ---------------------------
    beaconAddresses = list(sessionParameters.Beacons.keys())

    distanceMSEsFull:list[tuple[DynamicRun, int, list[float]]] = []
    distanceMSEsGood:list[tuple[DynamicRun, int, list[float]]] = []
    posMSEsFull:list[tuple[DynamicRun, int, float]]            = []
    posMSEsGood:list[tuple[DynamicRun, int, float]]            = []
    for run in baseAnalysisRuns:
        sys.stdout.write(f"Analysing {run.GetName()}, Run '{'Full' if run.BadCRCs else 'Good'}'\n")

        filterSession   = DP.LoadDynamicSession(testname, run, "filter", sourceDynamicPaths, sessionParameters, labels=["Time", "RSSI"])
        distanceSession = DP.LoadDynamicSession(testname, run, "distance", sourceDynamicPaths, sessionParameters, labels=["Time", "Distance"])
        positionSession = DP.LoadDynamicSession(testname, run, "position", sourceDynamicPaths, sessionParameters, singular=True, labels=["Time", "X", "Y"])

        # -------------------------
        #  Step 6a. Calculate MSEs
        # -------------------------
        for index, period in distanceSession:
            mses:list[float] = []
            for addr in beaconAddresses:
                found = False
                for i, addr2 in enumerate(distanceSession.Addresses):
                    if addr == addr2:
                        found = True
                        break
                if not found or period.Graphs[i] is None:
                    mses.append(float("nan"))
                    continue
                graph             = period.Graphs[i]
                indices           = np.searchsorted(period.TrueDistances[i][:,0], graph[:,0], side="right") - 1
                expectedDistances = period.TrueDistances[i][indices,1]
                mses.append(np.sum((expectedDistances - graph[:,1])**2))
            if run.BadCRCs:
                distanceMSEsFull.append((run, index, mses))
            else:
                distanceMSEsGood.append((run, index, mses))
        for index, period in positionSession:
            indices           = np.searchsorted(period.TruePos[:,0], period.Graphs[0][:,0], side="right") - 1
            expectedPositions = period.TruePos[indices,1:3]
            if run.BadCRCs:
                posMSEsFull.append((run, index, np.sum(np.linalg.norm(expectedPositions - period.Graphs[0][:,1:3], axis=1)**2)))
            else:
                posMSEsGood.append((run, index, np.sum(np.linalg.norm(expectedPositions - period.Graphs[0][:,1:3], axis=1)**2)))
                
        # ---------------------------------------
        #  Step 6b. Graph of Filtered RSSI values
        # ---------------------------------------
        PlotGraphs(run, filterSession, "rssiFiltered", "Filtered RSSI", lambda period,graph,i: graph[:,1], yLabel="RSSI")
        
        # -----------------------------------
        #  Step 6ca. Graph of Distance values
        # -----------------------------------
        PlotGraphs(run, distanceSession, "distance", "Distance", lambda period,graph,i: graph[:,1], yLabel="Distance [m]", showTrueDistance=True)
        # ----------------------------------
        #  Step 6cb. Graph of Distance error
        # ----------------------------------
        def DistErr(period:Session.DynamicPeriod, graph:np.ndarray, i:int) -> np.ndarray:
            indices           = np.searchsorted(period.TrueDistances[i][:,0], graph[:,0], side="right") - 1
            expectedDistances = period.TrueDistances[i][indices,1]
            return expectedDistances - graph[:,1]
        PlotGraphs(run, distanceSession, "distanceErr", "Distance Error", DistErr, yLabel="Distance [m]")

        # -------------------------------
        #  Step 6da. Plot Estimated Path
        # -------------------------------
        PlotPath(run, positionSession, "path", "Estimated Path", lambda period,graph,i: graph[:,1:3])

    if len(distanceMSEsFull) > 0:
        analysisDir = DP.GetTestAnalysisRootDir(testname, False, True)
        os.makedirs(analysisDir, exist_ok=True)
        with open(f"{analysisDir}/distMSEs.csv", "w") as outFile:
            writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
            header = [ "Processors", "Period" ]
            header.extend([ sessionParameters.Beacons[addr].Name for addr in beaconAddresses ])
            writer.writerow(header)
            for run, period, mses in distanceMSEsFull:
                row = [ run.GetName(), period ]
                row.extend(mses)
                writer.writerow(row)
    if len(distanceMSEsGood) > 0:
        analysisDir = DP.GetTestAnalysisRootDir(testname, False, False)
        os.makedirs(analysisDir, exist_ok=True)
        with open(f"{analysisDir}/distMSEs.csv", "w") as outFile:
            writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
            header = [ "Processors", "Period" ]
            header.extend([ sessionParameters.Beacons[addr].Name for addr in beaconAddresses ])
            writer.writerow(header)
            for run, period, mses in distanceMSEsGood:
                row = [ run.GetName(), period ]
                row.extend(mses)
                writer.writerow(row)
    if len(posMSEsFull) > 0:
        analysisDir = DP.GetTestAnalysisRootDir(testname, False, True)
        os.makedirs(analysisDir, exist_ok=True)
        with open(f"{analysisDir}/posMSEs.csv", "w") as outFile:
            writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
            writer.writerow([ "Processors", "Period", "MSE" ])
            for run, period, mse in posMSEsFull:
                writer.writerow([ run.GetName(), period, mse ])
    if len(posMSEsGood) > 0:
        analysisDir = DP.GetTestAnalysisRootDir(testname, False, False)
        os.makedirs(analysisDir, exist_ok=True)
        with open(f"{analysisDir}/posMSEs.csv", "w") as outFile:
            writer = csv.writer(outFile, delimiter=",", lineterminator="\n")
            writer.writerow([ "Processors", "Period", "MSE" ])
            for run, period, mse in posMSEsGood:
                writer.writerow([ run.GetName(), period, mse ])

if __name__ == "__main__":
    main()