import argparse
import time
import sys
import traceback
from pathlib import Path
# Hacky solution to re-use functionalioty
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from Figures.RSSIFigure import RSSIFigure
from UI.Figures import Figures

from API.SnifferInst import SnifferInst

import matplotlib

def snifferUpdate():
	snifferInst = SnifferInst()
	if snifferInst.IsAlive():
		packets = snifferInst.sniffer.getPackets()
		if len(packets) > 0:
			rssiFigure:RSSIFigure = Figures().GetFigure("RSSI")
			for packet in packets:
				rssiFigure.addPacket(packet)

def main():
	argParser = argparse.ArgumentParser(
		prog="nRF Sniffer",
		description="nRF Sniffer for BLE packets, does inplace calibration")
	argParser.add_argument("-u", "--update-speed",
						   action="store",
						   type=int,
						   default=1,
						   help="How many frames per update")
	argParser.add_argument("-f", "--frame-rate",
						   action="store",
						   type=int,
						   default=100,
						   help="How many frames per second")
	argParser.add_argument("testname")
	# TODO: Implement headless mode
	# argParser.add_argument("--headless", help="Run headless, no GUI, only simplistic console commands")
	args = argParser.parse_args()

	matplotlib.rcParams["toolbar"] = "None"
	matplotlib.rcParams["backend"] = "qtagg" # QtAgg seems to be running better than TkAgg

	figures = Figures()
	figures.OnUpdate(snifferUpdate)
	figures.GetFigure("RSSI").setTestname(args.testname)
	figures.OpenFigure("RSSI")

	updateSpeed:int = max(args.update_speed, 1)
	frameRate:int   = max(args.frame_rate, 1)

	sleepTime:float = 1.0 / frameRate
	
	try:
		curFrame:int = 0
		while figures.IsAlive():
			if curFrame % updateSpeed == 0:
				figures.Update()
			figures.Present()
			curFrame += 1
			time.sleep(sleepTime) # Wait for 1/framerate ms, because there is no way to explicitly wait for UI events from all figures at the same time afaik...
	except BaseException as e:
		traceback.print_exception(e)

	figures.Close()
	SnifferInst().Close()

if __name__ == "__main__":
	main()