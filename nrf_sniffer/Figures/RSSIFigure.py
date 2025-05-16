from UI.Figure import Figure
from UI.Elements.Rectangle import Rectangle
from UI.Elements.Button import Button
from UI.Elements.Text import Text
from UI.Elements.TextField import TextField
from UI.Elements.Slider import Slider
from API.Sniffer import CAPTURE_MODE_STATIC, CAPTURE_MODE_DYNAMIC, CAPTURE_MODE_TXTS
from API.SnifferIO import SnifferMessage
from API.SnifferInst import SnifferInst

from timer import time_ns
from MACAddress import MACAddress

from matplotlib.transforms import Affine2D
from matplotlib.collections import LineCollection

import numpy as np

class DeviceGraph:
	def __init__(self, address:MACAddress):
		self.address  = address
		self.name     = str(self.address)
		self.fullName = False

		rssiFigure:RSSIFigure = Figures().GetFigure("RSSI")
		transform = rssiFigure.timeTransform + rssiFigure.graphAx.transData

		color = rssiFigure.graphAx._get_lines.get_next_color()
		self.lines = LineCollection([], linewidths=0.5, color=color, label=self.name, transform=transform)
		self.points = rssiFigure.graphAx.scatter([], [], color=color, transform=transform)
		self.badcrc = rssiFigure.graphAx.scatter([], [], color=color, marker="x", transform=transform)
		rssiFigure.graphAx.add_collection(self.lines)

		self.updateTick:int = 0
		self.changed        = False

		# List of continous data points
		self.segments:list[list[tuple[int, int]]] = []
		self.badCRCpoints:list[tuple[int, int]]   = []

		self.curRSSI:int = 0

	def __del__(self):
		self.remove()

	def setName(self, name:str):
		self.name     = name
		self.fullName = True
		self.updateLabel()
	
	def setShortName(self, shortname:str):
		if self.fullName:
			return
		self.name     = shortname
		self.fullName = False
		self.updateLabel()

	def updateLabel(self):
		self.lines.set_label(f"{self.name} ({self.address.value[-1]:02X}) {self.curRSSI}")

	def addSample(self, time:int, rssi:int):
		self.curRSSI = rssi
		if len(self.segments) == 0 or time >= self.segments[-1][-1][0] + 1_200_000: # 1.2 seconds after previous sample
			self.segments.append([ (time, rssi) ]) # Start new line segment
		else:
			self.segments[-1].append((time, rssi))
		self.changed = True
		self.updateLabel()
	
	def addBadCRCSample(self, time:int, rssi:int):
		self.badCRCpoints.append((time, rssi))
		self.addSample(time, rssi)

	def update(self, curTime:int):
		self.updateTick += 1
		if self.updateTick >= 10: # GUI updates at under 100Hz, so we just cull old data once every 10th update (~10Hz)
			self.updateTick = 0

			minTime           = curTime - 60_000_000 # 60 seconds back
			self.segments     = [ segment for segment in ([ [ vertex for vertex in segment if vertex[0] >= minTime ] for segment in self.segments ]) if len(segment) > 0 ]
			self.badCRCpoints = [ point for point in self.badCRCpoints if point[0] >= minTime ]
			self.changed      = True

		if self.changed:
			self.changed = False
			offsets      = [ segment[0] for segment in self.segments if len(segment) == 1 ]
			self.points.set_offsets(offsets if len(offsets) > 0 else np.empty((0, 2)))
			self.lines.set_segments([ segment for segment in self.segments if len(segment) > 1 ])
			self.badcrc.set_offsets(self.badCRCpoints if len(self.badCRCpoints) > 0 else np.empty((0, 2)))

	def remove(self):
		self.lines.remove()
		self.lines = None
		self.points.remove()
		self.points = None
		self.badcrc.remove()
		self.badcrc = None

class RSSIFigure(Figure):
	"""
	RSSI Figure presents RSSI plot and capturing related buttons and settings.
	"""

	def __init__(self):
		super().__init__("RSSI", daemon=False)

		self.captureMode = CAPTURE_MODE_STATIC
		self.channel     = 37
		self.testname    = "Test"

		self.reconnectTime:int = time_ns() # Try to connect as soon as the process starts up
		self.captureTime:int   = None

		self.cpuTime:int    = time_ns()
		self.redrawTime:int = self.cpuTime
		self.lostTime:int   = 0

		self.redrawGraph        = False
		self.curSnifferTime:int = 0

		self.deviceGraphs:dict[MACAddress, DeviceGraph] = {}
		self.timeTransform:Affine2D = Affine2D()

	def setTestname(self, testname:str):
		self.testname = testname
		self.suffix   = self.testname

	def addPacket(self, packet:SnifferMessage):
		if packet.bleAdvAddress not in self.deviceGraphs:
			self.deviceGraphs[packet.bleAdvAddress] = DeviceGraph(packet.bleAdvAddress)
		graph = self.deviceGraphs[packet.bleAdvAddress]

		# NOTE: This only fixes drawing the realtime graph, if the sniffer is either disconnected or somehow dies it should stop capturing, which we can either take as valid or as cancelled.
		time = packet.timestamp + self.lostTime
		if packet.bleCRCOK:
			graph.addSample(time, packet.bleRSSI)
			if packet.bleHasName:
				if packet.bleName is not None:
					graph.setName(packet.bleName)
				else:
					graph.setShortName(packet.bleShortName)
		else:
			graph.addBadCRCSample(time, packet.bleRSSI)
		if time > self.curSnifferTime:
			self.curSnifferTime = time
			self.cpuTime        = time_ns()
			#self.redrawTime     = self.cpuTime + 10_000_000 # Redraw in 10ms

	def _onCaptureModeChange(self):
		self.captureMode = (self.captureMode + 1) % len(CAPTURE_MODE_TXTS)
		self.captureModeBtn.set_label(CAPTURE_MODE_TXTS[self.captureMode])

		if self.captureMode == CAPTURE_MODE_STATIC:
			self.captureTimeSlider.set_visible(True)
			self.captureXTextField.set_visible(True)
			self.captureXTextFieldLabel.set_visible(True)
			self.captureYTextField.set_visible(True)
			self.captureYTextFieldLabel.set_visible(True)
			self.captureIndexTextField.set_visible(False)
			self.captureIndexTextFieldLabel.set_visible(False)
		else:
			self.captureTimeSlider.set_visible(False)
			self.captureXTextField.set_visible(False)
			self.captureXTextFieldLabel.set_visible(False)
			self.captureYTextField.set_visible(False)
			self.captureYTextFieldLabel.set_visible(False)
			self.captureIndexTextField.set_visible(True)
			self.captureIndexTextFieldLabel.set_visible(True)
		self.onResize()

	def _onChannelChange(self, newValue):
		self.channel = int(newValue)
		snifferInst = SnifferInst()
		if snifferInst.IsAlive():
			snifferInst.sniffer.sendAdvChannelHopSequence([ self.channel ])

	def _onClickCapture(self):
		if self.captureTime is not None:
			# We're currently capturing, should we end or cancel?
			if self.captureMode == CAPTURE_MODE_DYNAMIC:
				# End
				totalTime        = time_ns() - self.captureTime
				self.captureTime = None
				self.captureTimeLabel.set_text(f"Done ({totalTime * 1e-9:.2f} seconds captured)")
				self._onCaptureEnd()
			elif self.captureMode == CAPTURE_MODE_STATIC:
				# Cancel
				self.captureTime = None
				self.captureTimeLabel.set_text("Cancelled")
				self._onCaptureCancelled()
			return
		snifferInst = SnifferInst()
		if not snifferInst.IsAlive() or not snifferInst.sniffer.startCapturing(
			testName=self.testname,
			captureName=(f"{self.captureXTextField.value:.2f}_{self.captureYTextField.value:.2f}" if self.captureMode == CAPTURE_MODE_STATIC else str(self.captureIndexTextField.value)),
			captureMode=self.captureMode):
			return

		self.captureTime = time_ns() + self.captureTimeSlider.val * 1_000_000_000 if self.captureMode == CAPTURE_MODE_STATIC else time_ns()
		self.captureModeBtn.disable()
		self.channelSlider.disable()
		self.captureTimeSlider.disable()
		self.captureXTextField.disable()
		self.captureYTextField.disable()
		self.captureIndexTextField.disable()
		self.captureBtn.set_label("Cancel Capture" if self.captureMode == CAPTURE_MODE_STATIC else "End Capture")

	def _onCaptureEnd(self):
		snifferInst = SnifferInst()
		if snifferInst.IsAlive():
			if snifferInst.sniffer.stopCapturing():
				print("BUG!!! Idk what to do here. Sniffer stopCapturing should always work!!!")
		self.captureBtn.set_label("Start Capture")
		self.captureModeBtn.enable()
		self.channelSlider.enable()
		self.captureTimeSlider.enable()
		self.captureXTextField.enable()
		self.captureYTextField.enable()
		self.captureIndexTextField.enable()

		if self.captureMode == CAPTURE_MODE_STATIC:
			if (int(self.captureYTextField.value) % 2) == 0:
				if int(self.captureXTextField.value) == 2:
					self.captureYTextField.set_value(self.captureYTextField.value + 1.0)
				else:
					self.captureXTextField.set_value(self.captureXTextField.value + 1.0)
			else:
				if int(self.captureXTextField.value) == 0:
					self.captureYTextField.set_value(self.captureYTextField.value + 1.0)
				else:
					self.captureXTextField.set_value(self.captureXTextField.value - 1.0)
		elif self.captureMode == CAPTURE_MODE_DYNAMIC:
			self.captureIndexTextField.set_value(self.captureIndexTextField.value + 1)

	def _onCaptureCancelled(self):
		snifferInst = SnifferInst()
		if snifferInst.IsAlive():
			if snifferInst.sniffer.cancelCapturing():
				print("BUG!!! Idk what to do here. Sniffer cancelCapturing should always work!!!")
		self.captureBtn.set_label("Start Capture")
		self.captureModeBtn.enable()
		self.channelSlider.enable()
		self.captureTimeSlider.enable()
		self.captureXTextField.enable()
		self.captureYTextField.enable()
		self.captureIndexTextField.enable()

	def _onConnected(self):
		self.lostTime = self.curSnifferTime + (time_ns() - self.cpuTime) // 1000
		self.reconnectTime = None
		self.captureBtn.enable()
		snifferInst = SnifferInst()
		snifferInst.sniffer.macAddressFilter = [ MACAddress("D6:FA:4A:58:94:B3"), MACAddress("D9:50:2E:00:6F:D8"), MACAddress("ED:3F:AF:BE:34:B8"), MACAddress("C6:7B:E3:B2:8F:C7"), MACAddress("F8:06:4F:CA:45:F5") ]
		snifferInst.sniffer.sendAdvChannelHopSequence([ self.channel ])
		snifferInst.sniffer.sendScan(findScanRsp=False, findAux=False, scanCoded=False)
		self.connectedStatusLabel.set_text(f"Connected to {snifferInst.sniffer.portnum}, at {snifferInst.sniffer.baudrate / 1000} Kbps")

	def onOpen(self):
		# TODO: Implement Preview Button
		# - Opens Preview Figure showing realtime data of what is expected to see after preprocessing:
		#   + Potentially the current heatmap of RSSI values
		#   + Potentially the packet arrival timings
		#   + Potentially the currently expected position
		# TODO: Implement Settings Button
		# - Opens Setting Figure showing the settings for a test session:
		#   + Height for sniffer during static testing
		#   + Static grid offsets dX and dY
		#   + Tunnel/Corridor shape:
		#     * Width, Depth and Height
		#     * List of points for the center line of the tunnel/corridor.
		#       - or potentially one of the walls
		#   + Position and Normal for each beacon (Normal could always be simplified to be (1.0, 0.0) if beacons are placed vertically through the tunnel/corridor).

		self.graphAx = self.addAxes(x=16, y=40, width=self.width - 32, height=self.height - 120)
		self.graphAx.yaxis.tick_right()
		self.graphAx.set_xlim(-60, 0)
		self.graphAx.set_ylim(-120, -10)
		self.graphAx.set_zorder(2)
		self.graphBgAx = self.addAxes(x=0, y=34, width=self.width, height=self.height - 110)
		self.graphBg   = Rectangle(ax=self.graphBgAx, color="white", zorder=0)

		self.startAxesLine(x=4, y=4, maxWidth=self.width - 8, maxHeight=32, anchor="NW", pad=4)
		# self.previewBtnAx           = self.addAxesLine(width=90, height=24)
		self.captureModeBtnAx       = self.addAxesLine(width=90, height=24)
		self.channelSliderLabelAx   = self.addAxesLine(width=64, height=32)
		self.channelSliderAx        = self.addAxesLine(width=50, height=32)
		self.skipAxesLine(widths=[64])
		self.connectedStatusLabelAx = self.addAxesLine(width=-1, height=32)

		self.startAxesLine(x=4, y=44, maxWidth=self.width - 8, maxHeight=32, anchor="SW", pad=4)
		self.captureBtnAx        = self.addAxesLine(width=120, height=24)
		self.captureTimeSliderAx = self.addAxesLine(width=120, height=32)
		self.skipAxesLine(widths=[24])
		self.captureTimeLabelAx  = self.addAxesLine(width=-1, height=32)

		self.startAxesLine(x=4, y=4, maxWidth=self.width - 8, maxHeight=32, anchor="SW", pad=4)
		self.captureXTextFieldLabelAx = self.addAxesLine(width=8, height=32)
		self.captureXTextFieldAx      = self.addAxesLine(width=120, height=24)
		self.captureYTextFieldLabelAx = self.addAxesLine(width=8, height=32)
		self.captureYTextFieldAx      = self.addAxesLine(width=120, height=24)
		self.startAxesLine(x=4, y=4, maxWidth=self.width - 8, maxHeight=32, anchor="SW", pad=4)
		self.captureIndexTextFieldLabelAx = self.addAxesLine(width=38, height=24)
		self.captureIndexTextFieldAx      = self.addAxesLine(width=120, height=24)

		#self.startAxesLine(x=4, y=4, maxWidth=self.width - 8, maxHeight=32, anchor="SE", pad=4)
		#self.settingsBtnAx = self.addAxesLine(width=120, height=24)

		#self.previewBtn           = Button(ax=self.previewBtnAx, label="Preview")
		self.captureModeBtn       = Button(ax=self.captureModeBtnAx, label=CAPTURE_MODE_TXTS[self.captureMode])
		self.channelSliderLabel   = Text(ax=self.channelSliderLabelAx, text="Channel")
		self.channelSlider        = Slider(ax=self.channelSliderAx, valmin=37, valmax=39, valinit=self.channel, valstep=1, valfmt="%u", tickcount=3)
		self.connectedStatusLabel = Text(ax=self.connectedStatusLabelAx, text="")
		#self.previewBtn.disable()
		#self.previewBtn.on_clicked(self._onOpenPreview)
		self.captureModeBtn.on_clicked(self._onCaptureModeChange)
		self.channelSlider.on_changed(self._onChannelChange)

		self.captureBtn        = Button(ax=self.captureBtnAx, label="Start Capture")
		self.captureTimeSlider = Slider(ax=self.captureTimeSliderAx, valmin=1, valmax=60, valinit=60, valstep=1, valfmt="%us")
		self.captureTimeLabel  = Text(ax=self.captureTimeLabelAx, text="No capture yet")
		self.captureBtn.on_clicked(self._onClickCapture)

		self.captureXTextFieldLabel = Text(ax=self.captureXTextFieldLabelAx, text="X")
		self.captureXTextField      = TextField(ax=self.captureXTextFieldAx, type=float, initial=0.0)
		self.captureYTextFieldLabel = Text(ax=self.captureYTextFieldLabelAx, text="Y")
		self.captureYTextField      = TextField(ax=self.captureYTextFieldAx, type=float, initial=0.0)

		self.captureIndexTextFieldLabel = Text(ax=self.captureIndexTextFieldLabelAx, text="Index")
		self.captureIndexTextField      = TextField(ax=self.captureIndexTextFieldAx, type=int, initial=0)

		#self.settingsBtn = Button(ax=self.settingsBtnAx, label="Settings")
		#self.settingsBtn.disable()
		#self.settingsBtn.on_clicked(self._onOpenSettings)

		if self.captureMode == CAPTURE_MODE_STATIC:
			self.captureTimeSlider.set_visible(True)
			self.captureXTextField.set_visible(True)
			self.captureXTextFieldLabel.set_visible(True)
			self.captureYTextField.set_visible(True)
			self.captureYTextFieldLabel.set_visible(True)
			self.captureIndexTextField.set_visible(False)
			self.captureIndexTextFieldLabel.set_visible(False)
		else:
			self.captureTimeSlider.set_visible(False)
			self.captureXTextField.set_visible(False)
			self.captureXTextFieldLabel.set_visible(False)
			self.captureYTextField.set_visible(False)
			self.captureYTextFieldLabel.set_visible(False)
			self.captureIndexTextField.set_visible(True)
			self.captureIndexTextFieldLabel.set_visible(True)

	def onClose(self): pass

	def onUpdate(self):
		curTime = time_ns()

		snifferInst = SnifferInst()
		if self.reconnectTime is None and not snifferInst.IsAlive():
			# Sniffer is disconnected, we will attempt to reconnect
			self.captureBtn.disable()
			self.reconnectTime = curTime + 2_000_000_000 # Reconnect in 2 seconds
			if self.captureMode == CAPTURE_MODE_STATIC:
				self.captureTime = None
				self.captureTimeLabel.set_text("Cancelled")
				self._onCaptureCancelled()
			else:
				totalTime        = time_ns() - self.captureTime
				self.captureTime = None
				self.captureTimeLabel.set_text(f"Done ({totalTime * 1e-9:.2f} seconds captured)")
				self._onCaptureEnd()
		
		if self.reconnectTime is not None:
			deltaTime = self.reconnectTime - curTime
			if deltaTime < 0:
				snifferInst.TryConnect(self._onConnected)
				deltaTime          = 0
				self.reconnectTime = curTime + 2_000_000_000 # Reconnect in 2 seconds
			self.connectedStatusLabel.set_text(f"Not connected, attempting reconnection in {deltaTime * 1e-9:.2f} seconds")
		if self.captureTime is not None:
			deltaTime = self.captureTime - curTime
			if self.captureMode == CAPTURE_MODE_STATIC and deltaTime < 0:
				deltaTime        = 0
				self.captureTime = None
				self._onCaptureEnd()
			self.captureTimeLabel.set_text(f"{deltaTime * 1e-9:.2f} seconds left" if deltaTime > 0 else "Done" if self.captureMode == CAPTURE_MODE_STATIC else f"{-deltaTime * 1e-9:.2f} seconds")
		if self.redrawTime is not None:
			deltaTime = self.redrawTime - curTime
			if deltaTime < 0:
				self.redrawTime += 10_000_000 # Redraw in 10ms
				self.redrawGraph = True
		
		if self.redrawGraph:
			self.redrawGraph = False
			time = self.curSnifferTime + (curTime - self.cpuTime) // 1000
			self.timeTransform.clear().translate(-time, 0).scale(1e-6, 1.0)
			for _, device in self.deviceGraphs.items():
				device.update(time)
			if len(self.deviceGraphs) > 0:
				self.graphAx.legend(loc="upper left")
			self.drawAxes(self.graphBgAx, withBlit=False)
			self.drawAxes(self.graphAx)
	
	def onResize(self):
		self.setMinimumSize(750, 200)

		self.relayoutMainPlot(ax=self.graphAx, leftMargin=16, rightMargin=16, topMargin=40, bottomMargin=80)
		self.graphBgAx.set_position((0, 34, self.width, self.height - 110))

		self.startAxesLine(x=4, y=4, maxWidth=self.width - 8, maxHeight=32, anchor="NW", pad=4)
		self.skipAxesLine(widths=[90, 32, 50, 64]) # TODO: When adding the Preview button back in, add 90 to the start of the widths array
		self.connectedStatusLabelAx.set_position(self.layoutAxesLine(width=-1, height=32))

		self.startAxesLine(x=4, y=44, maxWidth=self.width - 8, maxHeight=32, anchor="SW", pad=4)
		self.skipAxesLine(widths=[120, 120, 24] if self.captureMode == CAPTURE_MODE_STATIC else [120])
		self.captureTimeLabelAx.set_position(self.layoutAxesLine(width=-1, height=32))

from UI.Figures import Figures
Figures().DefineFigure(RSSIFigure)