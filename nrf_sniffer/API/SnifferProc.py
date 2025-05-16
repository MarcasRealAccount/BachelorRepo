import API.SnifferIO as SnifferIO
from MACAddress import MACAddress
from timer import time_ns

from multiprocessing.synchronize import Event as EventT
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Queue, Event, Value, Array, Pipe
from threading import Thread
import ctypes
import traceback
import os
import io
import shutil
import time

import numpy as np

CAPTURE_MODE_STATIC  = 0
CAPTURE_MODE_DYNAMIC = 1

SNIFFER_QUIT = -1

TIMER_SYNC_COUNT = 32 # NOTE: We need enough samples to get a good average.

class SnifferProcInfo:
	def __init__(self, portnum:str, baudrate:int):
		self.portnum  = portnum
		self.baudrate = Value(ctypes.c_int, baudrate)

		self.aliveEvent = Event()
		self.alive      = Value(ctypes.c_bool, False)
		self.wantClose  = Value(ctypes.c_bool, False)

		self.commands = Queue(maxsize=1024)
		self._packetsReader, self._packetsSender = Pipe(False)

		self._testName     = Array(ctypes.c_char, 128)
		self._captureName  = Array(ctypes.c_char, 128)
		self.captureMode   = Value(ctypes.c_int, CAPTURE_MODE_STATIC)
		self.wantCapture   = Value(ctypes.c_bool, False)
		self.cancelCapture = Value(ctypes.c_bool, False)
		self.capturing     = Value(ctypes.c_bool, False)
		self.captureEvent  = Event()

		self._macAddressFilterMem     = SharedMemory(create=True, size=512 * 6) # 512 capacity
		self._macAddressFilterCount   = Value(ctypes.c_int, 0)
		self._macAddressFilterChanged = Value(ctypes.c_bool, False)

	def close(self):
		self.commands.close()
		self._packetsReader.close()
		self._packetsSender.close()
		self._macAddressFilterMem.close()

	def sendCommand(self, id:int, payload:bytes|None = None):
		self.commands.put((id, payload), block=True)

	def sendPacket(self, packet:SnifferIO.SnifferMessage):
		self._packetsSender.send(packet)

	def readPackets(self) -> list[SnifferIO.SnifferMessage]:
		if not self._packetsReader.poll():
			return []
		
		packets:list[SnifferIO.SnifferMessage] = []
		while self._packetsReader.poll():
			packets.append(self._packetsReader.recv())
		return packets

	@property
	def testName(self) -> str:
		return self._testName.value.decode("utf-8")

	@testName.setter
	def testName(self, value:str):
		self._testName[:] = value.encode("utf-8")[:127].ljust(128, b'\0')
	
	@property
	def captureName(self) -> str:
		return self._captureName.value.decode("utf-8")

	@captureName.setter
	def captureName(self, value:str):
		self._captureName[:] = value.encode("utf-8")[:127].ljust(128, b'\0')

	@property
	def macAddressFilter(self) -> list[MACAddress]:
		with self._macAddressFilterCount.get_lock():
			macArray = np.ndarray((512, 6), dtype="uint8", buffer=self._macAddressFilterMem.buf)
			return [ MACAddress(macArray[i]) for i in range(self._macAddressFilterCount.value) ]
		return []
	
	@macAddressFilter.setter
	def macAddressFilter(self, value:list[MACAddress]):
		with self._macAddressFilterCount.get_lock():
			macArray = np.ndarray((512, 6), dtype="uint8", buffer=self._macAddressFilterMem.buf)
			for i in range(len(value)):
				macArray[i][:] = list(value[i].value)
			self._macAddressFilterCount.value   = len(value)
			self._macAddressFilterChanged.value = True

	def addMacAddress(self, address:MACAddress):
		with self._macAddressFilterCount.get_lock():
			macArray = np.ndarray((512, 6), dtype="uint8", buffer=self._macAddressFilterMem.buf)
			for i in range(self._macAddressFilterCount.value):
				if MACAddress(macArray[i]) == address:
					return
			
			macArray[self._macAddressFilterCount.value][:] = list(address.value)
			self._macAddressFilterCount.value += 1
			self._macAddressFilterChanged.value = True

	def removeMacAddress(self, address:MACAddress):
		with self._macAddressFilterCount.get_lock():
			macArray = np.ndarray((512, 6), dtype="uint8", buffer=self._macAddressFilterMem.buf)
			foundI   = -1
			for i in range(self._macAddressFilterCount.value):
				if MACAddress(macArray[i]) == address:
					foundI = i
					break
			if foundI < 0:
				return
			macArray[foundI:self._macAddressFilterCount.value - 1] = macArray[foundI+1:self._macAddressFilterCount.value]
			macArray[self._macAddressFilterCount.value][:] = [ 0, 0, 0, 0, 0, 0 ]
			
			self._macAddressFilterCount.value -= 1
			self._macAddressFilterChanged.value = True

class SnifferProc:
	@staticmethod
	def RunSnifferProc(procInfo:SnifferProcInfo):
		try:
			sniffer = SnifferIO.SnifferIO(procInfo.portnum, procInfo.baudrate.value)
		except BaseException as e:
			traceback.print_exception(e)
			procInfo.alive.value = False
			procInfo.aliveEvent.set()
			return

		procInfo.alive.value = True
		procInfo.aliveEvent.set()

		print("Sniffer Started")

		proc     = SnifferProc(procInfo, sniffer)
		txThread = Thread(target=proc.runTx, name="SnifferProcTX", daemon=False)
		txThread.start()
		proc.runRx()
		txThread.join()

		procInfo.alive.value = False
		procInfo.aliveEvent.set()
		procInfo.close()

		print("Sniffer Stopped")

	def __init__(self, procInfo:SnifferProcInfo, sniffer:SnifferIO.SnifferIO):
		self.sniffer  = sniffer
		self.running  = True
		self.procInfo = procInfo

		self.scanning = False

		self.cpuReqTimes:list[int]     = []
		self.timestampDelta:int        = 0
		self.timestampDeltas:list[int] = []
		self.nextTimerSync:int         = time_ns()

		self.rawCaptureDir:str  = ""
		self.rssiCaptureDir:str = ""
		self.rawCaptureHandles:dict[MACAddress,io.TextIOWrapper]  = {}
		self.rssiCaptureHandles:dict[MACAddress,io.TextIOWrapper] = {}

		self.macAddressFilter:list[MACAddress] = []

		self.knownAddresses:set[MACAddress] = set()

	def runTx(self):
		while self.running:
			try:
				id, payload = self.procInfo.commands.get(block=True)
				if id == SNIFFER_QUIT: # We will hijack an invalid command id to mean ".close()"
					self.running = False
					self.sniffer.close()
					break
				
				# NOTE: The note below might not matter anymore? It seems the current code doesn't have that problem, perhaps we did something better? We will still write out the additional timestamps in the event there are still problems with them.
				# NOTE: Because the current firmware version of the "nRF Sniffer for Bluetooth LE" (4.1.1) has a problem with timestamps: Sometimes it reports that a BLE packet was received 1 or more seconds in the future, but because they are still valid packets and they were truly received we can just fix the timestamp.
				# Therefore when the command is sent we want to record the current CPU timestamp, so we can synchronize the two timers when we receive the response.
				if id == SnifferIO.REQ_TIMESTAMP:
					self.cpuReqTimes.append(time_ns())

				if id == SnifferIO.REQ_SCAN_CONT:
					self.scanning = True

				self.sniffer.send(id, payload)
			except BaseException as e:
				traceback.print_exception(e, limit=2)
				time.sleep(1.0) # As a precaution we just sleep a bit after an exception occurs.

	def runRx(self):
		while self.running:
			try:
				if self.procInfo.wantClose.value:
					self.running = False
					self.procInfo.sendCommand(SNIFFER_QUIT)
					break

				message = self.sniffer.recv()
				# self.handleTimerResync()
				self.handleCapture()
				if message is None:
					if not self.sniffer.isAlive():
						self.running = False
						self.procInfo.sendCommand(SNIFFER_QUIT)
						break
					continue

				# TODO: We could capture the invalid messages as well, since we might get semi-valid messages which can provide some clues to some problems.
				if not message.valid or not self.scanning:
					continue

				self.handleMessage(message)
			except BaseException as e:
				traceback.print_exception(e, limit=2)
				time.sleep(0.1) # As a precaution we just sleep a bit after an exception occurs.
		
		# If we're still capturing after the sniffer has disconnected then we want to cancel the capturing (which for DYNAMIC is a regular stop)
		if self.procInfo.capturing.value:
			self.procInfo.wantCapture.value   = False
			self.procInfo.cancelCapture.value = self.procInfo.captureMode.value == CAPTURE_MODE_STATIC
			self.handleCapture()

	def handleTimerResync(self):
		curTime = time_ns()
		if curTime >= self.nextTimerSync:
			for i in range(TIMER_SYNC_COUNT):
				self.procInfo.sendCommand(SnifferIO.REQ_TIMESTAMP)
			self.nextTimerSync = curTime + 1_000_000_000 # Next resync in 1 second

	def handleCapture(self):
		if self.procInfo.wantCapture.value == self.procInfo.capturing.value:
			return
		if self.procInfo.wantCapture.value:
			self.procInfo.capturing.value = True

			self.rawCaptureDir  = f"tests/{self.procInfo.testName}/{'static_raw' if self.procInfo.captureMode.value == CAPTURE_MODE_STATIC else 'dynamic_raw'}/{self.procInfo.captureName}/"
			self.rssiCaptureDir = f"tests/{self.procInfo.testName}/{'static' if self.procInfo.captureMode.value == CAPTURE_MODE_STATIC else 'dynamic'}/{self.procInfo.captureName}/"
			shutil.rmtree(self.rawCaptureDir, ignore_errors=True)
			shutil.rmtree(self.rssiCaptureDir, ignore_errors=True)
			os.makedirs(self.rawCaptureDir, exist_ok=True)
			os.makedirs(self.rssiCaptureDir, exist_ok=True)

			self.procInfo.captureEvent.set()
		else:
			self.procInfo.capturing.value = False

			for _, handle in self.rawCaptureHandles.items():
				handle.close()
			for _, handle in self.rssiCaptureHandles.items():
				handle.close()
			if self.procInfo.cancelCapture.value:
				# If we cancelled the capture, we just move the capture into a cancelled folder, so we can still refer to the data
				v = time_ns()
				cancelledRawDir  = f"tests/{self.procInfo.testName}/{'static_raw_cancelled' if self.procInfo.captureMode.value == CAPTURE_MODE_STATIC else 'dynamic_raw_cancelled'}/"
				cancelledRssiDir = f"tests/{self.procInfo.testName}/{'static_cancelled' if self.procInfo.captureMode.value == CAPTURE_MODE_STATIC else 'dynamic_cancelled'}/"
				try:
					os.makedirs(cancelledRawDir, exist_ok=True)
					os.makedirs(cancelledRssiDir, exist_ok=True)
					os.rename(self.rawCaptureDir, f"{cancelledRawDir}/{self.procInfo.captureName}_{v}/")
					os.rename(self.rssiCaptureDir, f"{cancelledRssiDir}/{self.procInfo.captureName}_{v}/")
				except:
					# We failed to rename the capture period, so we'll assume it can't be cancelled and we'll just delete the period.
					shutil.rmtree(self.rawCaptureDir, ignore_errors=True)
					shutil.rmtree(self.rssiCaptureDir, ignore_errors=True)
			self.rawCaptureHandles.clear()
			self.rssiCaptureHandles.clear()
			self.procInfo.captureEvent.set()

	def handleMessageCapture(self, message:SnifferIO.SnifferMessage):
		if not self.procInfo.capturing.value:
			return
		
		if message.bleAdvAddress not in self.rawCaptureHandles:
			rawHandle  = open(f"{self.rawCaptureDir}/{message.bleAdvAddress.filename()}.csv", "w")
			rssiHandle = open(f"{self.rssiCaptureDir}/{message.bleAdvAddress.filename()}.csv", "w")
			rawHandle.write("Timestamp,CPUTimestamp,RSSI,CRCOK,PDUType,Channel,AuxType,PHY,PacketCounter,AA,CI,PDU\n")
			rssiHandle.write("Timestamp,CPUTimestamp,RSSI,CRCOK\n")
			
			self.rawCaptureHandles[message.bleAdvAddress]  = rawHandle
			self.rssiCaptureHandles[message.bleAdvAddress] = rssiHandle
		else:
			rawHandle  = self.rawCaptureHandles[message.bleAdvAddress]
			rssiHandle = self.rssiCaptureHandles[message.bleAdvAddress]
		
		rawHandle.write(f"{message.timestamp},{message.cpuParseTime},{message.bleRSSI},{message.bleCRCOK},{message.blePDUType},{message.bleChannel},{message.bleAuxType},{message.blePHY},{message.packetCounter},{message.bleAccessAddress},{message.bleCodingIndicator},{message.blePDU.hex().upper()}\n")
		rssiHandle.write(f"{message.timestamp},{message.cpuParseTime},{message.bleRSSI},{message.bleCRCOK}\n")

	def handleMessage(self, message:SnifferIO.SnifferMessage):
		if message.packetId in [ SnifferIO.EVENT_PACKET_ADV_PDU, SnifferIO.EVENT_PACKET_DATA_PDU ]:
			if message.bleCRCOK:
				# Filter out all scanner PDUs
				if message.bleFromScan:
					return
				
				self.knownAddresses.add(message.bleAdvAddress)
			else:
				# If the CRC is bad we can't really tell if it is from a scanner or not.
				message.bleAdvAddress = message.TryFindAdvAddress(self.knownAddresses)
				if message.bleAdvAddress is None:
					return
				
			# Should we filter the packet?
			if self.filterMessage(message):
				return
			
			if self.procInfo.capturing:
				self.handleMessageCapture(message)
			self.procInfo.sendPacket(message)

		elif message.packetId == SnifferIO.SWITCH_BAUD_RATE_RESP:
			# If the main process wanted to change baudrate, we need to handle that here
			if message.baudrate != 0:
				self.sniffer._switchBaudrate(message.baudrate)

		elif message.packetId == SnifferIO.RESP_TIMESTAMP:
			cpuTime        = (message.cpuParseTime + self.cpuReqTimes.pop(0)) // 2
			expectedTime   = 1000 * message.timestamp
			timestampDelta = cpuTime - expectedTime
			self.timestampDeltas.append(timestampDelta)
			
			if len(self.timestampDeltas) >= TIMER_SYNC_COUNT:
				factor = 1.0 / TIMER_SYNC_COUNT
				avg    = 0.0
				for delta in self.timestampDeltas:
					avg += factor * delta
				self.timestampDelta = int(avg)
				self.timestampDeltas.clear()

	def filterMessage(self, message:SnifferIO.SnifferMessage) -> bool:
		if self.procInfo._macAddressFilterChanged.value:
			self.procInfo._macAddressFilterChanged.value = False
			self.macAddressFilter = self.procInfo.macAddressFilter
		return len(self.macAddressFilter) > 0 and message.bleAdvAddress not in self.macAddressFilter

# NOTE: This was only necessary earlier, it seems to have been fixed somehow.
#	def fixMessageTimestamp(self, message:SnifferIO.SnifferMessage):
#		if self.timestampDelta == 0:
#			return
#		
#		expectedTime = self.timestampDelta + 1000 * message.timestamp
#		if abs(message.cpuParseTime - expectedTime) > 100_000_000:
#			# Greater difference than 100ms shouldn't happen so we fix the time to the current cpu time.
#			message.timestamp = (message.cpuParseTime - self.timestampDelta) // 1000