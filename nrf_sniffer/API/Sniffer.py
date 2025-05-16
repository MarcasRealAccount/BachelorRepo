import API.SnifferIO as SnifferIO
import API.SnifferProc as SnifferProc
from MACAddress import MACAddress

import multiprocessing as mp
from multiprocessing.managers import SyncManager
import traceback
import io
import ctypes
import os

import serial.tools.list_ports as listports

import numpy as np

SNIFFER_BAUDRATES = [ 2_000_000, 1_000_000, 460_800 ]

CAPTURE_MODE_STATIC  = 0
CAPTURE_MODE_DYNAMIC = 1
CAPTURE_MODE_TXTS    = [ "Static", "Dynamic" ]

class SnifferInfo:
	def __init__(self, portnum:str|None, baudrates:list[int], version:str):
		self.portnum   = portnum
		self.baudrates = baudrates
		self.version   = version

def FindSniffers() -> list[SnifferInfo]:
	openPorts = listports.comports()

	sniffers: list[SnifferInfo] = []
	for port in openPorts:
		if port.pid != 21034 or port.vid != 6421: # nRF Sniffer sues PID 21034 and VID 6421
			continue

		baudrates: list[int] = []
		try:
			sniffer = SnifferIO.SnifferIO(port.device, 480_800)
			sniffer.setTimeout(timeout=1.0)
			version = sniffer.getVersion()
			for rate in SNIFFER_BAUDRATES:
				if sniffer.trySwitchBaudrate(rate):
					baudrates.append(rate)
			if len(baudrates) > 0:
				sniffers.append(SnifferInfo(port.device, baudrates, version))
		except:
			pass
		sniffer.close()
	return sniffers

class Sniffer:
	"""
	Class representing the API access layer through a separate process
	"""

	def __init__(self, snifferInfo:SnifferInfo):
		self._procInfo = SnifferProc.SnifferProcInfo(snifferInfo.portnum, snifferInfo.baudrates[0])
		self._process  = mp.Process(target=SnifferProc.SnifferProc.RunSnifferProc, name="Sniffer", args=(self._procInfo,), daemon=False)
		self._process.start()
		self._procInfo.aliveEvent.wait(timeout=2.0)
		if not self._procInfo.alive.value:
			self.close()
			return
		
	def __del__(self):
		self.close()
	
	def isAlive(self) -> bool:
		return self._procInfo.alive.value

	def close(self):
		if self._process is not None:
			self._procInfo.wantClose.value = True
			self._process.join()
			self._procInfo._macAddressFilterMem.unlink() # This is ugly, Unix is pushing through the veil.
			self._procInfo.close()
			self._process = None

	@property
	def portnum(self):
		return self._procInfo.portnum if self.isAlive() else None

	@property
	def baudrate(self):
		return self._procInfo.baudrate.value if self.isAlive() else 0
	
	@baudrate.setter
	def baudrate(self, newBaudrate:int):
		self.send(SnifferIO.SWITCH_BAUD_RATE_REQ, int.to_bytes(newBaudrate, 4, "little"))

	@property
	def macAddressFilter(self) -> list[MACAddress]:
		return self._procInfo.macAddressFilter

	@macAddressFilter.setter
	def macAddressFilter(self, addresses:list[MACAddress]):
		self._procInfo.macAddressFilter = addresses

	def addMACAddress(self, address:MACAddress):
		self._procInfo.addMacAddress(address)

	def removeMACAddress(self, address:MACAddress):
		self._procInfo.removeMacAddress(address)

	def getPackets(self) -> list[SnifferIO.SnifferMessage]:
		"""
		Get available packets
		"""
		if not self.isAlive():
			return []
		return self._procInfo.readPackets()
	
	def startCapturing(self, testName:str, captureName:str, captureMode:int) -> bool:
		"""
		Tell sniffer process to start capturing.
		"""
		if not self.isAlive():
			return False
		self._procInfo.captureEvent.clear()
		self._procInfo.testName            = testName
		self._procInfo.captureName         = captureName
		self._procInfo.captureMode.value   = captureMode
		self._procInfo.cancelCapture.value = False
		self._procInfo.wantCapture.value   = True
		self._procInfo.captureEvent.wait(5.0) # If the sniffer process didn't respond in time, we can assume it broke.
		return self._procInfo.capturing.value
	
	def cancelCapturing(self) -> bool:
		"""
		Rell sniffer process to stop capturing and mark capture as prematurely cancelled.
		"""
		self._procInfo.captureEvent.clear()
		self._procInfo.cancelCapture.value = True
		self._procInfo.wantCapture.value   = False
		self._procInfo.captureEvent.wait(5.0) # If the sniffer process didn't respond in time, we can assume it broke.
		return self._procInfo.capturing.value

	def stopCapturing(self) -> bool:
		"""
		Tell sniffer process to stop capturing.
		"""
		if not self.isAlive():
			return False
		self._procInfo.captureEvent.clear()
		self._procInfo.cancelCapture.value = False
		self._procInfo.wantCapture.value   = False
		self._procInfo.captureEvent.wait(5.0) # If the sniffer process didn't respond in time, we can assume it broke.
		return self._procInfo.capturing.value

	def send(self, id:int, payload:bytes|None = None):
		"""
		Send sniffer a raw message.
		"""
		if not self.isAlive():
			return
		self._procInfo.sendCommand(id, payload)

	def sendFollow(self, addr:MACAddress, followOnlyAdvertisements = True, followOnlyLegacy = False, followCoded = False):
		"""
		Tell sniffer to start following devidce by its address and address type.
		"""
		flags = followOnlyAdvertisements | (followOnlyLegacy << 1) | (followCoded << 1)
		self.send(SnifferIO.REQ_FOLLOW, addr.value + bytes([ addr.type, flags ]))

	def sendScan(self, findScanRsp = False, findAux = False, scanCoded = False):
		"""
		Tell sniffer to start scanning.
		"""
		flags = findScanRsp | (findAux << 1) | (scanCoded << 2)
		self.send(SnifferIO.REQ_SCAN_CONT, bytes([ flags ]))
		self.send(SnifferIO.SET_TEMPORARY_KEY, bytes([ 0 ] * 16))
	
	def sendTemporaryKey(self, key:SnifferIO.Key):
		"""
		Provide a temporary key used to decrypt encrypted packets
		"""
		self.send(SnifferIO.SET_TEMPORARY_KEY, key.value)

	def sendPrivateKey(self, key:SnifferIO.Key32):
		"""
		Provide a private key used to decrypt encrypted packets
		"""
		self.send(SnifferIO.SET_PRIVATE_KEY, key.value)

	def sendLegacyLongTermKey(self, key:SnifferIO.Key):
		"""
		Provide a legacy long term key to decrypt encrypted packets
		"""
		self.send(SnifferIO.SET_LEGACY_LONG_TERM_KEY, key.value)
	
	def sendSecureConnectionLongTermKey(self, key:SnifferIO.Key):
		"""
		Provide a secure connection long term key to decrypt encrypted packets
		"""
		self.send(SnifferIO.SET_SC_LONG_TERM_KEY, key.value)

	def sendIdentityResolvingKey(self, key:SnifferIO.Key):
		"""
		Provide an identity resolving key to decrypt encrypted packets
		"""
		self.send(SnifferIO.SET_IDENTITY_RESOLVING_KEY, key.value)

	def sendAdvChannelHopSequence(self, sequence:list[int]):
		"""
		Provide advertisement channel hop sequence, this makes the sniffer hop from sequence[0] to sequence[1] to sequence[2] and back to sequence[0].
		If len(sequence) < 3 it will still only jumps between the channels provided in the list.
		"""
		for v in sequence:
			if v not in SnifferIO.VALID_ADV_CHANNELS:
				raise ValueError(f"Invalid channel value in sequence '{v}', can only be one of '{SnifferIO.VALID_ADV_CHANNELS}'")
		self.send(SnifferIO.SET_ADV_CHANNEL_HOP_SEQ, bytes([len(sequence)] + sequence + [37]*(3 - len(sequence))))

	def sendVersionReq(self):
		"""
		Send a request for the sniffer version.
		"""
		self.send(SnifferIO.REQ_VERSION)

	def sendTimestampReq(self):
		"""
		Send a request for the timestamp on the sniffer.
		Note: the timestamp will be delayed by the equation "constant + variable", where constant determines the time it takes for the request to be sent and received by the sniffer, and variable determines the time it takes for the request to be handled by the host and sniffer, which in most cases would be a few milliseconds at most.
		"""
		self.send(SnifferIO.REQ_TIMESTAMP)

	def sendGoIdle(self):
		"""
		Tells the sniffer to go into idle and do nothing but wait for new commands.
		Note: At the moment this function might not actually do anything.
		"""
		self.send(SnifferIO.GO_IDLE)