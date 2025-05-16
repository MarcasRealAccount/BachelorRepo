import API.Exceptions as Exceptions
from MACAddress import MACAddress

from timer import time_ns

import serial

SLIP_START     = 0xAB
SLIP_END       = 0xBC
SLIP_ESC       = 0xCD
SLIP_ESC_START = 0xAC
SLIP_ESC_END   = 0xBD
SLIP_ESC_ESC   = 0xCE

VALID_ADV_CHANNELS = [ 37, 38, 39 ]

REQ_FOLLOW                 = 0x00
EVENT_FOLLOW               = 0x01
EVENT_PACKET_ADV_PDU       = 0x02
EVENT_CONNECT              = 0x05
EVENT_PACKET_DATA_PDU      = 0x06
REQ_SCAN_CONT              = 0x07
EVENT_DISCONNECT           = 0x09
SET_TEMPORARY_KEY          = 0x0C
PING_REQ                   = 0x0D
PING_RESP                  = 0x0E
SWITCH_BAUD_RATE_REQ       = 0x13
SWITCH_BAUD_RATE_RESP      = 0x14
SET_ADV_CHANNEL_HOP_SEQ    = 0x17
SET_PRIVATE_KEY            = 0x18
SET_LEGACY_LONG_TERM_KEY   = 0x19
SET_SC_LONG_TERM_KEY       = 0x1A
REQ_VERSION                = 0x1B
RESP_VERSION               = 0x1C
REQ_TIMESTAMP              = 0x1D
RESP_TIMESTAMP             = 0x1E
SET_IDENTITY_RESOLVING_KEY = 0x1F
GO_IDLE                    = 0xFE

BLE_PHY_1M          = 0
BLE_PHY_2M          = 1
BLE_PHY_CODED       = 2
BLE_PHY_CODED_CI_S8 = 0
BLE_PHY_CODED_CI_S2 = 1

BLE_EXT_HEADER_ADV_ADDR    = 1
BLE_EXT_HEADER_TARGET_ADDR = 2
BLE_EXT_HEADER_CTEINFO     = 4
BLE_EXT_HEADER_ADI         = 8
BLE_EXT_HEADER_AUX_PTR     = 16
BLE_EXT_HEADER_SYNCINFO    = 32
BLE_EXT_HEADER_TXPOWER     = 64

BLE_PDU_TYPE_UNKNOWN               = 0
BLE_PDU_TYPE_ADV_IND               = 1  # 0b0000
BLE_PDU_TYPE_ADV_DIRECT_IND        = 2  # 0b0001
BLE_PDU_TYPE_ADV_NONCONN_IND       = 3  # 0b0010
BLE_PDU_TYPE_SCAN_REQ              = 4  # 0b0011
BLE_PDU_TYPE_AUX_SCAN_REQ          = 5  # 0b0011 Secondary
BLE_PDU_TYPE_SCAN_RSP              = 6  # 0b0100
BLE_PDU_TYPE_CONNECT_IND           = 7  # 0b0101
BLE_PDU_TYPE_AUX_CONNECT_REQ       = 8  # 0b0101 Secondary
BLE_PDU_TYPE_ADV_SCAN_IND          = 9  # 0b0110
BLE_PDU_TYPE_ADV_EXT_IND           = 10 # 0b0111
BLE_PDU_TYPE_AUX_ADV_IND           = 11 # 0b0111 Secondary
BLE_PDU_TYPE_AUX_SCAN_RSP          = 12 # 0b0111 Secondary
BLE_PDU_TYPE_AUX_SYNC_IND          = 13 # 0b0111 Periodic
BLE_PDU_TYPE_AUX_CHAIN_IND         = 14 # 0b0111 Secondary and Periodic
BLE_PDU_TYPE_AUX_CONNECT_RSP       = 15 # 0b1000 Secondary
BLE_PDU_TYPE_ADV_DECISION_IND      = 16 # 0b1001

BLE_PDU_TYPE_NAMES = [
	"UNKNOWN",               # BLE_PDU_TYPE_UNKNOWN
	"ADV_IND",               # BLE_PDU_TYPE_ADV_IND
	"ADV_DIRECT_IND",        # BLE_PDU_TYPE_ADV_DIRECT_IND
	"ADV_NONCONN_IND",       # BLE_PDU_TYPE_ADV_NONCONN_IND
	"SCAN_REQ",              # BLE_PDU_TYPE_SCAN_REQ
	"AUX_SCAN_REQ",          # BLE_PDU_TYPE_AUX_SCAN_REQ
	"SCAN_RSP",              # BLE_PDU_TYPE_SCAN_RSP
	"CONNECT_IND",           # BLE_PDU_TYPE_CONNECT_IND
	"AUX_CONNECT_REQ",       # BLE_PDU_TYPE_AUX_CONNECT_REQ
	"ADV_SCAN_IND",          # BLE_PDU_TYPE_ADV_SCAN_IND
	"ADV_EXT_IND",           # BLE_PDU_TYPE_ADV_EXT_IND
	"AUX_ADV_IND",           # BLE_PDU_TYPE_AUX_ADV_IND
	"AUX_SCAN_RSP",          # BLE_PDU_TYPE_AUX_SCAN_RSP
	"AUX_SYNC_IND",          # BLE_PDU_TYPE_AUX_SYNC_IND
	"AUX_CHAIN_IND",         # BLE_PDU_TYPE_AUX_CHAIN_IND
	"AUX_CONNECT_RSP",       # BLE_PDU_TYPE_AUX_CONNECT_RSP
	"ADV_DECISION_IND"       # BLE_PDU_TYPE_ADV_DECISION_IND
]

BLE_PDU_TYPES_FROM_ADV = [
	BLE_PDU_TYPE_ADV_IND,
	BLE_PDU_TYPE_ADV_DIRECT_IND,
	BLE_PDU_TYPE_ADV_NONCONN_IND,
	BLE_PDU_TYPE_SCAN_RSP,
	BLE_PDU_TYPE_ADV_SCAN_IND,
	BLE_PDU_TYPE_ADV_EXT_IND,
	BLE_PDU_TYPE_ADV_DECISION_IND,
	BLE_PDU_TYPE_AUX_ADV_IND,
	BLE_PDU_TYPE_AUX_SCAN_RSP,
	BLE_PDU_TYPE_AUX_SYNC_IND,
	BLE_PDU_TYPE_AUX_CHAIN_IND,
	BLE_PDU_TYPE_AUX_CONNECT_RSP
]
BLE_PDU_TYPES_FROM_SCAN = [
	BLE_PDU_TYPE_SCAN_REQ,
	BLE_PDU_TYPE_AUX_SCAN_REQ,
	BLE_PDU_TYPE_CONNECT_IND,
	BLE_PDU_TYPE_AUX_CONNECT_REQ
]

BLE_PDU_TYPES_MIN_SIZE = [
	0,  # BLE_PDU_TYPE_UNKNOWN
	6,  # BLE_PDU_TYPE_ADV_IND
	12, # BLE_PDU_TYPE_ADV_DIRECT_IND
	6,  # BLE_PDU_TYPE_ADV_NONCONN_IND
	12, # BLE_PDU_TYPE_SCAN_REQ
	12, # BLE_PDU_TYPE_AUX_SCAN_REQ
	6,  # BLE_PDU_TYPE_SCAN_RSP
	34, # BLE_PDU_TYPE_CONNECT_IND
	34, # BLE_PDU_TYPE_AUX_CONNECT_REQ
	6,  # BLE_PDU_TYPE_ADV_SCAN_IND
	1,  # BLE_PDU_TYPE_ADV_EXT_IND
	1,  # BLE_PDU_TYPE_AUX_ADV_IND
	1,  # BLE_PDU_TYPE_AUX_SCAN_RSP
	1,  # BLE_PDU_TYPE_AUX_SYNC_IND
	1,  # BLE_PDU_TYPE_AUX_CHAIN_IND
	1,  # BLE_PDU_TYPE_AUX_CONNECT_RSP
	5   # BLE_PDU_TYPE_ADV_DECISION_IND
]
BLE_PDU_TYPES_MAX_SIZE = [
	255, # BLE_PDU_TYPE_UNKNOWN, we dont know how large this unknown PDU is, so we should accept them regardless
	37,  # BLE_PDU_TYPE_ADV_IND
	12,  # BLE_PDU_TYPE_ADV_DIRECT_IND
	37,  # BLE_PDU_TYPE_ADV_NONCONN_IND
	12,  # BLE_PDU_TYPE_SCAN_REQ
	12,  # BLE_PDU_TYPE_AUX_SCAN_REQ
	37,  # BLE_PDU_TYPE_SCAN_RSP
	34,  # BLE_PDU_TYPE_CONNECT_IND
	34,  # BLE_PDU_TYPE_AUX_CONNECT_REQ
	37,  # BLE_PDU_TYPE_ADV_SCAN_IND
	255, # BLE_PDU_TYPE_ADV_EXT_IND
	255, # BLE_PDU_TYPE_AUX_ADV_IND
	255, # BLE_PDU_TYPE_AUX_SCAN_RSP
	255, # BLE_PDU_TYPE_AUX_SYNC_IND
	255, # BLE_PDU_TYPE_AUX_CHAIN_IND
	255, # BLE_PDU_TYPE_AUX_CONNECT_RSP
	22   # BLE_PDU_TYPE_ADV_DECISION_IND
]

_PRIMARY_ADV_TYPE_TO_BLE_PDU_TYPE = [
	BLE_PDU_TYPE_ADV_IND,          # 0b0000
	BLE_PDU_TYPE_ADV_DIRECT_IND,   # 0b0001
	BLE_PDU_TYPE_ADV_NONCONN_IND,  # 0b0010
	BLE_PDU_TYPE_SCAN_REQ,         # 0b0011
	BLE_PDU_TYPE_SCAN_RSP,         # 0b0100
	BLE_PDU_TYPE_CONNECT_IND,      # 0b0101
	BLE_PDU_TYPE_ADV_SCAN_IND,     # 0b0110
	BLE_PDU_TYPE_ADV_EXT_IND,      # 0b0111
	BLE_PDU_TYPE_UNKNOWN,          # 0b1000
	BLE_PDU_TYPE_ADV_DECISION_IND, # 0b1001
	BLE_PDU_TYPE_UNKNOWN,          # 0b1010
	BLE_PDU_TYPE_UNKNOWN,          # 0b1011
	BLE_PDU_TYPE_UNKNOWN,          # 0b1100
	BLE_PDU_TYPE_UNKNOWN,          # 0b1101
	BLE_PDU_TYPE_UNKNOWN,          # 0b1110
	BLE_PDU_TYPE_UNKNOWN           # 0b1111
]
_SECONDARY_ADV_TYPE_TO_BLE_PDU_TYPE = [
	BLE_PDU_TYPE_UNKNOWN,         # 0b0000
	BLE_PDU_TYPE_UNKNOWN,         # 0b0001
	BLE_PDU_TYPE_UNKNOWN,         # 0b0010
	BLE_PDU_TYPE_AUX_SCAN_REQ,    # 0b0011
	BLE_PDU_TYPE_UNKNOWN,         # 0b0100
	BLE_PDU_TYPE_AUX_CONNECT_REQ, # 0b0101
	BLE_PDU_TYPE_UNKNOWN,         # 0b0110
	BLE_PDU_TYPE_UNKNOWN,         # 0b0111 => Determined by AuxType given by the sniffer
	BLE_PDU_TYPE_AUX_CONNECT_RSP, # 0b1000
	BLE_PDU_TYPE_UNKNOWN,         # 0b1001
	BLE_PDU_TYPE_UNKNOWN,         # 0b1010
	BLE_PDU_TYPE_UNKNOWN,         # 0b1011
	BLE_PDU_TYPE_UNKNOWN,         # 0b1100
	BLE_PDU_TYPE_UNKNOWN,         # 0b1101
	BLE_PDU_TYPE_UNKNOWN,         # 0b1110
	BLE_PDU_TYPE_UNKNOWN          # 0b1111
]

_SNIFFER_AUX_TYPE_TO_BLE_PDU_TYPE = [ BLE_PDU_TYPE_AUX_ADV_IND, BLE_PDU_TYPE_AUX_CHAIN_IND, BLE_PDU_TYPE_AUX_SYNC_IND, BLE_PDU_TYPE_AUX_SCAN_RSP ]

_BLE_ADV_ADDRESS_RANGES = [ range(0, 6), range(2, 8) ]

class SnifferTimeout:
	"""
	Internal class to handle timeout mechanism.
	"""

	def __init__(self, timeout:int):
		# We want to use a monotonic clock if we can.
		self.start   = time_ns() if timeout > 0 else 0
		self.timeout = timeout

	def alive(self) -> bool:
		return (time_ns() - self.start) < self.timeout if self.timeout > 0 else True
	
	def reset(self):
		self.start = time_ns() if self.timeout > 0 else 0

class BLEAdvDataEntry:
	def __init__(self, type:int = 0, data:bytes = b"", partial:bool = False):
		self.type    = type
		self.data    = data
		self.partial = partial

class SnifferMessage:
	"""
	A decoded Sniffer message
	"""

	def __init__(self, data:bytes):
		self.cpuParseTime = time_ns()

		self.rawMessage:bytes    = data
		self.payloadLength:int   = int.from_bytes(data[0:2], "little") if data[2] >= 2 else data[1]
		self.protocolVersion:int = data[2]
		self.packetCounter:int   = int.from_bytes(data[3:5], "little")
		self.packetId:int        = data[5]
		self.payload:bytes       = data[6:]

		self.valid, self.errorMessage = self._validate()
		if not self.valid:
			return
		
		if self.packetId == SWITCH_BAUD_RATE_RESP:
			self.baudrate:int = int.from_bytes(self.payload[:4], "little")
		elif self.packetId == RESP_TIMESTAMP:
			self.timestamp:int = int.from_bytes(self.payload[:4], "little")
		elif self.packetId == EVENT_PACKET_ADV_PDU:
			self.bleValid, self.errorMessage = self._parseAdvPDU()
		elif self.packetId == EVENT_PACKET_DATA_PDU:
			# TODO: Implement data PDU parsing if necessary
			self.bleValid     = False
			self.errorMessage = "Can't parse DATA PDUs yet"

	def _validate(self) -> tuple[bool,str]:
		if self.payloadLength < len(self.payload):
			return False, f"Payload length too short ({self.payloadLength}), expected {len(self.payload)}"
		if self.payloadLength > len(self.payload):
			return False, f"Payload length too large ({self.payloadLength}), expected {len(self.payload)}"
		if self.protocolVersion < 3:
			return False, f"Protocol too old ({self.protocolVersion}), requires 3 or higher"
		if self.protocolVersion > 3:
			return False, f"Protocol too new ({self.protocolVersion}), requires 3 or lower"
		
		if self.packetId == SWITCH_BAUD_RATE_RESP:
			if self.payloadLength < 4:
				return False, f"Payload length for SWITCH_BAUD_RATE_RESP too short ({self.payloadLength}), expected 4"
		elif self.packetId == RESP_TIMESTAMP:
			if self.payloadLength < 4:
				return False, f"Payload length for RESP_TIMESTAMP too short ({self.payloadLength}), expected 4"

		return True, ""
	
	def _defaultInitAdvPDUProperties(self):
		self.bleChannel:int                   = 0
		self.bleRSSI:int                      = 0
		self.bleEventCounter:int              = 0
		self.timestamp:int                    = 0
		self.bleIsAdv:bool                    = True
		self.bleCRCOK:bool                    = False
		self.bleAuxType:int                   = 0
		self.bleAddrResolved:bool             = False
		self.blePHY:int                       = BLE_PHY_1M
		self.bleOK:bool                       = False
		self.bleCoded:bool                    = False
		self.bleAccessAddress:int             = 0
		self.bleCodingIndicator:int           = BLE_PHY_CODED_CI_S8
		self.blePDU:bytes                     = b""
		self.bleCRC:int                       = 0
		self.blePDUType:int                   = BLE_PDU_TYPE_UNKNOWN
		self.bleFromAdv:bool                  = False
		self.bleFromScan:bool                 = False
		self.bleAdvAddress:MACAddress|None    = None
		self.bleInitAddress:MACAddress|None   = None
		self.bleScanAddress:MACAddress|None   = None
		self.bleTargetAddress:MACAddress|None = None
		self.bleAdvData:list[BLEAdvDataEntry] = []
		self.bleExtAdvMode:int                = 0
		self.bleExtHeaderFlags:int            = 0
		self.bleHasTXPower:bool               = False
		self.bleShortName:str                 = None
		self.bleName:str                      = None

	def _parseAdvData(self, advData:bytes) -> list[BLEAdvDataEntry]:
		entries:list[BLEAdvDataEntry] = []
		offset = 0
		while offset < len(advData):
			length = advData[offset]
			if length == 0 or offset + 1 >= len(advData):
				break
			partial = offset + length >= len(advData)
			entries.append(BLEAdvDataEntry(advData[offset+1], advData[offset+2:offset+length+1], partial))
			offset += 1 + length
		return entries

	def _parseAdvPDU(self) -> tuple[bool, str]:
		self._defaultInitAdvPDUProperties()
		if len(self.payload) < 10 or self.payload[0] != 10:
			return False, f"Header metadata length is invalid ({self.payload[0]}), expected 10"
		
		flags                = self.payload[1]
		self.bleChannel      = self.payload[2]
		self.bleRSSI         = -self.payload[3]
		self.bleEventCounter = int.from_bytes(self.payload[4:6], "little")
		self.timestamp       = int.from_bytes(self.payload[6:10], "little")

		self.bleCRCOK        = not not (flags & 1)
		self.bleAuxType      = (flags >> 1) & 3 if self.bleChannel < 37 else 0
		self.bleAddrResolved = not not (flags & 8)
		self.blePHY          = (flags >> 4) & 7

		self.bleOK    = self.bleCRCOK
		self.bleCoded = self.blePHY == BLE_PHY_CODED
		
		if len(self.payload) < (18 if self.bleCoded else 17):
			return False, f"BLE packet is too short ({len(self.payload)}), expected at least {18 if self.bleCoded else 17}"
		
		self.bleAccessAddress   = int.from_bytes(self.payload[10:14], "little")
		self.bleCodingIndicator = self.payload[14] if self.bleCoded else 0
		self.blePDU             = self.payload[15:17] + self.payload[18:-3] if self.bleCoded else self.payload[14:16] + self.payload[17:-3]
		self.bleHeader          = self.blePDU[0]
		self.bleLength          = self.blePDU[1]
		self.blePayload         = self.blePDU[2:]
		self.bleCRC             = int.from_bytes(self.payload[-3:], "little")

		self.bleAdvType = self.bleHeader & 15
		self.bleChSel   = not not (self.bleHeader & 32)
		self.bleTxAdd   = not not (self.bleHeader & 64)
		self.bleRxAdd   = not not (self.bleHeader & 128)

		if self.bleChannel < 37:
			if self.bleAdvType == 7:
				self.blePDUType = _SNIFFER_AUX_TYPE_TO_BLE_PDU_TYPE[self.bleAuxType]
			else:
				self.blePDUType = _SECONDARY_ADV_TYPE_TO_BLE_PDU_TYPE[self.bleAdvType]
		else:
			self.blePDUType = _PRIMARY_ADV_TYPE_TO_BLE_PDU_TYPE[self.bleAdvType]

		self.bleFromAdv  = self.blePDUType in BLE_PDU_TYPES_FROM_ADV
		self.bleFromScan = not self.bleFromAdv

		if not self.bleOK:
			return True, "BLE packet bad"
		
		if self.bleLength < len(self.blePayload):
			return False, f"BLE packet is too short ({self.bleLength}), expected {len(self.blePayload)}"
		if self.bleLength > len(self.blePayload):
			return False, f"BLE packet is too large ({self.bleLength}), expected {len(self.blePayload)}"
		if self.bleLength < BLE_PDU_TYPES_MIN_SIZE[self.blePDUType]:
			return False, f"BLE {BLE_PDU_TYPE_NAMES[self.blePDUType]} packet is too short ({self.bleLength}), expected at least {BLE_PDU_TYPES_MIN_SIZE[self.blePDUType]}"
		if self.bleLength > BLE_PDU_TYPES_MAX_SIZE[self.blePDUType]:
			return False, f"BLE {BLE_PDU_TYPE_NAMES[self.blePDUType]} packet is too large ({self.bleLength}), expected at most {BLE_PDU_TYPES_MAX_SIZE[self.blePDUType]}"

		self._bleAdvData = b""
		self._bleACAD    = b""
		if self.blePDUType in [ BLE_PDU_TYPE_ADV_IND, BLE_PDU_TYPE_ADV_NONCONN_IND, BLE_PDU_TYPE_ADV_SCAN_IND, BLE_PDU_TYPE_SCAN_RSP ]:
			self.bleAdvAddress     = MACAddress(self.blePayload[:6], self.bleTxAdd)
			self._bleAdvData       = self.blePayload[6:]
		elif self.blePDUType == BLE_PDU_TYPE_ADV_DIRECT_IND:
			self.bleAdvAddress     = MACAddress(self.blePayload[:6], self.bleTxAdd)
			self.bleTargetAddress  = MACAddress(self.blePayload[6:12], self.bleRxAdd)
		elif self.blePDUType in [ BLE_PDU_TYPE_SCAN_REQ, BLE_PDU_TYPE_AUX_SCAN_REQ ]:
			self.bleScanAddress = MACAddress(self.blePayload[:6], self.bleTxAdd)
			self.bleAdvAddress  = MACAddress(self.blePayload[6:12], self.bleRxAdd)
		elif self.blePDUType in [ BLE_PDU_TYPE_CONNECT_IND, BLE_PDU_TYPE_AUX_CONNECT_REQ ]:
			self.bleInitAddress      = MACAddress(self.blePayload[:6], self.bleTxAdd)
			self.bleAdvAddress       = MACAddress(self.blePayload[6:12], self.bleRxAdd)
			self.bleACLAccessAddress = int.from_bytes(self.blePayload[12:16], "little")
			self.bleCRCInit          = int.from_bytes(self.blePayload[16:19], "little")
			self.bleWinSize          = self.blePayload[19]
			self.bleWinOffset        = int.from_bytes(self.blePayload[20:22], "little")
			self.bleInterval         = int.from_bytes(self.blePayload[22:24], "little")
			self.bleLatency          = int.from_bytes(self.blePayload[24:26], "little")
			self.bleTimeout          = int.from_bytes(self.blePayload[26:28], "little")
			self.bleChannelMap       = int.from_bytes(self.blePayload[28:33], "little")
			end                      = self.blePayload[33]
			self.bleHop              = end & 31
			self.bleSCA              = (end >> 5) & 7
		elif self.blePDUType in [ BLE_PDU_TYPE_ADV_EXT_IND, BLE_PDU_TYPE_AUX_ADV_IND, BLE_PDU_TYPE_AUX_SCAN_RSP, BLE_PDU_TYPE_AUX_SYNC_IND, BLE_PDU_TYPE_AUX_CHAIN_IND, BLE_PDU_TYPE_AUX_CONNECT_RSP ]:
			header = self.blePayload[0]
			length = header & 63

			self.bleExtAdvMode = (header >> 6) & 3
			self.bleExtHeader  = self.blePayload[1:1+length]
			self._bleAdvData   = self.blePayload[1+length:]

			if length > 0:
				self.bleExtHeaderFlags = self.bleExtHeader[0]

				requiredLen = 1
				if not not (self.bleExtHeaderFlags & 1): requiredLen += 6
				if not not (self.bleExtHeaderFlags & 2): requiredLen += 6
				if not not (self.bleExtHeaderFlags & 4): requiredLen += 1
				if not not (self.bleExtHeaderFlags & 8): requiredLen += 2
				if not not (self.bleExtHeaderFlags & 16): requiredLen += 3
				if not not (self.bleExtHeaderFlags & 32): requiredLen += 18
				if not not (self.bleExtHeaderFlags & 64): requiredLen += 1
				if length < requiredLen:
					return False, f"BLE Extended Header is too short ({length}), expected at least {requiredLen}"

				offset = 1
				if not not (self.bleExtHeaderFlags & 1):
					self.bleAdvAddress = MACAddress(self.bleExtHeader[offset:offset+6], self.bleTxAdd)
					offset += 6

				if not not (self.bleExtHeaderFlags & 2):
					self.bleTargetAddress = MACAddress(self.bleExtHeader[offset:offset+6], self.bleRxAdd)
					offset += 6

				if not not (self.bleExtHeaderFlags & 4):
					cteinfo         = self.bleExtHeader[offset]
					self.bleCTETime = cteinfo & 31
					self.bleCTEType = (cteinfo >> 6) & 3
					offset += 1

				if not not (self.bleExtHeaderFlags & 8):
					advDataInfo       = int.from_bytes(self.bleExtHeader[offset:offset+2], "little")
					self.bleAdvDataID = advDataInfo & 4095
					self.bleAdvSetID  = (advDataInfo >> 12) & 15
					offset += 2

				if not not (self.bleExtHeaderFlags & 16):
					auxPtr                      = int.from_bytes(self.bleExtHeader[offset:offset+3], "little")
					self.bleAuxPtrChannel       = auxPtr & 63
					self.bleAuxPtrClockAccuracy = not not (auxPtr & 64)
					self.bleAuxPtrOffsetUnits   = not not (auxPtr & 128)
					self.bleAuxPtrOffset        = (auxPtr >> 8) & 8191
					self.bleAuxPtrPHY           = (auxPtr >> 21) & 7
					offset += 3
					
				if not not (self.bleExtHeaderFlags & 32):
					offsetInfo = int.from_bytes(self.bleExtHeader[offset:offset+2], "little")
					chmsca     = int.from_bytes(self.bleExtHeader[offset+4:offset+9], "little")

					self.bleOffsetBase           = offsetInfo & 8191
					self.bleOffsetUnits          = not not (offsetInfo & 8192)
					self.bleOffsetAdjust         = not not (offsetInfo & 16384)
					self.bleInterval             = int.from_bytes(self.bleExtHeader[offset+2:offset+4], "little")
					self.bleChannelMap           = chmsca & 137438953471 # 37 bits
					self.bleSCA                  = (chmsca >> 37) & 7
					self.bleACLAccessAddress     = int.from_bytes(self.bleExtHeader[offset+9:offset+13], "little")
					self.bleCRCInit              = int.from_bytes(self.bleExtHeader[offset+13:offset+16], "little")
					self.blePeriodicEventCounter = int.from_bytes(self.bleExtHeader[offset+16:offset+18], "little")
					
					offset += 18
				
				if not not (self.bleExtHeaderFlags & 64):
					self.bleHasTXPower = True
					self.bleTxPower    = self.bleExtHeader[offset]
					self.blePathloss   = self.bleTxPower - self.bleRSSI
					offset += 1

				if offset < length:
					self._bleACAD = self.bleExtHeader[offset:]
		elif self.blePDUType == BLE_PDU_TYPE_ADV_DECISION_IND:
			header = self.blePayload[0]
			self.bleDecisionTypeFlags = header & 63
			self.bleExtAdvMode = (header >> 6) & 3
			
			self.bleExtHeaderFlags = self.blePayload[1]
			if not (self.bleExtHeaderFlags & 16):
				return False, f"BLE ADV_DECISION_IND mandates that AuxPtr is to be present"
			if (self.bleExtHeaderFlags & 38) != 0:
				return False, f"BLE ADV_DECISION_IND mandates that TargetA, CTEInfo and SyncInfo is not present"

			requiredLen = 4
			if not not (self.bleExtHeaderFlags & 1): requiredLen += 6
			if not not (self.bleExtHeaderFlags & 8): requiredLen += 2
			if not not (self.bleExtHeaderFlags & 64): requiredLen += 1
			if length < requiredLen:
				return False, f"BLE ADV_DECISION_IND Extended Header is too short ({length}), expected at least {requiredLen}"
			
			self.bleExtHeader    = self.blePayload[1:1+requiredLen]
			self.bleDecisionData = self.blePayload[1+requiredLen:]

			offset = 1
			if not not (self.bleExtHeaderFlags & 1):
				self.bleAdvAddress = MACAddress(self.bleExtHeader[offset:offset+6], self.bleTxAdd)
				offset += 6

			if not not (self.bleExtHeaderFlags & 8):
				advDataInfo       = int.from_bytes(self.bleExtHeader[offset:offset+2], "little")
				self.bleAdvDataID = advDataInfo & 4095
				self.bleAdvSetID  = (advDataInfo >> 12) & 15
				offset += 2

			auxPtr                      = int.from_bytes(self.bleExtHeader[offset:offset+3], "little")
			self.bleAuxPtrChannel       = auxPtr & 63
			self.bleAuxPtrClockAccuracy = not not (auxPtr & 64)
			self.bleAuxPtrOffsetUnits   = not not (auxPtr & 128)
			self.bleAuxPtrOffset        = (auxPtr >> 8) & 8191
			self.bleAuxPtrPHY           = (auxPtr >> 21) & 7
			offset += 3
				
			if not not (self.bleExtHeaderFlags & 64):
				self.bleHasTXPower = True
				self.bleTxPower    = self.bleExtHeader[offset]
				self.blePathloss   = self.bleTxPower - self.bleRSSI
				offset += 1

			requiredLen = 0
			if self.bleDecisionTypeFlags & 1: requiredLen += 6
			if len(self.bleDecisionData) < requiredLen:
				return False, f"BLE ADV_DECISION_IND Decision Data is too short ({len(self.bleDecisionData)}), expected at least {requiredLen}"
			
			offset = 0
			if self.bleDecisionTypeFlags & 1:
				self.bleDDResolvableTagHash  = int.from_bytes(self.bleDecisionData[offset:offset+3], "little")
				self.bleDDResolvableTagPrand = int.from_bytes(self.bleDecisionData[offset+3:offset+6], "little")
				offset += 6
			if offset < len(self.bleDecisionData):
				self.bleDDArbitraryData = self.bleDecisionData[offset:]

		advDataEntries  = self._parseAdvData(self._bleAdvData)
		acadDataEntries = self._parseAdvData(self._bleACAD)
		self.bleAdvData = acadDataEntries + advDataEntries

		for data in self.bleAdvData:
			if data.type == 0x08: # Short local name
				self.bleShortName = str(data.data, encoding="utf-8", errors="replace")
			elif data.type == 0x09: # Complete local name
				self.bleName = str(data.data, encoding="utf-8", errors="replace")
			elif data.type == 0x0A: # TX Power
				self.bleHasTXPower = True
				self.bleTxPower    = int.from_bytes(data.data[0:1], signed=True)
				self.blePathloss   = self.bleTxPower - self.bleRSSI
		return True, ""
	
	def TryFindAdvAddress(self, knownAddresses:list[MACAddress]) -> MACAddress:
		"""
		Try to determine what mac address this PDU could be for, since we have no way of knowing where a bit flipped.
		We just hope the bit wasn't part of the adv address field.
		"""
		if self.bleCRCOK:
			return self.bleAdvAddress

		ranges = _BLE_ADV_ADDRESS_RANGES
		for _range in ranges:
			if _range.start >= len(self.blePDU) or _range.stop >= len(self.blePDU) or _range.start < 0 or _range.stop < 0:
				continue
			addr = MACAddress(self.blePDU[_range.start:_range.stop:_range.step])
			for knownAddr in knownAddresses:
				if addr == knownAddr:
					return knownAddr
		return None

	@property
	def bleResolvedName(self) -> str:
		return self.bleName or self.bleShortName or (str(self.bleAdvAddress) if self.bleFromAdv else str(self.bleInitAddress or self.bleScanAddress or self.bleTargetAddress)) or "?"

	@property
	def bleHasName(self) -> bool:
		return self.bleName is not None or self.bleShortName is not None

class Key:
	"""
	128 bit key
	"""

	def __init__(self, value:bytes|list[int]):
		if type(value) == bytes:
			if len(value) != 16:
				raise ValueError(f"Key requires 16 bytes, only received {len(value)} bytes")
			self.value = value
		elif type(value) == list:
			if len(value) != 16:
				raise ValueError(f"Key requires 16 bytes, only received {len(value)} bytes")
			self.value = bytes(value)
		else:
			raise ValueError(f"Key requires bytes or list[int] types, received {str(type(value))}")

class Key32:
	"""
	256 bit key
	"""

	def __init__(self, value:bytes|list[int]):
		if type(value) == bytes:
			if len(value) != 32:
				raise ValueError(f"Key32 requires 32 bytes, only received {len(value)} bytes")
			self.value = value
		elif type(value) == list:
			if len(value) != 32:
				raise ValueError(f"Key32 requires 32 bytes, only received {len(value)} bytes")
			self.value = bytes(value)
		else:
			raise ValueError(f"Key32 requires bytes or list[int] types, received {str(type(value))}")

class SnifferIO:
	"""
	Represents the Input/Output mechanics for the Sniffer API, does the Sniffer message reading and writing.

	In comparison to Sniffer, this class gives direct access to the sniffer communication.
	"""

	def __init__(self, portnum:str|None, baudrate:int):
		self.portnum           = portnum
		self.baudrate          = baudrate
		self._timeout:int      = 0
		self._timeoutCount:int = 0
		self._packetCount:int  = 0

		# Since we want to handle timeout gracefully, we still need to keep the previous message data for the next read attempt, unless timeout is 0, at which point we wont return until a message has been read in full.
		self._prevData:bytearray = bytearray()
		self._prevEscaped:bool   = False

		self._buffer:bytearray = bytearray(64)
		self._bufferIndex:int  = 0
		self._bufferSize:int   = 0

		self._serial:serial.Serial = None
		try:
			self._serial = serial.Serial(
				port=self.portnum,
				baudrate=self.baudrate,
				timeout=0.1,
				rtscts=True,
				exclusive=True,
				write_timeout=0.1)
			if hasattr(self._serial, "set_buffer_size"):
				# On Windows request 64 KiB buffer, as they are by default only requesting a 4 KiB buffer, which would be filled up in potentially 16ms, 64 KiB would fill up in 262ms which should be more than enough time to read and handle all of it.
				# * Calculations at 2 Mbps = 2'000'000 bits per second. However currently we only get 1 Mbps, so 32ms and 524ms for that speed.
				self._serial.set_buffer_size(rx_size=65536)
		except:
			if self._serial is not None:
				self._serial.close()
				self._serial = None
			raise Exceptions.TimedOutException("Could not connect to sniffer.")
		
	def __del__(self):
		self.close()

	def isAlive(self) -> bool:
		"""
		Is the Serial port still alive?
		
		Note: serial ports dont convey if they're active or not, so this function might not detect disconnects very quickly, and it's instead up to the recv and send functions to determine the other end is down.
		"""
		return self._serial is not None and self._serial.is_open

	def close(self):
		"""
		Closes the backing Serial port, if we are on a separate thread attempting to close it prematurely we have to notify the reading thread to stop reading, which we can do with Serial.cancel_read().

		This will work in most cases, since we would be using multiple Python Threads on the same Python Process to do this, therefore only one thread will run the function at a time, so it shouldn't cause a problem, though there might be a potential case where Python decides to let the other thread execute whilst this thread is calling the native function for cancelling the io operation, but assuming the other thread is waiting for the io operation to complete that thread shouldn't be doing any writing so it should be fine.
		"""
		if self._serial is not None:
			if hasattr(self._serial, "cancel_read"):
				self._serial.cancel_read()
			self._serial.close()
			self._serial = None

	def setTimeout(self, timeout:float):
		"""
		Sets the timeout for the read function.
		"""
		self._timeout = int(timeout * 1e9)

	def _slip_send(self, data:bytes):
		"""
		Writes a single SLIP message.
		"""
		if self._serial is None or not self._serial.is_open:
			raise Exceptions.TimedOutException("Could not connect to sniffer.")
		
		slipData = bytearray()
		slipData.append(SLIP_START)
		for c in data:
			if c == SLIP_START:
				slipData.append(SLIP_ESC)
				slipData.append(SLIP_ESC_START)
			elif c == SLIP_END:
				slipData.append(SLIP_ESC)
				slipData.append(SLIP_ESC_END)
			elif c == SLIP_ESC:
				slipData.append(SLIP_ESC)
				slipData.append(SLIP_ESC_ESC)
			else:
				slipData.append(c)
		slipData.append(SLIP_END)
		try:
			self._serial.write(bytes(slipData))
		except:
			raise Exceptions.TimedOutException("Timed out during serial write")

	def _buf_recv(self) -> tuple[bool, int]:
		"""
		Reads a single byte, using buffered io.
		"""
		while self._bufferIndex == self._bufferSize:
			try:
				self._bufferSize = self._serial.readinto(self._buffer)
			except Exception as e:
				# Sometimes the operating system functions will throw exceptions when you attempt to read from an unplugged port, so we will assume any exceptions thrown from there is caused by that.
				self._bufferSize = 0
				self.close()
				return (False, 0)
			self._bufferIndex = 0
			# If nothing was read, we assume the device is potentially unplugged.
			# We have the full timeout equal to a few seconds equivalent.
			if self._bufferSize == 0:
				self._timeoutCount += 1
				if self._timeoutCount >= 5:
					self.close()
					return (False, 0)
				continue

			self._timeoutCount = 0
			break

		v = self._buffer[self._bufferIndex]
		self._bufferIndex += 1
		return (True, v)

	def _slip_recv(self) -> bytes|None:
		"""
		Reads a single SLIP message.
		"""
		if self._serial is None or not self._serial.is_open:
			return None

		timeout = SnifferTimeout(self._timeout)
		# If we had a previous unfinished message we dont want to skip forward until a message start.
		if len(self._prevData) == 0:
			while timeout.alive():
				read, value = self._buf_recv()
				if not read:
					return None
				if value == SLIP_START:
					break
			if not timeout.alive():
				return None
		
		endOfMessage = False
		message      = self._prevData
		escaped      = self._prevEscaped
		while timeout.alive():
			read, value = self._buf_recv()
			if not read:
				return None
			
			if escaped:
				if value == SLIP_ESC_START:
					message.append(SLIP_START)
				elif value == SLIP_ESC_END:
					message.append(SLIP_END)
				elif value == SLIP_ESC_ESC:
					message.append(SLIP_ESC)
				else:
					message.append(SLIP_END) # NOTE: Not sure what value should be placed here if there's an illegal escape sequence...
				escaped = False
			else:
				if value == SLIP_END:
					endOfMessage = True
					break
				elif value == SLIP_ESC:
					escaped = True
					continue
				else:
					message.append(value)
					
		# If we timed out before reaching the end of the message we keep track of the state and return no message.
		# However if we did receive the full message we need to clear the previous state, so the next time we read a message we start from a clean slate.
		if not endOfMessage:
			self._prevData    = message
			self._prevEscaped = escaped
			return None
		self._prevData    = bytearray()
		self._prevEscaped = False
		return bytes(message)
	
	def recv(self) -> SnifferMessage|None:
		"""
		Reads a single Sniffer message.
		"""

		message = self._slip_recv()
		if message is None:
			return None
		return SnifferMessage(message)
	
	def send(self, id:int, payload:bytes|None=None):
		"""
		Sends a single Sniffer message.
		"""

		packetCount = self._packetCount
		self._packetCount = (self._packetCount + 1) & 65535 # We have to wrap around within the unsigned 16 bit number. I.e. we just do (x + 1) & 65535 to keep the lower 16 bits after the increment.
		if payload is None:
			data = bytes([ 0, 0, 3 ]) + int.to_bytes(packetCount, 2, "little") + bytes([ id ])
		else:
			data = int.to_bytes(len(payload), 2, "little") + bytes([ 3 ]) + int.to_bytes(packetCount, 2, "little") + bytes([ id ]) + payload
		self._slip_send(data)

	def sendFollow(self, addr:MACAddress, followOnlyAdvertisements = True, followOnlyLegacy = False, followCoded = False):
		"""
		Tell sniffer to start following device by its address and address type
		"""
		flags = followOnlyAdvertisements | (followOnlyLegacy << 1) | (followCoded << 2)
		self.send(REQ_FOLLOW, addr.value + bytes([ addr.type, flags ]))

	def sendScan(self, findScanRsp = False, findAux = False, scanCoded = False):
		"""
		Tell sniffer to start scanning.
		"""
		flags = findScanRsp | (findAux << 1) | (scanCoded << 2)
		self.send(REQ_SCAN_CONT, bytes([ flags ]))
		self.send(SET_TEMPORARY_KEY, bytes([ 0 ] * 16)) # For the Sniffer to start scanning we have to clear its default undefined temporary key, once we have done that it will start understanding unencrypted beacon packets

	def sendTemporaryKey(self, key:Key):
		"""
		Provide a temporary key used to decrypt encrypted packets
		"""
		self.send(SET_TEMPORARY_KEY, key.value)

	def sendPrivateKey(self, key:Key32):
		"""
		Provide a private key used to decrypt encrypted packets
		"""
		self.send(SET_PRIVATE_KEY, key.value)

	def sendLegacyLongTermKey(self, key:Key):
		"""
		Provide a legacy long term key to decrypt encrypted packets
		"""
		self.send(SET_LEGACY_LONG_TERM_KEY, key.value)
	
	def sendSecureConnectionLongTermKey(self, key:Key):
		"""
		Provide a secure connection long term key to decrypt encrypted packets
		"""
		self.send(SET_SC_LONG_TERM_KEY, key.value)

	def sendIdentityResolvingKey(self, key:Key):
		"""
		Provide an identity resolving key to decrypt encrypted packets
		"""
		self.send(SET_IDENTITY_RESOLVING_KEY, key.value)

	def sendAdvChannelHopSequence(self, sequence:list[int]):
		"""
		Provide advertisement channel hop sequence, this makes the sniffer hop from sequence[0] to sequence[1] to sequence[2] and back to sequence[0].
		If len(sequence) < 3 it will still only jumps between the channels provided in the list.
		"""
		for v in sequence:
			if v not in VALID_ADV_CHANNELS:
				raise ValueError(f"Invalid channel value in sequence '{v}', can only be one of '{VALID_ADV_CHANNELS}'")
		self.send(SET_ADV_CHANNEL_HOP_SEQ, bytes([len(sequence)] + sequence + [37]*(3 - len(sequence))))

	def sendVersionReq(self):
		"""
		Send a request for the sniffer version.
		"""
		self.send(REQ_VERSION)

	def sendTimestampReq(self):
		"""
		Send a request for the timestamp on the sniffer.
		Note: the timestamp will be delayed by the equation "constant + variable", where constant determines the time it takes for the request to be sent and received by the sniffer, and variable determines the time it takes for the request to be handled by the host and sniffer, which in most cases would be a few milliseconds at most.
		"""
		self.send(REQ_TIMESTAMP)

	def sendGoIdle(self):
		"""
		Tells the sniffer to go into idle and do nothing but wait for new commands.
		Note: At the moment this function might not actually do anything.
		"""
		self.send(GO_IDLE)

	def getVersion(self) -> str:
		"""
		Request version and wait for response, all other messages will be void-ed.
		"""

		self.send(REQ_VERSION)
		count = 0
		while True:
			# As a precaution, if we didn't receive a response within 32 messages we just resend the request.
			# Eventually it must respond.
			# TODO: Should we instead just `return None` to indicate that it didn't receive a response?
			count += 1
			if count >= 32:
				self.send(REQ_VERSION)
				count = 0

			response = self.recv()
			if response is None or not response.valid or response.packetId != RESP_VERSION:
				continue
			return str(response.payload, encoding="ASCII")

	def _switchBaudrate(self, newBaudrate:int):
		self._serial.baudrate = newBaudrate
		self.baudrate         = newBaudrate

	def trySwitchBaudrate(self, newBaudrate:int) -> bool:
		"""
		Try to switch serial baudrate, this function will wait for the response and handle it, all other messages will be void-ed.
		"""

		self.send(SWITCH_BAUD_RATE_REQ, int.to_bytes(newBaudrate, 4, "little"))
		while True:
			response = self.recv()
			if response is None or not response.valid or response.packetId != SWITCH_BAUD_RATE_RESP:
				continue
				
			if newBaudrate == int.from_bytes(response.payload[:4], "little"):
				self._switchBaudrate(newBaudrate)
				return True
			return False
		
	def trySwitchBaudrates(self, baudrates:list[int]) -> int:
		"""
		Try to switch serial baudrate, this function will wait for the response and handle it, all other messages will be void-ed.
		
		This function goes over the baudrates from index 0 to len(baudrates), whichever is accepted will be used.
		
		Returns either the current baudrate if no changes occurred or the new baudrate.
		"""

		for baudrate in baudrates:
			if self.trySwitchBaudrate(baudrate):
				return baudrate
		return self.baudrate