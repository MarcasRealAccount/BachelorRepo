class MACAddress:
	"""
	A Bluetooth address containing a 6 byte MACAddress and a type specifier (private or public).
	"""

	def Parse(value:str|bytes|list[int]) -> bytes|None:
		isHex = lambda x: (x >= '0' and x <= '9') or (x >= 'a' and x <= 'f') or (x >= 'A' and x <= 'F')
		hexToInt = lambda x: ord(x) - ord('0') if x >= '0' and x <= '9' else 10 + ord(x) - ord('a') if x >= 'a' and x <= 'f' else 10 + ord(x) - ord('A') if x >= 'A' and x <= 'F' else 0
		hexPairToInt = lambda x,y: (hexToInt(x) << 4) | (hexToInt(y))

		if type(value) == str:
			# 00:00:00:00:00:00
			# 00-00-00-00-00-00
			# 00_00_00_00_00_00
			# 0000.0000.0000
			# 000000000000
			if len(value) == 12: # Must be 000000000000
				for i in range(len(value)):
					if not isHex(value[i]):
						return None
				octets:list[int] = []
				octets.append(hexPairToInt(value[10], value[11]))
				octets.append(hexPairToInt(value[8], value[9]))
				octets.append(hexPairToInt(value[6], value[7]))
				octets.append(hexPairToInt(value[4], value[5]))
				octets.append(hexPairToInt(value[2], value[3]))
				octets.append(hexPairToInt(value[0], value[1]))
				return bytes(octets)
			elif len(value) == 14: # Must be 0000.0000.0000
				for i in [ 0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13 ]:
					if not isHex(value[i]):
						return None
				for i in [ 4, 9 ]:
					if value[i] != ".":
						return None
				octets:list[int] = []
				octets.append(hexPairToInt(value[12], value[13]))
				octets.append(hexPairToInt(value[10], value[11]))
				octets.append(hexPairToInt(value[7], value[8]))
				octets.append(hexPairToInt(value[5], value[6]))
				octets.append(hexPairToInt(value[2], value[3]))
				octets.append(hexPairToInt(value[0], value[1]))
				return bytes(octets)
			elif len(value) == 17: # Must be one of the others
				for i in [ 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16 ]:
					if not isHex(value[i]):
						return None
				if value[2] != ":" and value[2] != "-" and value[2] != "_":
					return None
				for i in [ 5, 8, 11, 14 ]:
					if value[i] != value[2]:
						return None
				octets:list[int] = []
				octets.append(hexPairToInt(value[15], value[16]))
				octets.append(hexPairToInt(value[12], value[13]))
				octets.append(hexPairToInt(value[9], value[10]))
				octets.append(hexPairToInt(value[6], value[7]))
				octets.append(hexPairToInt(value[3], value[4]))
				octets.append(hexPairToInt(value[0], value[1]))
				return bytes(octets)
			else:
				return None
		elif type(value) == bytes:
			if len(value) != 6:
				return None
			return value
		elif hasattr(value, "__len__"):
			if len(value) != 6:
				return None
			return bytes(value)
		else:
			return None
	
	def __init__(self, value:str|bytes|list[int], type:bool = False):
		self.value:bytes = MACAddress.Parse(value)
		if self.value is None:
			raise ValueError(f"'{value}' is not a valid MAC Address")
		self.type = type

	def filename(self) -> str:
		"""
		Produces 00_00_00_00_00_00 style representation.
		"""
		return f"{self.value[5]:02X}_{self.value[4]:02X}_{self.value[3]:02X}_{self.value[2]:02X}_{self.value[1]:02X}_{self.value[0]:02X}"
	
	def __repr__(self) -> str:
		"""
		Produces 00:00:00:00:00:00 style representation.
		"""
		return f"{self.value[5]:02X}:{self.value[4]:02X}:{self.value[3]:02X}:{self.value[2]:02X}:{self.value[1]:02X}:{self.value[0]:02X}"
	
	def __eq__(self, value) -> bool:
		"""
		Compares only the MACAddress part of the Bluetooth address.
		"""
		if type(value) == str or type(value) == bytes or hasattr(value, "__len__"):
			value = MACAddress.Parse(value)
		elif type(value) == MACAddress:
			value = value.value
		else:
			return False
		return self.value == value
	
	def __hash__(self) -> int:
		return self.value.__hash__()