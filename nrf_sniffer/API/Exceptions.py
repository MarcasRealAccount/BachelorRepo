class TimedOutException(Exception):
	"""
	Exception raised when an operation times out.
	"""
	def __init__(self, message="The operation has timed out."):
		super().__init__(message)