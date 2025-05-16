import time as ptime
from collections.abc import Callable

_TIMER_NS:Callable[[], int] = None

def _select_time_source():
	global _TIMER_NS
	timers = [
		(ptime.perf_counter_ns, ptime.get_clock_info("perf_counter")),
		(ptime.process_time_ns, ptime.get_clock_info("process_time")),
		(ptime.thread_time_ns, ptime.get_clock_info("thread_time")),
		(ptime.monotonic_ns, ptime.get_clock_info("monotonic")),
		(ptime.time_ns, ptime.get_clock_info("time"))
	]
	timers = sorted([ timer for timer in timers if timer[1].monotonic ], key=lambda x: x[1].resolution)
	_TIMER_NS = timers[0][0]
if _TIMER_NS is None: _select_time_source()

def time_ns() -> int:
	"""
	Provides a monotonic time value in nanoseconds with the highest resolution.
	"""
	return _TIMER_NS()

def time() -> float:
	"""
	Provides a monotonic time value in seconds with the highest resolution.
	"""
	return _TIMER_NS() * 1e-9