from UI.Figure import Figure

from collections.abc import Callable

class Figures:
	"""
	Holds instances of each Figure, and makes it possible to open and close figures by name.
	"""

	def __new__(cls):
		if not hasattr(cls, "instance"):
			cls.instance = super(Figures, cls).__new__(cls)
		return cls.instance
	
	def __init__(self):
		self._figures:dict[str, Figure]         = getattr(self, "_figures", {})
		self._onUpdate:list[Callable[[], None]] = getattr(self, "_onUpdate", [])

	def __del__(self):
		self.Close()

	def OnUpdate(self, callback:Callable[[], None]):
		self._onUpdate.append(callback)

	def DefineFigure(self, clazz:type[Figure], *args, **kwargs):
		"""
		Define a Figure, will invoke the clazz constructor passing in the args and kwargs.
		"""
		fig:Figure = clazz(*args, **kwargs)
		assert(fig.name not in self._figures)
		self._figures[fig.name] = fig

	def GetFigure(self, name:str) -> Figure|None:
		"""
		Get a defined Figure by name.
		"""
		return self._figures[name] if name in self._figures else None
	
	def OpenFigure(self, name:str):
		"""
		Open a defined Figure by name.
		"""
		fig = self.GetFigure(name)
		if fig is not None and not fig.alive:
			fig.open()

	def CloseFigure(self, name:str):
		"""
		Close a defined Figure by name.
		"""
		fig = self.GetFigure(name)
		if fig is not None and fig.alive:
			fig.close()

	def IsAlive(self) -> bool:
		"""
		Checks if there's a non daemon defined Figure that is alive.
		"""
		for _, figure in self._figures.items():
			if not figure.daemon and figure.alive:
				return True
		return False
	
	def Update(self):
		"""
		Update all defined figures.
		"""
		for callback in self._onUpdate:
			callback()
		for _, figure in self._figures.items():
			figure.update()

	def Present(self):
		"""
		Present all defined figures.
		"""
		for _, figure in self._figures.items():
			figure.present()

	def Close(self):
		"""
		Close all defined figures.
		"""
		for _, figure in self._figures.items():
			if figure.alive:
				figure.close()