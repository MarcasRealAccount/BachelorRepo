from UI.Axes import Axes

import matplotlib.pyplot as plt

class Figure:
	"""
	Definition for a Figure.
	"""

	def __init__(self, name:str, size:tuple[float, float] = (1280, 720), daemon:bool = True):
		from matplotlib.figure import Figure as mplFigure
		from matplotlib.transforms import Affine2D

		self.name       = name
		self.suffix:str = None
		self.width      = size[0]
		self.height     = size[1]
		self.daemon     = daemon
		self.alive      = False

		self._wantsRedraw = False
		self._resizing    = False

		self.fig:mplFigure = None

		self.transPixelNW = Affine2D()
		self.transPixelNE = Affine2D()
		self.transPixelSW = Affine2D()
		self.transPixelSE = Affine2D()

		self.curX:float      = 0.0
		self.curY:float      = 0.0
		self.maxWidth:float  = 0.0
		self.maxHeight:float = 0.0
		self.anchor:str      = "NW"
		self.pad:float       = 4.0

	def onOpen(self): pass
	def onClose(self): pass
	def onUpdate(self): pass
	def onRedraw(self): pass
	def onResize(self): pass

	def setMinimumSize(self, width:float, height:float):
		self.fig.canvas.manager.window.setMinimumSize(int(width), int(height))

	def startAxesLine(self, x:float, y:float, maxWidth:float, maxHeight:float, anchor = "NW", pad:float = 4.0):
		"""
		Start a layouted line of axes elements.
		"""
		self.curX      = x
		self.curY      = y
		self.maxWidth  = maxWidth
		self.maxHeight = maxHeight
		self.anchor    = anchor
		self.pad       = pad

	def skipAxesLine(self, widths:list[float]):
		"""
		Skips past the elements given in widths.
		"""
		for width in widths:
			self.curX += self.pad + width

	def layoutAxesLine(self, width:float, height:float) -> tuple[float,float,float,float]:
		"""
		Create the Bounding box for an element at the current layouted x and y position.
		"""
		bb = (self.curX, self.curY + (self.maxHeight - height) * 0.5, width if width >= 0 else self.maxWidth - self.curX, height)
		self.curX += self.pad + bb[2]
		return bb

	def addAxesLine(self, width:float, height:float) -> Axes:
		"""
		Create a new Axes located at the current layouted x and y position.
		"""
		return self.addAxes(*self.layoutAxesLine(width, height), anchor=self.anchor)

	def addAxes(self, x:float, y:float, width:float, height:float, anchor = "NW") -> Axes:
		"""
		Create a new Axes.
		"""
		transform = None
		if anchor == "NW":
			transform = self.transPixelNW
		elif anchor == "NE":
			transform = self.transPixelNE
		elif anchor == "SW":
			transform = self.transPixelSW
		elif anchor == "SE":
			transform = self.transPixelSE
		else:
			transform = self.transPixelNW
		return self.fig.add_axes(Axes(self.fig, (x, y, width, height), transform=transform))

	def relayoutMainPlot(self, ax:Axes, leftMargin:float, rightMargin:float, topMargin:float, bottomMargin:float):
		"""
		Layout main plot to extend in both axes.
		"""
		renderer    = self.fig.canvas.get_renderer()
		ytickWidth  = max((tick.get_window_extent(renderer).width for tick in ax.get_yticklabels() if tick.get_text()), default=0.0)
		xtickHeight = max((tick.get_window_extent(renderer).height for tick in ax.get_xticklabels() if tick.get_text()), default=0.0)
		if ax.yaxis.get_ticks_position() == "left":
			leftMargin += ytickWidth + 8
		else:
			rightMargin += ytickWidth + 8
		if ax.xaxis.get_ticks_position() == "top":
			topMargin += xtickHeight + 8
		else:
			bottomMargin += xtickHeight + 8
		ax.set_position((leftMargin, topMargin, self.width - rightMargin - leftMargin, self.height - bottomMargin - topMargin))

	def open(self):
		"""
		Open Figure.
		"""
		import matplotlib.pyplot as plt

		self.alive = True
		self.fig   = plt.figure(num=f"{self.name} {self.suffix}" if self.suffix is not None else self.name, figsize=(self.width / 100, self.height / 100))
		self.fig.canvas.mpl_connect("resize_event", self._onResize)
		self.fig.canvas.mpl_connect("draw_event", self._onDraw)
		self.fig.canvas.mpl_connect("close_event", self._onClose)
		self.fig.canvas.nrf_fig = self
		self._resizing = True # We dont want to render during onOpen or onResize
		self.onOpen()
		self.onResize()
		self._resizing = False
		self.fig.show()

		import time
		time.sleep(0.01)

	def close(self):
		"""
		Close Figure.
		"""
		self.alive = False
		self.onClose()
		plt.close(self.fig)

	def update(self):
		"""
		Update Figure if it is alive.
		"""
		if not self.alive:
			return
		self.onUpdate()

	def present(self):
		"""
		Present Figure if it is alive.
		"""
		if not self.alive:
			return
		self.fig.canvas.flush_events()
		if self._wantsRedraw:
			self.fig.canvas.draw()
			self._wantsRedraw = False

	def canRedraw(self):
		return not self._resizing

	def drawAxes(self, ax:Axes, withBlit:bool = True):
		if self._resizing:
			return
		
		if self.fig.canvas.supports_blit:
			ax.draw_artist(ax)
			if withBlit:
				self.fig.canvas.blit(ax.bbox)
		else:
			self._wantsRedraw = True

	def wantRedraw(self):
		self._wantsRedraw = True

	def _onResize(self, event):
		from UI.Figures import Figures
		self._resizing = True
		self.width, self.height = self.fig.get_size_inches() * 100
		dpiScale = self.fig.dpi / 100
		self.transPixelSW.clear().scale(dpiScale)
		self.transPixelSE.clear().scale(-dpiScale, dpiScale).translate(self.width * dpiScale, 0.0)
		self.transPixelNW.clear().scale(dpiScale, -dpiScale).translate(0.0, self.height * dpiScale)
		self.transPixelNE.clear().scale(-dpiScale).translate(self.width * dpiScale, self.height * dpiScale)
		self.onResize()
		Figures().Update()
		self._resizing = False

	def _onDraw(self, event):
		self.onRedraw()

	def _onClose(self, event):
		self.alive = False
		self.onClose()