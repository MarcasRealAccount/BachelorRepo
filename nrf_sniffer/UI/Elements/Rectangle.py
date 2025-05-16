from matplotlib.widgets import AxesWidget
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle as MPLRectangle
from matplotlib import cbook, colors

class Rectangle(AxesWidget):
	def __init__(self, ax:Axes, **kwargs):
		super().__init__(ax)

		self._useblit = self.canvas.supports_blit

		self._rect = MPLRectangle(xy=(0.0,0.0), width=1.0, height=1.0, **kwargs)
		ax.add_patch(self._rect)
		
		ax.set_navigate(False)
		ax.set_axis_off()
		ax.set_xticks([])
		ax.set_yticks([])

	def set_visible(self, visible:bool):
		self.ax.set_visible(visible)
		self.drawon   = visible
		self.eventson = visible
		self.canvas.nrf_fig.wantRedraw()

	def get_facecolor(self):
		return self._rect.get_facecolor()

	def get_edgecolor(self):
		return self._rect.get_edgecolor()

	def set_facecolor(self, facecolor, redraw:bool=True):
		self._rect.set_facecolor(facecolor)
		if redraw:
			self._redraw()

	def set_edgecolor(self, edgecolor, redraw:bool=True):
		self._rect.set_edgecolor(edgecolor)
		if redraw:
			self._redraw()

	def set_colors(self, facecolor, edgecolor, redraw:bool=True):
		self._rect.set_facecolor(facecolor)
		self._rect.set_edgecolor(edgecolor)
		if redraw:
			self._redraw()

	def _redraw(self):
		if self.drawon:
			self.canvas.nrf_fig.drawAxes(self.ax)