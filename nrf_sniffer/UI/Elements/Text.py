from UI.Elements.Rectangle import Rectangle

from matplotlib.axes import Axes

class Text(Rectangle):
	def __init__(self, ax:Axes, text:str, **kwargs):
		super().__init__(ax, color="white", zorder=0)

		self.text = self.ax.text(0.0, 0.5, text, clip_on=True, zorder=1, horizontalalignment="left", verticalalignment="center", **kwargs)

	def set_text(self, text:str):
		self.text.set_text(text)
		self._redraw()