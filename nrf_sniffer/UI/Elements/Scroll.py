from UI.Elements.Rectangle import Rectangle

import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib import cbook, colors

import numpy as np

class ScrollBar(Rectangle):
	class ScrolledBbox(mtransforms.Bbox):
		def __init__(self, points, scrollbar, **kwargs):
			super().__init__(points, **kwargs)
			self._scrollbar:ScrollBar = scrollbar
			self._ready = False
			self._tpoints:np.ndarray = None
		
		def get_points(self):
			if self._invalid == 0:
				return self._tpoints
			p = super().get_points()
			self._tpoints = np.copy(p)
			if not self._ready:
				return self._tpoints
			if self._scrollbar.vertical:
				self._tpoints[::-1,1] = 1.0 - self._tpoints[:,1] + (self._scrollbar.scroll / self._scrollbar._curViewSize)
			else:
				self._tpoints[:,0] -= (self._scrollbar.scroll / self._scrollbar._curViewSize)
			return self._tpoints
		
		def contains(self, x, y):
			if not self._ready:
				return super().contains(x, y)
			if self._scrollbar.vertical:
				return super().contains(x, 1.0 - y + (self._scrollbar.scroll / self._scrollbar._curViewSize))
			else:
				return super().contains(x + (self._scrollbar.scroll / self._scrollbar._curViewSize), y)
			
		def fully_contains(self, x, y):
			if not self._ready:
				return super().fully_contains(x, y)
			if self._scrollbar.vertical:
				return super().fully_contains(x, 1.0 - y + (self._scrollbar.scroll / self._scrollbar._curViewSize))
			else:
				return super().fully_contains(x + (self._scrollbar.scroll / self._scrollbar._curViewSize), y)

	def __init__(self, ax:Axes, initScroll:float=None, minScroll:float=0.0, maxScroll:float=1.0, color="0.9", hovercolor="0.85", pressedcolor="0.75", disabledcolor="0.3", direction:str="vertical"):
		super().__init__(ax, facecolor="0.5", edgecolor="0.0", transform=ax.transAxes)

		if initScroll is None:
			initScroll = minScroll
		elif initScroll < minScroll:
			initScroll = minScroll
		elif initScroll > maxScroll:
			initScroll = maxScroll

		if direction == "vertical":
			self.vertical     = True
			self._curViewSize = self._get_view_size()
			self.ax.set_xlim(0.0, 1.0)
			self.ax.set_ylim(maxScroll, minScroll)
			self.handle = self.ax.axhspan(initScroll, initScroll + self._curViewSize)
		else:
			self.vertical     = False
			self._curViewSize = self._get_view_size()
			self.ax.set_xlim(minScroll, maxScroll)
			self.ax.set_ylim(0.0, 1.0)
			self.handle = self.ax.axvspan(initScroll, initScroll + self._curViewSize)

		self.color         = color
		self.hovercolor    = hovercolor
		self.pressedcolor  = pressedcolor
		self.disabledcolor = disabledcolor

		self.scroll    = initScroll
		self.minScroll = minScroll
		self.maxScroll = maxScroll

		self._observers = cbook.CallbackRegistry(signals=["changed"])
		self.connect_event("button_press_event", self._click)
		self.connect_event("button_release_event", self._release)
		self.connect_event("motion_notify_event", self._motion)
		self.connect_event("scroll_event", self._scroll)
		self.connect_event("resize_event", self._resize)

		self._hovered  = False
		self._pressed  = False
		self._disabled = False

		self._axes:list[Axes] = []
	
	def on_changed(self, callback) -> int:
		return self._observers.connect("changed", lambda value: callback(value))
	
	def disconnect(self, cid:int):
		self._observers.disconnect(cid)

	def addAxes(self) -> Axes:
		ax = self.ax.figure.add_axes(ScrollBar.ScrolledBbox([[0,0], [1,1]], self))
		ax._position._ready = True
		self._axes.append(ax)
		return ax

	def _get_view_size(self) -> float:
		extent = self.ax.get_window_extent()
		return extent.height if self.vertical else extent.width

	def _scroll_in_bounds(self, scroll:float) -> float:
		if scroll >= self.maxScroll - self._curViewSize:
			scroll = self.maxScroll - self._curViewSize
		if scroll <= self.minScroll:
			scroll = self.minScroll
		return scroll

	def _update_scroll(self, scroll:float):
		redrawSelf = False
		c          = self.disabledcolor if self._disabled else self.pressedcolor if self._pressed else self.hovercolor if self._hovered else self.color
		scroll     = self._scroll_in_bounds(scroll)
		viewSize   = self._get_view_size()
		if not colors.same_color(c, self.get_facecolor()):
			self.set_facecolor(c, redraw=False)
			redrawSelf = True
		if scroll not in [None, self.scroll] or viewSize != self._curViewSize:
			self.scroll       = scroll
			self._curViewSize = viewSize
			redrawSelf        = False

			for ax in self._axes:
				ax._position.invalidate()

			if self.vertical:
				self.handle.set_y(self.scroll)
				self.handle.set_height(self._curViewSize)
			else:
				self.handle.set_x(self.scroll)
				self.handle.set_width(self._curViewSize)
			self.canvas.nrf_fig.wantRedraw()
		if redrawSelf:
			self._redraw()

	def _update(self, event):
		xdata, ydata = self._get_data_coords(event)
		coord = ydata if self.vertical else xdata
		coord = coord - self._curViewSize / 2
		origScroll = self.scroll
		self._update_scroll(coord)
		if self.eventson and self.scroll != origScroll:
			self._observers.process("changed", self.scroll)

	def _click(self, event):
		if not self.eventson or self.ignore(event) or not self.ax.contains(event)[0] or event.button != 1:
			return
		
		if event.canvas.mouse_grabber != self.ax:
			event.canvas.grab_mouse(self.ax)
			self._pressed = True
			self._update(event)

	def _release(self, event):
		if self.ignore(event) or event.canvas.mouse_grabber != self.ax:
			return
		
		event.canvas.release_mouse(self.ax)
		self._pressed = False
		if self.eventson and self.ax.contains(event)[0]:
			self._update(event)
		else:
			self._update_scroll(self.scroll)

	def _motion(self, event):
		if not self._pressed and self.ignore(event):
			return
		
		self._hovered = self.ax.contains(event)[0]
		if self._pressed:
			self._update(event)
		else:
			self._update_scroll(self.scroll)

	def _scroll(self, event):
		if not self.eventson or self.ignore(event):# or not self.ax.contains(event)[0]:
			return
		
		origScroll = self.scroll
		self._update_scroll(self.scroll - event.step * 20)
		if self.eventson and self.scroll != origScroll:
			self._observers.process("changed", self.scroll)

	def _resize(self, event):
		self._update_scroll(self.scroll)