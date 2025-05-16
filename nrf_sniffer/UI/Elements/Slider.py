from numbers import Number

import numpy as np

from matplotlib.widgets import AxesWidget
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle as MPLRectangle
from matplotlib import cbook, colors

from collections.abc import Callable

class Slider(AxesWidget):
	def __init__(self, ax:Axes, valmin:float, valmax:float, *,
				 valinit:float=0.5, valstep:float|None=None, valfmt:str|None=None, tickcount:int=2,
				 handle_color="r", track_color="0.76", track_hovercolor="0.58", track_pressedcolor="0.4", track_disabledcolor="0.2"):
		super().__init__(ax)

		self._useblit = self.canvas.supports_blit

		self.track_color         = track_color
		self.track_hovercolor    = track_hovercolor
		self.track_pressedcolor  = track_pressedcolor
		self.track_disabledcolor = track_disabledcolor

		self.valmin  = valmin
		self.valmax  = valmax
		self.valstep = valstep
		self.valinit = self._val_in_bounds(valinit)
		self.val     = self.valinit
		self.valfmt  = valfmt

		ax.set_navigate(False)
		ax.set_axis_off()
		ax.set_xticks([])
		ax.set_yticks([])

		ax.set_xlim((valmin - 0.05 * (valmax - valmin), valmax + 0.1 * (valmax - valmin)))
		ax.set_ylim((0.0, 1.0))

		self.bg      = MPLRectangle(xy=(0.0,0.0), width=1.0, height=1.0, color="white", animated=True, transform=ax.transAxes)
		self.track   = ax.axhline(y=0.8, xmin=0.0, xmax=1.0, color=self.track_color, lw=2.0, clip_on=True, zorder=self.ax.zorder + 2)
		self.handle  = ax.axvline(x=self.val, ymin=0.7, ymax=1.0, color=handle_color, lw=2.0, clip_on=True, zorder=self.ax.zorder + 3)
		self.labelBg = MPLRectangle(xy=(0.0,0.0), width=1.0, height=1.0, color="white", animated=True, transform=ax.transAxes, clip_on=False)
		self.label   = ax.text(1.02, 1.0, self._format_val(), verticalalignment="top", horizontalalignment="left", transform=ax.transAxes, clip_on=False)
		ax.add_patch(self.bg)
		ax.add_patch(self.labelBg)

		self.ticks      = ax.vlines([ valmin + (valmax - valmin) * x / (tickcount - 1) for x in range(tickcount) ], ymin=0.5, ymax=1.0, color=self.track_color, lw=1.0, clip_on=False, zorder=self.ax.zorder + 1)
		self.ticklabels = [ ax.text(valmin + (valmax - valmin) * x / (tickcount - 1), 0, self._format_value(valmin + (valmax - valmin) * x / (tickcount - 1)), verticalalignment="bottom", horizontalalignment="center", clip_on=False) for x in range(tickcount) ]

		self._observers = cbook.CallbackRegistry(signals=["changed"])
		self.connect_event("button_press_event", self._click)
		self.connect_event("button_release_event", self._release)
		self.connect_event("motion_notify_event", self._motion)

		self._hovered  = False
		self._pressed  = False
		self._disabled = False

	def on_changed(self, func:Callable[[float], None]) -> int:
		return self._observers.connect("changed", lambda value: func(value))
	
	def disconnect(self, cid):
		self._observers.disconnect(cid)

	def set_val(self, value:float):
		self._update_val(value)

	def set_visible(self, visible:bool):
		self.eventson = visible
		self.drawon   = visible
		self.ax.set_visible(visible)
		self.canvas.nrf_wants_redraw = True

	def enable(self):
		self._disabled = False
		self.eventson  = True
		self._update_val(self.val)

	def disable(self):
		self._disabled = True
		self.eventson  = False
		self._update_val(self.val)

	def _update_val(self, val:float):
		redraw = False
		c      = self.track_disabledcolor if self._disabled else self.track_pressedcolor if self._pressed else self.track_hovercolor if self._hovered else self.track_color
		val    = self._val_in_bounds(val)
		if not colors.same_color(c, self.track.get_color()):
			self.track.set_color(c)
			redraw = True
		if val not in [None, self.val]:
			self.val = val
			redraw   = True
			self.handle.set_xdata([ self.val ])

		if redraw:
			self._redraw()

	def _stepped_val(self, val:float) -> float:
		if isinstance(self.valstep, Number):
			val = (self.valmin + round((val - self.valmin) / self.valstep) * self.valstep)
		elif self.valstep is not None:
			valstep = np.asanyarray(self.valstep)
			if valstep.ndim != 1:
				raise ValueError(f"valstep must have 1 dimension but has {valstep.ndim}")
			val = valstep[np.argmin(np.abs(valstep - val))]
		return val

	def _val_in_bounds(self, val:float) -> float:
		val = self._stepped_val(val)

		if val <= self.valmin:
			val = self.valmin
		elif val >= self.valmax:
			val = self.valmax
		return val
	
	def _format_value(self, value:float) -> str:
		if self.valfmt is not None:
			return self.valfmt % value
		else:
			return str(value)

	def _format_val(self) -> str:
		return self._format_value(self.val)

	def _update(self, event):
		xdata, ydata = self._get_data_coords(event)
		self._update_val(xdata)
		if self.eventson:
			self._observers.process("changed", self.val)

	def _click(self, event):
		if not self.eventson or self.ignore(event) or not self.ax.contains(event)[0]:
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
			self._update_val(self.val)
	
	def _motion(self, event):
		if not self._pressed and self.ignore(event):
			return
		
		self._hovered = self.ax.contains(event)[0]
		if self._pressed:
			self._update(event)
		else:
			self._update_val(self.val)

	def _redraw(self):
		if not self.drawon or not self.canvas.nrf_fig.canRedraw():
			return
		
		if self._useblit:
			self.ax.draw_artist(self.bg)
			self.ax.draw_artist(self.ticks)
			self.ax.draw_artist(self.track)
			self.ax.draw_artist(self.handle)
			for ticklabel in self.ticklabels:
				self.ax.draw_artist(ticklabel)
			self.canvas.blit(self.ax.bbox)

			bbox = self.label.get_window_extent(self.canvas.get_renderer())
			bbox2 = bbox.transformed(self.ax.transAxes.inverted())
			self.labelBg.set_xy(bbox2.min)
			self.labelBg.set_width(bbox2.width)
			self.labelBg.set_height(bbox2.height)
			self.label.set_text(self._format_val())
			self.ax.draw_artist(self.labelBg)
			self.ax.draw_artist(self.label)
			self.canvas.blit(bbox)
		else:
			self.canvas.nrf_fig.wantRedraw()