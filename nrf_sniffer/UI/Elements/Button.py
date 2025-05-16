from UI.Elements.Rectangle import Rectangle

from matplotlib.axes import Axes
from matplotlib import cbook, colors

from collections.abc import Callable

class Button(Rectangle):
	def __init__(self, ax:Axes, label:str="Click", color="0.9", hovercolor="0.85", pressedcolor="0.75", disabledcolor="0.3"):
		super().__init__(ax, facecolor=color, edgecolor="0.0")

		self.label = ax.text(0.5, 0.5, label, verticalalignment="center", horizontalalignment="center", transform=ax.transAxes)

		self._observers = cbook.CallbackRegistry(signals=["clicked"])
		self.connect_event("button_press_event", self._click)
		self.connect_event("button_release_event", self._release)
		self.connect_event("motion_notify_event", self._motion)

		self.color         = color
		self.hovercolor    = hovercolor
		self.pressedcolor  = pressedcolor
		self.disabledcolor = disabledcolor

		self._hovered  = False
		self._pressed  = False
		self._disabled = False

	def on_clicked(self, func:Callable[[], None]) -> int:
		return self._observers.connect("clicked", lambda: func())
	
	def disconnect(self, cid:int):
		self._observers.disconnect(cid)

	def set_label(self, label:str):
		self.label.set_text(label)
		self._redraw()

	def enable(self):
		self._disabled = False
		self.eventson  = True
		self._update_color()
	
	def disable(self):
		self._disabled = True
		self.eventson  = False
		self._update_color()

	def _update_color(self):
		c = self.disabledcolor if self._disabled else self.pressedcolor if self._pressed else self.hovercolor if self._hovered else self.color
		if not colors.same_color(c, self.get_facecolor()):
			self.set_facecolor(c)

	def _click(self, event):
		if not self.eventson or self.ignore(event) or not self.ax.contains(event)[0] or event.button != 1:
			return
		if event.canvas.mouse_grabber != self.ax:
			event.canvas.grab_mouse(self.ax)
			self._pressed = True
			self._update_color()

	def _release(self, event):
		if self.ignore(event) or event.canvas.mouse_grabber != self.ax:
			return
		
		event.canvas.release_mouse(self.ax)
		if self.eventson and self.ax.contains(event)[0]:
			self._observers.process("clicked")

		self._pressed = False
		self._update_color()

	def _motion(self, event):
		if self.ignore(event):
			return
		
		self._hovered = self.ax.contains(event)[0]
		self._update_color()