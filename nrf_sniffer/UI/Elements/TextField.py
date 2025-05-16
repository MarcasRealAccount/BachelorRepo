from UI.Elements.Rectangle import Rectangle

from matplotlib.axes import Axes
from matplotlib import cbook, colors, _api
import matplotlib as mpl
from contextlib import ExitStack

class TextField(Rectangle):
	def __init__(self, ax:Axes, type:type, initial=None, color="0.9", hovercolor="0.85", pressedcolor="0.75", disabledcolor="0.3"):
		super().__init__(ax, facecolor=color, edgecolor="0.0")

		self.type  = type
		self.value = self._value(initial)

		self.ax.set_xlim(0, 1)

		self.textDisp = self.ax.text(0.05, 0.5, str(self.value), clip_on=True, parse_math=False, zorder=1, horizontalalignment="left", verticalalignment="center", transform=ax.transAxes)
		self.cursor   = self.ax.axvline(0, 0.1, 0.9, zorder=2, color="k", lw=1)

		self._observers = cbook.CallbackRegistry(signals=["changed"])
		self.connect_event("button_press_event", self._click)
		self.connect_event("button_release_event", self._release)
		self.connect_event("motion_notify_event", self._motion)
		self.connect_event("key_press_event", self._keypress)
		self.connect_event("resize_event", self._resize)

		self.color         = color
		self.hovercolor    = hovercolor
		self.pressedcolor  = pressedcolor
		self.disabledcolor = disabledcolor

		self.cursor_index = 0

		self._hovered  = False
		self._pressed  = False
		self._disabled = False
		self._capture  = False

	@property
	def text(self):
		return self.textDisp.get_text()

	def on_changed(self, func) -> int:
		return self._observers.connect("changed", lambda newValue: func(newValue))
	
	def disconnect(self, cid:int):
		self._observers.disconnect(cid)

	def set_value(self, value):
		self.value = self._value(value)
		self.textDisp.set_text(str(self.value))
		self._redraw()

	def enable(self):
		self._disabled = False
		self.eventson  = True
		self._update_color()
	
	def disable(self):
		self._disabled = True
		self.eventson  = False
		self._update_color()

	def _value(self, value):
		if self.type == float:
			return float(value) if value is not None else 0.0
		elif self.type == int:
			return int(value) if value is not None else 0
		elif self.type == str:
			return str(value) if value is not None else ""
		else:
			return None	
		
	def _update_color(self):
		c = self.disabledcolor if self._disabled else self.pressedcolor if self._pressed else self.hovercolor if self._hovered else self.color
		if not colors.same_color(c, self.get_facecolor()):
			self.set_facecolor(c)

	def _begin_capture(self):
		self._capture = True
		stack = ExitStack()
		self._on_stop_capture = stack.close
		toolmanager = getattr(self.canvas.manager, "toolmanager", None)
		if toolmanager is not None:
			toolmanager.keypresslock(self)
			stack.callback(toolmanager.keypresslock.release, self)
		else:
			with _api.suppress_matplotlib_deprecation_warning():
				stack.enter_context(mpl.rc_context({ k: [] for k in mpl.rcParams if k.startswith("keymap.") }))

	def _end_capture(self, redraw=True):
		if self._capture:
			self._on_stop_capture()
			self._on_stop_capture = None
			if self.type == float:
				self.textDisp.set_text(str(float(self.text or "0")))
			elif self.type == int:
				self.textDisp.set_text(str(int(self.text or "0")))
			self.value = self._value(self.text)
			self._observers.process("changed", self.text)
		self._capture = False
		self._redraw(redraw)

	def _click(self, event):
		if not self.eventson or self.ignore(event) or not self.ax.contains(event)[0]:
			if self._capture:
				self._end_capture()
			return

		if event.canvas.mouse_grabber != self.ax:
			event.canvas.grab_mouse(self.ax)
			self._pressed = True
			self._update_color()
		if not self._capture:
			self._begin_capture()
		self.cursor_index = self.textDisp._char_index_at(event.x)
		self._redraw()

	def _release(self, event):
		if self.ignore(event) or event.canvas.mouse_grabber != self.ax:
			return
		
		event.canvas.release_mouse(self.ax)
		self._pressed = False
		self._update_color()

	def _motion(self, event):
		if self.ignore(event):
			return
		
		self._hovered = self.ax.contains(event)[0]
		self._update_color()

	def _keypress(self, event):
		if self.ignore(event) or not self.eventson:
			return
		if self._capture:
			key  = event.key
			text = self.text
			if len(key) == 1:
				skip = self._handleFloatSkip(key) if self.type == float else self._handleIntSkip(key) if self.type == int else False
				if skip:
					return
				text = text[:self.cursor_index] + key + text[self.cursor_index:]
				self.cursor_index += 1
			elif key == "up":
				if self.type == float:
					text = str(float(text) + 1.0)
				elif self.type == int:
					text = str(int(text) + 1)
			elif key == "down":
				if self.type == float:
					text = str(float(text) - 1.0)
				elif self.type == int:
					text = str(int(text) - 1)
			elif key == "right":
				if self.cursor_index != len(text):
					self.cursor_index += 1
			elif key == "left":
				if self.cursor_index != 0:
					self.cursor_index -= 1
			elif key == "home":
				self.cursor_index = 0
			elif key == "end":
				self.cursor_index = len(text)
			elif key == "backspace":
				if self.cursor_index != 0:
					text = text[:self.cursor_index - 1] + text[self.cursor_index:]
					self.cursor_index -= 1
			elif key == "delete":
				if self.cursor_index != len(self.text):
					text = text[:self.cursor_index] + text[self.cursor_index + 1:]
			elif key == "enter" or key == "return":
				if self.type == float:
					text = str(float(text or "0"))
				elif self.type == int:
					text = str(int(text or "0"))
				self.value = self._value(self.text)
			self.textDisp.set_text(text)
			self._redraw()
			if self.eventson:
				if key in ["enter", "return"]:
					self._observers.process('changed', self.text)

	def _resize(self, event):
		if self._capture:
			self._end_capture(False)

	def _redraw(self, redraw=True):
		if self._capture:
			text      = self.text
			widthtext = text[:self.cursor_index]

			self.textDisp.set_text(widthtext or "")
			bb_widthtext = self.textDisp.get_window_extent()
			bb_textfield = self.ax.get_window_extent()
			requiredWidth = bb_widthtext.width
			offset = 0.05 + min(-0.1 + (bb_textfield.width - requiredWidth) / bb_textfield.width, 0)
			self.textDisp.set_position(xy=(offset, 0.5))

			self.cursor.set_xdata([ offset + requiredWidth / bb_textfield.width ])
			self.cursor.set_visible(True)
			self.textDisp.set_text(text)
		else:
			self.cursor.set_visible(False)
		if redraw:
			super()._redraw()

	def _handleFloatSkip(self, character):
		if character == '-':
			return self.cursor_index != 0 or (len(self.text) > 0 and self.text[0] == '-')
		elif character == '.':
			return self.text.find('.') != -1
		elif character < '0' or character > '9':
			return True
		return False
	
	def _handleIntSkip(self, character):
		if character == '-':
			return self.cursor_index != 0 or (len(self.text) > 0 and self.text[0] == '-')
		elif character < '0' or character > '9':
			return True
		return False