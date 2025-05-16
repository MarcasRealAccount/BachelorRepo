from matplotlib.axes import Axes as mplAxes
import matplotlib.transforms as mtransforms
from matplotlib import _api

import numpy as np

class BboxTranslateTo(mtransforms.Affine2DBase):
	"""
	`BboxTransformTo` is a transformation that linearly transforms points from
	the unit bounding box to a given `Bbox`.
	"""

	is_separable = True

	def __init__(self, boxout, **kwargs):
		"""
		Create a new `BboxTransformTo` that linearly transforms
		points from the unit bounding box to *boxout*.
		"""
		_api.check_isinstance(mtransforms.BboxBase, boxout=boxout)

		super().__init__(**kwargs)
		self._boxout = boxout
		self.set_children(boxout)
		self._mtx = None
		self._inverted = None

	__str__ = mtransforms._make_str_method("_boxout")

	def get_matrix(self):
		# docstring inherited
		if self._invalid:
			outl, outb, outw, outh = self._boxout.bounds
			self._mtx = np.array([[1.0, 0.0, outl],
								  [0.0, 1.0, outb],
								  [0.0, 0.0,  1.0]],
								 float)
			self._inverted = None
			self._invalid = 0
		return self._mtx

class Axes(mplAxes):
	def __init__(self, fig, *args, **kwargs):
		self.transform = kwargs["transform"] if "transform" in kwargs else fig.transSubFigure
		super().__init__(fig, *args, **kwargs)

		self.transPixel = BboxTranslateTo(self._position) + self.transform
		
	def set_figure(self, fig):
		import matplotlib.transforms as mtransforms

		# docstring inherited
		super().set_figure(fig)

		self.bbox = mtransforms.TransformedBbox(self._position, self.transform)
		# these will be updated later as data is added
		self.dataLim = mtransforms.Bbox.null()
		self._viewLim = mtransforms.Bbox.unit()
		self.transScale = mtransforms.TransformWrapper(mtransforms.IdentityTransform())

		self._set_lim_and_transforms()

	def inset_axes(self, bounds, *, transform=None, zorder=5, **kwargs):
		"""
        Add a child inset Axes to this existing Axes.


        Parameters
        ----------
        bounds : [x0, y0, width, height]
            Lower-left corner of inset Axes, and its width and height.

        transform : `.Transform`
            Defaults to `ax.transPixel`, i.e. the units of *rect* are in
            Axes-relative pixel coordinates.

        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', \
'polar', 'rectilinear', str}, optional
            The projection type of the inset `~.axes.Axes`. *str* is the name
            of a custom projection, see `~matplotlib.projections`. The default
            None results in a 'rectilinear' projection.

        polar : bool, default: False
            If True, equivalent to projection='polar'.

        axes_class : subclass type of `~.axes.Axes`, optional
            The `.axes.Axes` subclass that is instantiated.  This parameter
            is incompatible with *projection* and *polar*.  See
            :ref:`axisartist_users-guide-index` for examples.

        zorder : number
            Defaults to 5 (same as `.Axes.legend`).  Adjust higher or lower
            to change whether it is above or below data plotted on the
            parent Axes.

        **kwargs
            Other keyword arguments are passed on to the inset Axes class.

        Returns
        -------
        ax
            The created `~.axes.Axes` instance.

        Examples
        --------
        This example makes two inset Axes, the first is in Axes-relative
        coordinates, and the second in data-coordinates::

            fig, ax = plt.subplots()
            ax.plot(range(10))
            axin1 = ax.inset_axes([0.8, 0.1, 0.15, 0.15])
            axin2 = ax.inset_axes(
                    [5, 7, 2.3, 2.3], transform=ax.transData)

        """
		if transform is None:
			transform = self.transPixel
		return super().inset_axes(bounds, transform=transform, zorder=zorder, **kwargs)