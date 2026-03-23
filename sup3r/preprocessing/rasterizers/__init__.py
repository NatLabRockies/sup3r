"""Container subclass with methods for extracting a specific spatiotemporal
extents from data. :class:`.Rasterizer` objects mostly operate on
:class:`~sup3r.preprocessing.loaders.Loader` objects, which just load data from
files but do not do anything else to the data. :class:`.Rasterizer` objects are
mostly operated on by :class:`~sup3r.preprocessing.derivers.Deriver` objects,
which derive new features from the data contained in :class:`.Rasterizer`
objects."""

from .base import BaseRasterizer
from .dual import DualRasterizer
from .exo import (
    BaseExoRasterizer,
    DerivedFeatureRasterizer,
    ExoRasterizer,
    ObsRasterizer,
)
from .extended import Rasterizer
