"""Sup3r preprocessing module. Here you will find things that have access to
data, which we call ``Containers``. ``Loaders``, ``Rasterizers``, ``Samplers``,
``Derivers``, ``Handlers``, ``Batchers``, etc, are all subclasses of
``Containers.`` Rather than having a single object that does everything -
extract data, compute features, sample the data for batching, split into train
and val, etc, we have fundamental objects that do one of these things and we
build multi-purpose objects with class factories. These factory generated
objects are DataHandlers and BatchHandlers.

If you want to extract a specific spatiotemporal extent from a data file then
use :class:`.Rasterizer`. If you want to split into a test and validation set
then use :class:`.Rasterizer` to extract different temporal extents separately.
If you've already rasterized data and written that to a file and then want to
sample that data for batches, then use a :class:`.Loader` (or a
:class:`.DataHandler`), and give that object to a :class:`.BatchHandler`. If
you want to have training and validation batches then load those separate data
sets, and provide these to :class:`.BatchHandler`. If you want to have a
BatchQueue containing pairs of low / high res data, rather than coarsening
high-res to get low res, then load lr and hr data with separate Loaders or
DataHandlers, use :class:`.DualRasterizer` to match the lr and hr grids, and
provide this to :class:`.DualBatchHandler`.
"""

from .base import Container, Sup3rDataset
from .batch_handlers import (
    BatchHandler,
    BatchHandlerCC,
    BatchHandlerDC,
    BatchHandlerMom1,
    BatchHandlerMom1SF,
    BatchHandlerMom2,
    BatchHandlerMom2Sep,
    BatchHandlerMom2SepSF,
    BatchHandlerMom2SF,
    DualBatchHandler,
)
from .batch_queues import BatchQueueDC, DualBatchQueue, SingleBatchQueue
from .collections import Collection, StatsCollection
from .data_handlers import (
    DailyDataHandler,
    DataHandler,
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
    DataHandlerNCforCC,
    DataHandlerNCforCCwithPowerLaw,
    ExoData,
    ExoDataHandler,
)
from .derivers import Deriver
from .loaders import Loader, LoaderH5, LoaderNC
from .names import COORD_NAMES, DIM_NAMES, FEATURE_NAMES, Dimension
from .rasterizers import (
    BaseExoRasterizer,
    DerivedFeatureRasterizer,
    DualRasterizer,
    ExoRasterizer,
    ObsRasterizer,
    Rasterizer,
)
from .samplers import (
    DualSampler,
    DualSamplerCC,
    Sampler,
    SamplerDC,
)
