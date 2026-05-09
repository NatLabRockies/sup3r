"""Abstract Loader class merely for loading data from file paths. This data is
always loaded lazily."""

import copy
import datetime
import logging
from abc import ABC, abstractmethod
from datetime import datetime as dt
from functools import cached_property
from typing import Callable

import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.names import FEATURE_NAMES, Dimension
from sup3r.preprocessing.utilities import (
    expand_paths,
    lower_names,
    ordered_dims,
)
from sup3r.utilities.utilities import xr_open_mfdataset

from .utilities import standardize_names, standardize_values

logger = logging.getLogger(__name__)


class BaseLoader(Container, ABC):
    """
    Base loader. "Loads" files so that a `.data` attribute provides access
    to the data in the files as a dask array with shape (lats, lons, time,
    features). This object provides a `__getitem__` method that can be used by
    :class:`~sup3r.preprocessing.samplers.Sampler` objects to build batches or
    by :class:`~sup3r.preprocessing.rasterizers.Rasterizer` objects to derive /
    extract specific features / regions / time_periods.
    """

    BASE_LOADER: Callable = xr_open_mfdataset

    def __init__(
        self,
        file_paths,
        features='all',
        res_kwargs=None,
        chunks='auto',
        feature_aliases=None,
        BaseLoader=None,
    ):
        """
        Parameters
        ----------
        file_paths : str | pathlib.Path | list
            Location(s) of files to load
        features : list | str
            Features to return in loaded dataset. If 'all' then all available
            features will be returned.
        res_kwargs : dict
            Additional keyword arguments passed through to the ``BaseLoader``.
            BaseLoader is usually xr.open_mfdataset for NETCDF files and
            MultiFileResourceX for H5 files.
        chunks : dict | str | None
            Dictionary of chunk sizes to pass through to
            ``dask.array.from_array()`` or ``xr.Dataset().chunk()``. Will be
            converted to a tuple when used in ``from_array()``. These are the
            methods for H5 and NETCDF data, respectively. This argument can
            be "auto" in additional to a dictionary. If this is None then the
            data will not be chunked and instead loaded directly into memory.
        feature_aliases : dict
            Optional dictionary of feature aliases to use when loading data.
            This is useful for renaming features to expected sup3r names.
            For example, {'sp': 'pressure_0m', 'u10': u_10m'}.
        BaseLoader : Callable
            Optional base loader update. The default for H5 files is
            MultiFileResourceX and for NETCDF or ZARR is xarray.open_mfdataset
        """
        logger.info(
            'Loading features: %s from files: %s', features, file_paths
        )
        super().__init__()
        self.res_kwargs = res_kwargs or {}
        self.file_paths = file_paths
        self.chunks = chunks
        BASE_LOADER = BaseLoader or self.BASE_LOADER
        self._res = BASE_LOADER(self.file_paths, **self.res_kwargs)
        data = lower_names(self._load())
        data = self._add_attrs(data)
        data = standardize_values(data)
        feature_aliases = feature_aliases or {}
        feature_names = FEATURE_NAMES.copy()
        feature_names.update(feature_aliases)
        data = standardize_names(data, feature_names).astype(np.float32)
        data = data.transpose(*ordered_dims(data.dims), ...)
        features = list(data.dims) if features == [] else features
        self.data = data[features] if features != 'all' else data

        if 'meta' in self._res:
            self.data.meta = self._res.meta

        if self.chunks is None:
            logger.debug('Pre-loading data into memory for: %s', features)
            self.data.compute()

    def _parse_chunks(self, dims, feature=None):
        """Get chunks for given dimensions from ``self.chunks``."""
        chunks = copy.deepcopy(self.chunks)
        if (
            isinstance(chunks, dict)
            and feature is not None
            and feature in chunks
        ):
            chunks = chunks[feature]
        if isinstance(chunks, dict):
            chunks = {k: v for k, v in chunks.items() if k in dims}
        return chunks

    def _add_attrs(self, data):
        """Add meta data to dataset."""
        attrs = getattr(self._res, 'global_attrs', {})
        attrs = attrs if isinstance(attrs, dict) else {'global_attrs': attrs}
        attrs.update(getattr(self._res, 'attrs', {}))
        attrs['date_modified'] = attrs.get(
            'date_modified', dt.now(datetime.timezone.utc).isoformat()
        )
        attrs['source_files'] = attrs.get('source_files', self.file_paths)
        data.attrs.update(attrs)
        return data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, trace):
        self._res.close()

    @property
    def file_paths(self):
        """Get file paths for input data"""
        return self._file_paths

    @file_paths.setter
    def file_paths(self, file_paths):
        """Set file paths attr and do initial glob / sort

        Parameters
        ----------
        file_paths : str | list
            A list of files to extract raster data from. Each file must have
            the same number of timesteps. Can also pass a string or list of
            strings with a unix-style file path which will be passed through
            glob.glob
        """
        self._file_paths = expand_paths(file_paths)
        msg = (
            f'No valid files provided to {self.__class__.__name__}. '
            f'Received file_paths={file_paths}. Aborting.'
        )
        assert file_paths is not None and len(self._file_paths) > 0, msg

    @abstractmethod
    def _load(self):
        """'Load' data into this container. Does not actually load from disk
        into memory. Just wraps data from files in an xarray.Dataset.

        Returns
        -------
        xr.Dataset
        """

    @cached_property
    @abstractmethod
    def _lat_lon_shape(self):
        """Get shape of lat lon grid only."""

    @cached_property
    @abstractmethod
    def _is_flattened(self):
        """Check if data is flattened or not"""

    @cached_property
    def _lat_lon_dims(self):
        """Get dim names for lat lon grid. Either
        ``Dimension.FLATTENED_SPATIAL`` or ``(Dimension.SOUTH_NORTH,
        Dimension.WEST_EAST)``"""
        if self._is_flattened:
            return (Dimension.FLATTENED_SPATIAL,)
        return Dimension.dims_2d()

    @property
    def _time_independent(self):
        return 'time_index' not in self._res and 'time' not in self._res

    def _is_spatial_dset(self, data):
        """Check if given data is spatial only. We compare against the size of
        the spatial domain."""
        return len(data.shape) == 1 and len(data) == self._lat_lon_shape[0]
