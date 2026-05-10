"""Extended ``Rasterizer`` that can rasterize flattened data."""

import logging
import os

import numpy as np
import xarray as xr

from sup3r.preprocessing.loaders import Loader
from sup3r.preprocessing.names import Dimension

from .base import BaseRasterizer

logger = logging.getLogger(__name__)


class Rasterizer(BaseRasterizer):
    """Extended `Rasterizer` class which also handles the flattened data format
    used for some H5 files (e.g. Wind Toolkit or NSRDB data), and rasterizes
    directly from file paths rather than taking a Loader as input"""

    def __init__(
        self,
        file_paths,
        features='all',
        res_kwargs=None,
        chunks='auto',
        target=None,
        shape=None,
        time_slice=slice(None),
        threshold=None,
        raster_file=None,
        max_delta=20,
        feature_aliases=None,
        BaseLoader=None,
    ):
        """
        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to LoaderClass
        features : list | str
            Features to return in loaded dataset. If 'all' then all
            available features will be returned.
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
        target : tuple
            (lat, lon) lower left corner of raster. Either need
            target+shape or raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or
            raster_file.
        time_slice : slice | list
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1) the full
            time dimension is selected. Can be also be a list ``[start, stop,
            step]``
        threshold : float
            Nearest neighbor euclidean distance threshold. If the
            coordinates are more than this value away from the target
            lat/lon, an error is raised.
        raster_file : str | None
            File for raster_index array for the corresponding target and shape.
            If specified the raster_index will be loaded from the file if it
            exists or written to the file if it does not yet exist. If None and
            raster_index is not provided raster_index will be calculated
            directly. Either need target+shape, raster_file, or raster_index
            input.
        max_delta : int
            Optional maximum limit on the raster shape that is retrieved at
            once. If shape is (20, 20) and max_delta=10, the full raster will
            be retrieved in four chunks of (10, 10). This helps adapt to
            non-regular grids that curve over large distances.
        feature_aliases : dict
            Optional dictionary of feature aliases to use when loading data.
            This is useful for renaming features to expected sup3r names.
            For example, {'sp': 'pressure_0m', 'u10': u_10m'}.
        BaseLoader : Callable
            Optional base loader method update. This is a function which
            takes `file_paths` and `**kwargs` and returns an initialized
            base loader with those arguments. The default for h5 is a
            method which returns MultiFileWindX(file_paths, **kwargs) and
            for nc the default is
            xarray.open_mfdataset(file_paths, **kwargs)
        """
        self.raster_file = raster_file
        self.max_delta = max_delta
        preload = chunks is None
        self.loader = Loader(
            file_paths,
            features=features,
            res_kwargs=res_kwargs,
            chunks='auto' if preload else chunks,
            feature_aliases=feature_aliases,
            BaseLoader=BaseLoader,
        )
        super().__init__(
            loader=self.loader,
            features=features,
            target=target,
            shape=shape,
            time_slice=time_slice,
            threshold=threshold,
        )
        if self.raster_file is not None and not os.path.exists(
            self.raster_file
        ):
            self.save_raster_index()

        if preload:
            self.data.compute()

    def rasterize_data(self):
        """Get rasterized data."""
        if not self.loader.flattened:
            return super().rasterize_data()
        return self._rasterize_flat_data()

    def _rasterize_flat_data(self):
        """Rasterize data from flattened source data, usually coming from WTK
        or NSRDB data."""
        dims = (Dimension.SOUTH_NORTH, Dimension.WEST_EAST)
        coords = {
            Dimension.LATITUDE: (dims, self.lat_lon[..., 0]),
            Dimension.LONGITUDE: (dims, self.lat_lon[..., 1]),
            Dimension.TIME: self.time_index,
        }
        data_vars = {}
        feats = list(self.loader.data_vars)
        data = self.loader[feats].isel(
            **{Dimension.FLATTENED_SPATIAL: self.raster_index.flatten()}
        )
        for f in feats:
            if Dimension.TIME in self.loader[f].dims:
                dat = data[f].isel({Dimension.TIME: self.time_slice})
                dat = dat.data.reshape(
                    (*self.grid_shape, len(self.time_index))
                )
                data_vars[f] = ((*dims, Dimension.TIME), dat)
            else:
                dat = data[f].data.reshape(self.grid_shape)
                data_vars[f] = (dims, dat)
        return xr.Dataset(
            coords=coords, data_vars=data_vars, attrs=self.loader.attrs
        )

    def save_raster_index(self):
        """Save raster index to cache file."""
        os.makedirs(os.path.dirname(self.raster_file), exist_ok=True)
        np.savetxt(self.raster_file, self.raster_index)
        logger.debug('Saved raster_index to %s', self.raster_file)

    def get_raster_index(self):
        """Get set of slices or indices selecting the requested region from
        the contained data."""

        if not self.loader.flattened:
            return super().get_raster_index()
        return self._get_flat_data_raster_index()

    def _get_flat_data_raster_index(self):
        """Get raster index for the flattened source data, which usually comes
        from WTK or NSRDB data."""

        if self.raster_file is None or not os.path.exists(self.raster_file):
            logger.debug(
                'Calculating raster_index for target=%s, shape=%s.',
                self._target,
                self._grid_shape,
            )
            msg = ('Either shape + target or a raster_file must be provided '
                   'for flattened data rasterization.')
            assert (
                self._target is not None and self._grid_shape is not None
            ), msg
            raster_index = self.loader._res.get_raster_index(
                self._target, self._grid_shape, max_delta=self.max_delta
            )
        else:
            raster_index = np.loadtxt(self.raster_file).astype(np.int32)
            logger.debug('Loaded raster_index from %s', self.raster_file)

        return raster_index

    def get_lat_lon(self):
        """Get the 2D array of coordinates corresponding to the requested
        target and shape."""

        if not self.loader.flattened:
            return super().get_lat_lon()
        return self._get_flat_data_lat_lon()

    def _get_flat_data_lat_lon(self):
        """Get lat lon for flattened source data. Output is shape (y, x, 2)
        where 2 is (lat, lon)"""
        if hasattr(self.full_lat_lon, 'vindex'):
            return self.full_lat_lon.vindex[self.raster_index]
        return self.full_lat_lon[self.raster_index]
