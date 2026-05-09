"""Base loading class for H5 files.

TODO: Explore replacing rex handlers with xarray. xarray should be able to
load H5 files fine. We would still need get_raster_index method in Rasterizers
though.
"""

import logging
from functools import cached_property
from warnings import warn

import dask.array as da
import numpy as np
import xarray as xr
from rex import MultiFileWindX

from sup3r.preprocessing.names import Dimension

from .base import BaseLoader

logger = logging.getLogger(__name__)


class LoaderH5(BaseLoader):
    """Base H5 loader. "Loads" h5 files so that a `.data` attribute
    provides access to the data in the files. This object provides a
    `__getitem__` method that can be used by
    :class:`~sup3r.preprocessing.samplers.Sampler` objects to build batches or
    by :class:`~sup3r.preprocessing.rasterizers.Rasterizer` objects to derive /
    extract specific features / regions / time_periods.
    """

    BASE_LOADER = MultiFileWindX

    @property
    def _time_steps(self):
        return (
            len(self._res['time_index'])
            if not self._time_independent
            else None
        )

    @cached_property
    def _lat_lon_shape(self):
        """Get shape of spatial domain only."""
        if 'latitude' in self._res.h5:
            return self._res.h5['latitude'].shape
        return self._res.h5['meta']['latitude'].shape

    @cached_property
    def _is_flattened(self):
        """Check if dims include a single spatial dimension."""
        return len(self._lat_lon_shape) == 1

    def _res_shape(self):
        """Get shape of H5 file.

        Note
        ----
        Flattened files are 2D but we have 3D H5 files available through
        caching and bias correction factor calculations."""
        return (
            self._lat_lon_shape
            if self._time_independent
            else (self._time_steps, *self._lat_lon_shape)
        )

    def _get_coords(self, dims):
        """Get coords dict for xr.Dataset construction."""
        coords: dict[str, tuple] = {}
        if not self._time_independent:
            coords[Dimension.TIME] = self._res['time_index']
        coord_base = (
            self._res.h5
            if 'latitude' in self._res.h5
            else self._res.h5['meta']
        )
        coord_dims = dims[-len(self._lat_lon_shape) :]
        chunks = self._parse_chunks(coord_dims)
        lats = da.asarray(
            coord_base['latitude'], dtype=np.float32, chunks=chunks
        )
        lats = (coord_dims, lats)
        lons = da.asarray(
            coord_base['longitude'], dtype=np.float32, chunks=chunks
        )
        lons = (coord_dims, lons)
        coords.update({Dimension.LATITUDE: lats, Dimension.LONGITUDE: lons})
        return coords

    def _get_dset_tuple(self, dset, dims, chunks):
        """Get tuple of (dims, array, attrs) for given dataset. Used in
        data_vars entries. This accounts for multiple data shapes and
        dimensions. e.g. Data can be 2D spatial only, 2D flattened
        spatiotemporal, 3D spatiotemporal, 4D spatiotemporal (with presssure
        levels), etc
        """
        # if self._res includes time-dependent and time-independent variables
        # and chunks is 3-tuple we only use the spatial chunk for
        # time-indepdent variables
        dset_chunks = chunks
        if len(chunks) == 3 and len(self._res.h5[dset].shape) == 2:
            dset_chunks = chunks[-len(self._res.h5[dset].shape)]
        arr = da.asarray(
            self._res.h5[dset], dtype=np.float32, chunks=dset_chunks
        )
        arr /= self.scale_factor(dset)
        if len(arr.shape) == 4:
            msg = (
                f'{dset} array is 4 dimensional. Assuming this is an array '
                'of spatiotemporal quantiles.'
            )
            logger.warning(msg)
            warn(msg)
            arr_dims = Dimension.dims_4d_bc()
        elif len(arr.shape) == 3 and self._time_independent:
            msg = (
                f'{dset} array is 3 dimensional but {self.file_paths} has '
                f'no time index. Assuming this is an array of bias correction '
                'factors.'
            )
            logger.warning(msg)
            warn(msg)
            if arr.shape[-1] == 1:
                arr_dims = (*Dimension.dims_2d(), Dimension.GLOBAL_TIME)
            else:
                arr_dims = Dimension.dims_3d()
        elif self._is_spatial_dset(arr):
            arr_dims = (Dimension.FLATTENED_SPATIAL,)
        elif len(arr.shape) == 2:
            arr_dims = dims[-len(arr.shape) :]
        elif len(arr.shape) == 1:
            msg = (
                f'Received 1D feature "{dset}" with shape that does not equal '
                'the length of the meta nor the time_index.'
            )
            is_ts = not self._time_independent and len(arr) == self._time_steps
            assert is_ts, msg
            arr_dims = (Dimension.TIME,)
        else:
            arr_dims = dims[: len(arr.shape)]
        return (arr_dims, arr, dict(self._res.h5[dset].attrs))

    def _parse_chunks(self, dims, feature=None):
        """Get chunks for given dimensions from ``self.chunks``."""
        chunks = super()._parse_chunks(dims=dims, feature=feature)
        if not isinstance(chunks, dict):
            return chunks
        return tuple(chunks.get(d, 'auto') for d in dims)

    def _check_for_elevation(self, data_vars, dims, chunks):
        """Check if this is a flattened h5 file with elevation data and add
        elevation to data_vars if it is."""

        flattened_with_elevation = (
            len(self._lat_lon_shape) == 1
            and hasattr(self._res, 'meta')
            and 'elevation' in self._res.meta
        )
        if flattened_with_elevation:
            elev = self._res.meta['elevation'].values.astype(np.float32)
            elev = da.asarray(elev)
            if not self._time_independent:
                t_steps = len(self._res['time_index'])
                elev = da.repeat(elev[None, ...], t_steps, axis=0)
            elev = elev.rechunk(chunks)
            data_vars['elevation'] = (dims, elev)
        return data_vars

    def _get_data_vars(self, dims):
        """Define data_vars dict for xr.Dataset construction."""
        data_vars = {}
        logger.debug('Rechunking features with chunks: %s', self.chunks)
        chunks = self._parse_chunks(dims)
        data_vars = self._check_for_elevation(
            data_vars, dims=dims, chunks=chunks
        )

        feats = set(self._res.h5.datasets)
        exclude = {
            'meta',
            'time_index',
            'coordinates',
            'latitude',
            'longitude',
        }
        for f in feats - exclude:
            data_vars[f] = self._get_dset_tuple(
                dset=f, dims=dims, chunks=chunks
            )
        return data_vars

    def _get_dims(self):
        """Get tuple of named dims for dataset."""
        if len(self._lat_lon_shape) == 2:
            dims = Dimension.dims_2d()
        else:
            dims = (Dimension.FLATTENED_SPATIAL,)
        if not self._time_independent:
            dims = (Dimension.TIME, *dims)
        return dims

    def _load(self) -> xr.Dataset:
        """Wrap data in xarray.Dataset(). Handle differences with flattened and
        cached h5."""
        dims = self._get_dims()
        coords = self._get_coords(dims)
        data_vars = {
            k: v
            for k, v in self._get_data_vars(dims).items()
            if k not in coords
        }
        return xr.Dataset(coords=coords, data_vars=data_vars).astype(
            np.float32
        )

    def scale_factor(self, feature):
        """Get scale factor for given feature. Data is stored in scaled form to
        reduce memory."""
        feat = feature if feature in self._res.datasets else feature.lower()
        feat = self._res.h5[feat]
        return np.float32(
            1.0
            if not hasattr(feat, 'attrs')
            else feat.attrs.get('scale_factor', 1.0)
        )
