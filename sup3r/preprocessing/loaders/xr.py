"""Base loading classes. These are containers which also load data from
file_paths and include some sampling ability to interface with batcher
classes."""

import logging
from functools import cached_property
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.preprocessing.names import COORD_NAMES, DIM_NAMES, Dimension
from sup3r.preprocessing.utilities import lower_names, ordered_dims
from sup3r.utilities.utilities import xr_open_mfdataset

from .base import BaseLoader

logger = logging.getLogger(__name__)


class LoaderX(BaseLoader):
    """Base xarray loader. Can load any file type supported by xarray.
    Primarily used to "load" netcdf or zarr files. The ``.data`` attribute
    provides access to the data in the files. This object provides a
    ``__getitem__`` method that can be used by Sampler objects to build batches
    or by other objects to derive / extract specific features / regions /
    time_periods."""

    @classmethod
    def BASE_LOADER(cls, file_paths, **kwargs):
        """Lowest level interface to data."""
        return xr_open_mfdataset(file_paths, **kwargs)

    def _enforce_descending_lats(self, dset):
        """Make sure latitudes are in descending order so that min lat / lon is
        at ``lat_lon[-1, 0]``."""
        invert_lats = not self._is_flattened and (
            dset[Dimension.LATITUDE][-1, 0] > dset[Dimension.LATITUDE][0, 0]
        )
        if invert_lats:
            for var in [*list(Dimension.coords_2d()), *list(dset.data_vars)]:
                if Dimension.SOUTH_NORTH in dset[var].dims:
                    new_var = dset[var].isel(south_north=slice(None, None, -1))
                    dset.update({var: new_var})
        return dset

    @classmethod
    def _enforce_descending_levels(cls, dset):
        """Make sure levels are in descending order so that max pressure is at
        ``level[0]``."""
        invert_levels = (
            dset[Dimension.PRESSURE_LEVEL][-1]
            > dset[Dimension.PRESSURE_LEVEL][0]
            if Dimension.PRESSURE_LEVEL in dset
            else False
        )
        if invert_levels:
            for var in list(dset.data_vars):
                if Dimension.PRESSURE_LEVEL in dset[var].dims:
                    new_var = dset[var].isel(
                        {Dimension.PRESSURE_LEVEL: slice(None, None, -1)}
                    )
                    dset.update(
                        {var: (dset[var].dims, new_var.data, dset[var].attrs)}
                    )
            new_press = dset[Dimension.PRESSURE_LEVEL][::-1]
            dset.update({Dimension.PRESSURE_LEVEL: new_press})
        return dset

    @cached_property
    def _lat_lon_shape(self):
        """Get shape of lat lon grid only."""
        return self._res[Dimension.LATITUDE].shape

    @cached_property
    def _is_flattened(self):
        """Check if dims include a single spatial dimension."""
        check = (
            len(self._lat_lon_shape) == 1
            and self._res[Dimension.LATITUDE].dims
            == self._res[Dimension.LONGITUDE].dims
        )
        return check

    def _get_coords(self, res):
        """Get coordinate dictionary to use in
        ``xr.Dataset().assign_coords()``."""
        lats = res[Dimension.LATITUDE].data.astype(np.float32)
        lons = res[Dimension.LONGITUDE].data.astype(np.float32)

        # remove time dimension if there's a single time step
        if lats.ndim == 3:
            lats = lats.squeeze()
        if lons.ndim == 3:
            lons = lons.squeeze()

        if len(lats.shape) == 1 and not self._is_flattened:
            lons, lats = da.meshgrid(lons, lats)

        dim_names = self._lat_lon_dims
        coords = {
            Dimension.LATITUDE: (dim_names, lats),
            Dimension.LONGITUDE: (dim_names, lons),
        }

        if Dimension.TIME in res:
            if Dimension.TIME in res.indexes:
                times = res.indexes[Dimension.TIME]
            else:
                times = res[Dimension.TIME]

            if hasattr(times, 'to_datetimeindex'):
                times = times.to_datetimeindex()

            coords[Dimension.TIME] = times
        return coords

    def _get_dims(self, res):
        """Get dimension name map using our standard mappping and the names
        used for coordinate dimensions."""
        rename_dims = {k: v for k, v in DIM_NAMES.items() if k in res.dims}
        lat_dims = res[Dimension.LATITUDE].dims
        lon_dims = res[Dimension.LONGITUDE].dims
        if len(lat_dims) == 1 and len(lon_dims) == 1:
            dim_names = self._lat_lon_dims
            rename_dims[lat_dims[0]] = dim_names[0]
            rename_dims[lon_dims[0]] = dim_names[-1]
        else:
            msg = (
                'Latitude and Longitude dimension names are different. '
                'This is weird.'
            )
            if lon_dims != lat_dims:
                logger.warning(msg)
                warn(msg)
            else:
                rename_dims.update(
                    dict(zip(ordered_dims(lat_dims), Dimension.dims_2d()))
                )
        return rename_dims

    def _rechunk_dsets(self, res):
        """Apply given chunk values for each field in res.coords and
        res.data_vars."""
        for dset in [*list(res.coords), *list(res.data_vars)]:
            chunks = self._parse_chunks(dims=res[dset].dims, feature=dset)

            # specifying chunks to xarray.open_mfdataset doesn't automatically
            # apply to coordinates so we do that here
            if chunks != 'auto' or dset in Dimension.coords_2d():
                res[dset] = res[dset].chunk(chunks)
        return res

    def _load(self):
        """Load netcdf ``xarray.Dataset()``."""
        res = lower_names(self._res)
        rename_coords = {
            k: v for k, v in COORD_NAMES.items() if k in res and v not in res
        }
        self._res = res.rename(rename_coords)

        if not all(coord in self._res for coord in Dimension.coords_2d()):
            err = (
                'Could not find valid coordinates in given files: %s\n'
                'Expected: %s\n'
                'Dataset coords: %s'
            )
            logger.error(
                err,
                self.file_paths,
                [str(x) for x in Dimension.coords_2d()],
                list(self._res.coords),
            )
            raise OSError(
                err
                % (
                    self.file_paths,
                    [str(x) for x in Dimension.coords_2d()],
                    list(self._res.coords),
                )
            )

        res = self._res.swap_dims(self._get_dims(self._res))
        res = res.assign_coords(self._get_coords(res))
        res = self._enforce_descending_lats(res)
        res = self._rechunk_dsets(res)
        return self._enforce_descending_levels(res).astype(np.float32)
