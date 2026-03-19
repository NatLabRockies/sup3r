"""Output handling"""

import datetime
import logging
from datetime import datetime as dt

import numpy as np
import xarray as xr

from sup3r.preprocessing.names import Dimension
from sup3r.writers import Cacher

from .base import OutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerNC(OutputHandler):
    """Forward pass OutputHandler for NETCDF files"""

    @classmethod
    def _write_output(
        cls,
        data,
        features,
        lat_lon,
        times,
        out_file,
        meta_data=None,
        max_workers=None,
        invert_uv=False,
        nn_fill=False,
        row_inds=None,
        col_inds=None,
        gids=None,
    ):
        """Write forward pass output to NETCDF file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        lat_lon : ndarray
            Array of high res lat/lon for output data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        times : pd.Datetimeindex
            List of times for high res output data
        out_file : string
            Output file path
        meta_data : dict | None
            Dictionary of meta data from model
        max_workers : int | None
            Max workers to use for inverse transform.
        invert_uv : bool
            Whether to convert u and v wind components to windspeed and
            direction
        nn_fill : bool
            Whether to fill data outside of limits with nearest neighbour or
            cap to limits
        row_inds : np.ndarray
            Array of row indices for the full high resolution grid. This is
            used to help with spatial chunk data collection and should be
            included if the output data is spatially chunked.
        col_inds : np.ndarray
            Array of column indices for the full high resolution grid. This is
            used to help with spatial chunk data collection and should be
            included if the output data is spatially chunked.
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        """
        data, features = cls._transform_output(
            data=data,
            features=features,
            lat_lon=lat_lon,
            invert_uv=invert_uv,
            nn_fill=nn_fill,
            max_workers=max_workers,
        )

        data_vars = {
            Dimension.TIME: times,
            Dimension.LATITUDE: (Dimension.dims_2d(), lat_lon[:, :, 0]),
            Dimension.LONGITUDE: (Dimension.dims_2d(), lat_lon[:, :, 1]),
        }
        if gids is not None:
            data_vars['gids'] = (Dimension.dims_2d(), gids)
        if row_inds is not None and col_inds is not None:
            for dim, inds in zip(Dimension.dims_2d(), [row_inds, col_inds]):
                data_vars[dim] = (dim, inds)
        for i, f in enumerate(features):
            data_vars[f] = (
                (Dimension.TIME, *Dimension.dims_2d()),
                np.transpose(data[..., i], axes=(2, 0, 1)).astype(np.float32),
            )

        if all(d in data_vars for d in Dimension.dims_2d()):
            coords = {dim: data_vars.pop(dim) for dim in Dimension.dims_2d()}
        else:
            coords = {
                coord: data_vars.pop(coord) for coord in Dimension.coords_2d()
            }
        coords[Dimension.TIME] = data_vars.pop(Dimension.TIME)

        attrs = meta_data or {}
        now = dt.now(datetime.timezone.utc).isoformat()
        attrs['date_modified'] = now
        attrs['date_created'] = attrs.get('date_created', now)

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        Cacher._write_single(
            out_file=out_file,
            data=ds,
            features=list(data_vars.keys()),
            max_workers=max_workers,
        )
