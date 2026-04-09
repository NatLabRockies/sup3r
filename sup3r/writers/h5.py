"""Output handling

TODO: Remove redundant code re. Cachers
"""

import logging
import os

import numpy as np
import pandas as pd

from .base import OutputHandler

logger = logging.getLogger(__name__)


class OutputHandlerH5(OutputHandler):
    """Class to handle writing output to H5 file"""

    @classmethod
    def _write_output(
        cls,
        data,
        features,
        lat_lon,
        times,
        out_file,
        meta_data=None,
        invert_uv=False,
        nn_fill=False,
        max_workers=None,
        row_inds=None,
        col_inds=None,
        gids=None,
        overwrite=False,
    ):
        """Write forward pass output to H5 file

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
        invert_uv : bool
            Whether to convert u and v wind components to windspeed and
            direction
        nn_fill : bool
            Whether to fill data outside of limits with nearest
            neighbour or cap to limits
        max_workers : int | None
            Max workers to use for inverse transform.
        row_inds : np.ndarray
            Array of row indices for the full high resolution grid. This is
            used to collect spatially contiguous data for stitching output
            chunks back together.
        col_inds : np.ndarray
            Array of column indices for the full high resolution grid. This is
            used to collect spatially contiguous data for stitching output
            chunks back together.
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        overwrite : bool
            Whether to overwrite existing output file or skip writing if file
            already exists. Default is False to avoid accidentally overwriting
            files.
        """
        if os.path.exists(out_file):
            if overwrite:
                logger.warning(f'Overwriting existing file at {out_file}.')
            else:
                logger.info(
                    f'File already exists at {out_file}. Skipping write.'
                )
                return
        msg = (
            f'Output data shape ({data.shape}) and lat_lon shape '
            f'({lat_lon.shape}) conflict.'
        )
        assert data.shape[:2] == lat_lon.shape[:-1], msg
        msg = (
            f'Output data shape ({data.shape}) and times shape '
            f'({len(times)}) conflict.'
        )
        assert data.shape[-2] == len(times), msg
        data, features = cls._transform_output(
            data.copy(),
            features,
            lat_lon,
            max_workers=max_workers,
            invert_uv=invert_uv,
            nn_fill=nn_fill,
        )
        gids = (
            gids
            if gids is not None
            else np.arange(np.prod(lat_lon.shape[:-1]))
        )
        row_inds = (
            row_inds if row_inds is not None else np.arange(lat_lon.shape[0])
        )
        col_inds = (
            col_inds if col_inds is not None else np.arange(lat_lon.shape[1])
        )
        meta = pd.DataFrame({
            'gid': gids.flatten(),
            'row_ind': np.repeat(
                row_inds[:, np.newaxis], len(col_inds), axis=1
            ).flatten(),
            'col_ind': np.repeat(
                col_inds[np.newaxis, :], len(row_inds), axis=0
            ).flatten(),
            'latitude': lat_lon[..., 0].flatten(),
            'longitude': lat_lon[..., 1].flatten(),
        })
        data_list = []
        for i, _ in enumerate(features):
            flat_data = data[..., i].reshape((-1, len(times)))
            flat_data = np.transpose(flat_data, (1, 0))
            data_list.append(flat_data)
        cls.write_data(out_file, features, times, data_list, meta, meta_data)
