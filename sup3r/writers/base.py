"""Output handling

TODO: OutputHandlers should be combined with Cacher objects.
"""

import json
import logging
import os
import re
from abc import abstractmethod

import dask
import numpy as np
import pandas as pd
from rex.outputs import Outputs as BaseRexOutputs
from scipy.interpolate import griddata

from sup3r.preprocessing.derivers.utilities import (
    invert_uv,
    parse_feature,
)
from sup3r.preprocessing.utilities import get_time_index_freqs
from sup3r.utilities import VERSION_RECORD
from sup3r.utilities.utilities import (
    enforce_limits,
    get_dset_attrs,
    get_tmp_file,
    pd_date_range,
    safe_serialize,
    xr_open_mfdataset,
)

logger = logging.getLogger(__name__)


class OutputMixin:
    """Methods used by various Output and Collection classes"""

    @staticmethod
    def get_time_dim_name(filepath):
        """Get the name of the time dimension in the given file

        Parameters
        ----------
        filepath : str
            Path to the file

        Returns
        -------
        time_key : str
            Name of the time dimension in the given file
        """

        handle = xr_open_mfdataset(filepath)
        valid_vars = set(handle.dims)
        time_key = list({'time', 'Time'}.intersection(valid_vars))
        if len(time_key) > 0:
            return time_key[0]
        return 'time'

    @staticmethod
    def _init_h5(out_file, time_index, meta, global_attrs):
        """Initialize the output h5 file to save data to.

        Parameters
        ----------
        out_file : str
            Output file path - must not yet exist.
        time_index : pd.datetimeindex
            Full datetime index of final output data.
        meta : pd.DataFrame
            Full meta dataframe for the final output data.
        global_attrs : dict
            Namespace of file-global attributes for the final output data.
        """

        with RexOutputs(out_file, mode='w-') as f:
            logger.info('Initializing output file: {}'.format(out_file))
            logger.info(
                'Initializing output file with shape {} '
                'and meta data:\n{}'.format((len(time_index), len(meta)), meta)
            )
            f.time_index = time_index
            f.meta = meta
            f.run_attrs = global_attrs

    @classmethod
    def _ensure_dset_in_output(cls, out_file, dset, data=None):
        """Ensure that dset is initialized in out_file and initialize if not.

        Parameters
        ----------
        out_file : str
            Pre-existing H5 file output path
        dset : str
            Dataset name
        data : Union[np.ndarray, da.core.Array] | None
            Optional data to write to dataset if initializing.
        """

        with RexOutputs(out_file, mode='a') as f:
            if dset not in f.dsets:
                attrs, dtype = get_dset_attrs(dset)
                logger.info(
                    'Initializing dataset "{}" with shape {} and '
                    'dtype {}'.format(dset, f.shape, dtype)
                )
                f._create_dset(
                    dset,
                    f.shape,
                    dtype,
                    attrs=attrs,
                    data=data,
                    chunks=attrs.get('chunks', None),
                )

    @classmethod
    def write_data(
        cls, out_file, dsets, time_index, data_list, meta, global_attrs=None
    ):
        """Write list of datasets to out_file.

        Parameters
        ----------
        out_file : str
            Pre-existing H5 file output path
        dsets : list
            list of datasets to write to out_file
        time_index : pd.DatetimeIndex()
            Pandas datetime index to use for file time_index.
        data_list : list
            List of np.ndarray objects to write to out_file
        meta : pd.DataFrame
            Full meta dataframe for the final output data.
        global_attrs : dict
            Namespace of file-global attributes for the final output data.
        """
        tmp_file = get_tmp_file(out_file)
        with RexOutputs(tmp_file, 'w') as fh:
            fh.meta = meta
            fh.time_index = time_index

            for dset, data in zip(dsets, data_list):
                attrs, dtype = get_dset_attrs(dset)
                fh.add_dataset(
                    tmp_file,
                    dset,
                    data,
                    dtype=dtype,
                    attrs=attrs,
                    chunks=attrs['chunks'],
                )
                logger.info(f'Added {dset} to output file {out_file}.')

            if global_attrs is not None:
                attrs = {
                    k: v if isinstance(v, str) else safe_serialize(v)
                    for k, v in global_attrs.items()
                }
                fh.run_attrs = attrs

        os.replace(tmp_file, out_file)
        msg = (
            'Saved output of size '
            f'{(len(data_list), *data_list[0].shape)} to: {out_file}'
        )
        logger.info(msg)


class RexOutputs(BaseRexOutputs):
    """Base class to handle NREL h5 formatted output data"""

    @property
    def full_version_record(self):
        """Get record of versions for dependencies

        Returns
        -------
        dict
            Dictionary of package versions for dependencies
        """
        versions = super().full_version_record
        versions.update(VERSION_RECORD)
        return versions

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = VERSION_RECORD['sup3r']
        self.h5.attrs['full_version_record'] = json.dumps(
            self.full_version_record
        )
        self.h5.attrs['package'] = 'sup3r'


class OutputHandler(OutputMixin):
    """Class to handle forward pass output. This includes transforming features
    back to their original form and outputting to the correct file format.
    """

    @classmethod
    def get_renamed_features(cls, features):
        """Rename features based on transformation from u/v to
        windspeed/winddirection

        Parameters
        ----------
        features : list
            List of output features

        Returns
        -------
        list
            List of renamed features u/v -> windspeed/winddirection for each
            height
        """
        heights = [
            parse_feature(f).height
            for f in features
            if re.match('u_(.*?)m'.lower(), f.lower())
        ]
        renamed_features = features.copy()

        for height in heights:
            u_idx = features.index(f'u_{height}m')
            v_idx = features.index(f'v_{height}m')

            renamed_features[u_idx] = f'windspeed_{height}m'
            renamed_features[v_idx] = f'winddirection_{height}m'

        return renamed_features

    @classmethod
    def invert_uv_features(cls, data, features, lat_lon, max_workers=None):
        """Invert U/V to windspeed and winddirection. Performed in place.

        Parameters
        ----------
        data : ndarray
            High res data from forward pass
            (spatial_1, spatial_2, temporal, features)
        features : list
            List of output features. If this doesn't contain any names matching
            u_*m, this method will do nothing.
        lat_lon : ndarray
            High res lat/lon array
            (spatial_1, spatial_2, 2)
        max_workers : int | None
            Max workers to use for inverse transform. If None the maximum
            possible will be used
        """

        heights = [
            parse_feature(f).height
            for f in features
            if re.match('u_(.*?)m'.lower(), f.lower())
        ]

        if heights:
            logger.info(
                'Converting u/v to ws/wd for H5 output with max_workers=%s',
                max_workers,
            )
            logger.debug(
                'Found heights %s for output features %s', heights, features
            )

        tasks = []
        for height in heights:
            u_idx = features.index(f'u_{height}m')
            v_idx = features.index(f'v_{height}m')
            task = dask.delayed(cls.invert_uv_single_pair)(
                data, lat_lon, u_idx, v_idx
            )
            tasks.append(task)
            logger.info('Added %s futures to convert u/v to ws/wd', len(tasks))
        if max_workers == 1:
            dask.compute(*tasks, scheduler='single-threaded')
        else:
            dask.compute(*tasks, scheduler='threads', num_workers=max_workers)
        logger.info('Finished converting u/v to ws/wd')

    @staticmethod
    def invert_uv_single_pair(data, lat_lon, u_idx, v_idx):
        """Perform inverse transform in place on a single u/v pair.

        Parameters
        ----------
        data : ndarray
            High res data from forward pass
            (spatial_1, spatial_2, temporal, features)
        lat_lon : ndarray
            High res lat/lon array
            (spatial_1, spatial_2, 2)
        u_idx : int
            Index in data for U component to transform
        v_idx : int
            Index in data for V component to transform
        """
        ws, wd = invert_uv(data[..., u_idx], data[..., v_idx], lat_lon)
        data[..., u_idx] = ws
        data[..., v_idx] = wd

    @classmethod
    def _transform_output(
        cls,
        data,
        features,
        lat_lon,
        invert_uv=False,
        nn_fill=False,
        max_workers=None,
    ):
        """Transform output data before writing to H5 file

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
        invert_uv : bool
            Whether to convert u and v wind components to windspeed and
            direction
        nn_fill : bool
            Whether to fill data outside of accepted limits (e.g. relative
            humidity 0-100) with nearest neighbour or cap to limits. If true
            then any values outside of limits will be filled with the nearest
            neighbour value within limits. If false then values will be capped
            to limits. This is only used for writing output and does not affect
            the data passed through the generator.
        max_workers : int | None
            Max workers to use for inverse transform. If None the max_workers
            will be estimated based on memory limits.
        """
        if invert_uv and any(
            re.match('u_(.*?)m'.lower(), f.lower())
            or re.match('v_(.*?)m'.lower(), f.lower())
            for f in features
        ):
            cls.invert_uv_features(
                data, features, lat_lon, max_workers=max_workers
            )
            features = cls.get_renamed_features(features)
        data = enforce_limits(features=features, data=data, nn_fill=nn_fill)
        return data, features

    @staticmethod
    def pad_lat_lon(lat_lon):
        """Pad lat lon grid with additional rows and columns to use for
        interpolation routine

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        shape : tuple
            (lons, lats) Shape of high res grid

        Returns
        -------
        lat_lon : ndarray
            Array of padded lat lons
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        """

        # add row and column to boundaries
        padded_grid = np.zeros((2 + lat_lon.shape[0], 2 + lat_lon.shape[1], 2))

        # fill in interior values
        padded_grid[1:-1, 1:-1, :] = lat_lon

        # define edge spacing
        left_diffs = padded_grid[:, 2, 1] - padded_grid[:, 1, 1]
        right_diffs = padded_grid[:, -2, 1] - padded_grid[:, -3, 1]
        top_diffs = padded_grid[1, :, 0] - padded_grid[2, :, 0]
        bottom_diffs = padded_grid[-3, :, 0] - padded_grid[-2, :, 0]

        # use edge spacing to define new boundary values
        padded_grid[:, 0, 1] = padded_grid[:, 1, 1] - left_diffs
        padded_grid[:, 0, 0] = padded_grid[:, 1, 0]
        padded_grid[:, -1, 1] = padded_grid[:, -2, 1] + right_diffs
        padded_grid[:, -1, 0] = padded_grid[:, -2, 0]
        padded_grid[0, :, 0] = padded_grid[1, :, 0] + top_diffs
        padded_grid[0, :, 1] = padded_grid[1, :, 1]
        padded_grid[-1, :, 0] = padded_grid[-2, :, 0] - bottom_diffs
        padded_grid[-1, :, 1] = padded_grid[-2, :, 1]

        # use surrounding cells to define corner values
        # top left
        padded_grid[0, 0, 0] = padded_grid[0, 1, 0]
        padded_grid[0, 0, 1] = padded_grid[1, 0, 1]
        # top right
        padded_grid[0, -1, 0] = padded_grid[0, -2, 0]
        padded_grid[0, -1, 1] = padded_grid[1, -1, 1]
        # bottom left
        padded_grid[-1, 0, 0] = padded_grid[-1, 1, 0]
        padded_grid[-1, 0, 1] = padded_grid[-2, 0, 1]
        # bottom right
        padded_grid[-1, -1, 0] = padded_grid[-1, -2, 0]
        padded_grid[-1, -1, 1] = padded_grid[-2, -1, 1]

        return padded_grid

    @staticmethod
    def is_increasing_lons(lat_lon):
        """Check if longitudes are in increasing order. Need to check this
        for interpolation routine. This is primarily to identify whether the
        lons go through the 180 -> -180 boundary, which creates a
        discontinuity. For example, [130, 180, -130, -80]. If any lons go from
        positive to negative the lons need to be shifted to the range 0-360.

        Parameters
        ----------
        lat_lon : ndarray
            Array of lat/lon for input data.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)

        Returns
        -------
        bool
            Whether all lons are in increasing order or not

        """
        for i in range(lat_lon.shape[0]):
            if lat_lon[i, :, 1][-1] < lat_lon[i, :, 1][0]:
                return False
        return True

    @classmethod
    def get_lat_lon(cls, low_res_lat_lon, shape):
        """Get lat lon arrays for high res output file

        Parameters
        ----------
        low_res_lat_lon : ndarray
            Array of lat/lon for input data. Longitudes must be arranged in
            a counter-clockwise direction (when looking down from above the
            north pole). e.g. [-50, -25, 25, 50] or [130, 180, -130, -80]. The
            latter passes through the 180 -> -180 boundary and will be
            temporarily shifted to the 0-360 range before interpolation.
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        shape : tuple
            (lons, lats) Shape of high res grid

        Returns
        -------
        lat_lon : ndarray
            Array of lat lons for high res output file
            (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        """
        assert low_res_lat_lon.shape[0] > 1 and low_res_lat_lon.shape[1] > 1, (
            'low res lat/lon must have at least 2 rows and 2 columns'
        )

        logger.debug('Getting high resolution lat / lon grid')

        # ensure lons are between -180 and 180
        logger.debug('Ensuring correct longitude range.')
        low_res_lat_lon[..., 1] = (low_res_lat_lon[..., 1] + 180) % 360 - 180

        # check if lons go through the 180 -> -180 boundary.
        if not cls.is_increasing_lons(low_res_lat_lon):
            logger.debug('Ensuring increasing longitudes.')
            low_res_lat_lon[..., 1] = (low_res_lat_lon[..., 1] + 360) % 360

        # pad lat lon grid
        logger.debug('Padding low-res lat / lon grid.')
        padded_grid = cls.pad_lat_lon(low_res_lat_lon)
        lats = padded_grid[..., 0].flatten()
        lons = padded_grid[..., 1].flatten()

        lr_y, lr_x = low_res_lat_lon.shape[:-1]
        hr_y, hr_x = shape

        # assume outer bounds of mesh (0, 10) w/ points on inside of that range
        y = np.linspace(0, 10, lr_y, endpoint=False) + 5 / lr_y
        x = np.linspace(0, 10, lr_x, endpoint=False) + 5 / lr_x

        # add values due to padding
        y = np.concatenate([[y[0] - 10 / lr_y], y, [y[-1] + 10 / lr_y]])
        x = np.concatenate([[x[0] - 10 / lr_x], x, [x[-1] + 10 / lr_x]])

        # remesh (0, 10) with high res spacing
        new_y = np.arange(0, 10, 10 / hr_y) + 5 / hr_y
        new_x = np.arange(0, 10, 10 / hr_x) + 5 / hr_x

        logger.debug('Running meshgrid.')
        X, Y = np.meshgrid(x, y, copy=False)
        old = np.array([Y.flatten(), X.flatten()], dtype=np.float32).T
        X, Y = np.meshgrid(new_x, new_y, copy=False)
        new = np.array([Y.flatten(), X.flatten()], dtype=np.float32).T

        logger.debug('Running griddata.')
        lons = griddata(old, lons, new)
        lats = griddata(old, lats, new)

        lons = (lons + 180) % 360 - 180
        lat_lon = np.dstack((lats.reshape(shape), lons.reshape(shape)))
        logger.debug('Finished getting high resolution lat / lon grid')

        return lat_lon

    @staticmethod
    def get_times(low_res_times, shape):
        """Get array of times for high res output file

        Parameters
        ----------
        low_res_times : pd.Datetimeindex
            List of times for low res input data. If there is only a single low
            res timestep, it is assumed the data is daily.
        shape : int
            Number of time steps for high res time array

        Returns
        -------
        ndarray
            Array of times for high res output file.
        """
        logger.debug('Getting high resolution time indices')
        logger.debug(
            f'Low res times: {low_res_times[0]} to {low_res_times[-1]}'
        )
        t_enhance = int(shape / len(low_res_times))

        offset, _ = get_time_index_freqs(low_res_times)
        freq = pd.tseries.offsets.DateOffset(
            seconds=int(offset.seconds / t_enhance)
        )
        times = pd_date_range(
            low_res_times[0], low_res_times[-1] + offset, freq=freq
        )[:-1]

        has_leap = any((low_res_times.month == 2) & (low_res_times.day == 29))
        if not has_leap:
            leap_mask = (times.month == 2) & (times.day == 29)
            times = times[~leap_mask]

        logger.debug(f'High res times: {times[0]} to {times[-1]}')
        assert len(times) == shape, (
            f'High res times length {len(times)} does not match expected '
            f'shape {shape}'
        )
        return times

    @classmethod
    @abstractmethod
    def _write_output(
        cls,
        data,
        features,
        lat_lon,
        times,
        out_file,
        meta_data,
        invert_uv=False,
        nn_fill=False,
        max_workers=None,
        row_inds=None,
        col_inds=None,
        gids=None,
        overwrite=False,
    ):
        """Write output to file with specified times and lats/lons"""

    @classmethod
    def write_output(
        cls,
        data,
        features,
        low_res_lat_lon,
        low_res_times,
        out_file,
        meta_data=None,
        invert_uv=False,
        nn_fill=False,
        max_workers=None,
        gids=None,
        row_inds=None,
        col_inds=None,
        overwrite=False,
    ):
        """Write forward pass output to file

        Parameters
        ----------
        data : ndarray
            (spatial_1, spatial_2, temporal, features)
            High resolution forward pass output
        features : list
            List of feature names corresponding to the last dimension of data
        low_res_lat_lon : ndarray
            Array of lat/lon for input data. (spatial_1, spatial_2, 2)
            Last dimension has ordering (lat, lon)
        low_res_times : pd.Datetimeindex
            List of times for low res source data
        out_file : string
            Output file path
        meta_data : dict | None
            Dictionary of meta data from model
        invert_uv : bool
            Whether to convert u and v wind components to windspeed and
            direction
        nn_fill : bool
            Whether to fill data outside of limits with nearest neighbour or
            cap to limits
        max_workers : int | None
            Max workers to use for inverse uv transform. If None the
            max_workers will be estimated based on memory limits.
        gids : list
            List of coordinate indices used to label each lat lon pair and to
            help with spatial chunk data collection
        row_inds : np.ndarray
            Array of row indices for the full high resolution grid. This is
            used to collect spatially contiguous data for stitching output
            chunks back together.
        col_inds : np.ndarray
            Array of column indices for the full high resolution grid. This is
            used to collect spatially contiguous data for stitching output
            chunks back together.
        overwrite : bool
            Whether to overwrite existing output file or skip writing if file
            already exists. Default is False to avoid accidentally overwriting
            files.
        """
        lat_lon = cls.get_lat_lon(low_res_lat_lon, data.shape[:2])
        times = cls.get_times(low_res_times, data.shape[-2])
        cls._write_output(
            data,
            features,
            lat_lon,
            times,
            out_file,
            meta_data=meta_data,
            invert_uv=invert_uv,
            nn_fill=nn_fill,
            max_workers=max_workers,
            row_inds=row_inds,
            col_inds=col_inds,
            gids=gids,
            overwrite=overwrite,
        )
