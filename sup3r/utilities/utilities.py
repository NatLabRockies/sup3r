"""Miscellaneous utilities shared across multiple modules"""

import json
import logging
import os
import random
import re
import string
import time
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from packaging import version
from scipy import ndimage as nd

ATTR_DIR = os.path.dirname(os.path.realpath(__file__))
ATTR_FP = os.path.join(ATTR_DIR, 'output_attrs.json')
with open(ATTR_FP) as f:
    OUTPUT_ATTRS = json.load(f)

RANDOM_GENERATOR = np.random.default_rng(seed=42)

logger = logging.getLogger(__name__)


def encode_str_times(times):
    """Encode list of string times to byte strings for H5 storage."""
    times = pd.DatetimeIndex(times).astype(str)
    return [t.encode('utf-8') for t in times]


def decode_str_times(times):
    """Decode list of byte string times from H5 storage to normal strings."""
    if isinstance(times[0], bytes):
        times = [t.decode('utf-8') for t in times]
    return pd.DatetimeIndex(times)


def get_tmp_file(file):
    """Get a temporary file name based on the input file name."""
    if not isinstance(file, str):
        raise TypeError('file must be a string.')
    tmp_file = file + '.tmp'
    if os.path.exists(tmp_file):
        logger.warning(
            'Temporary file %s already exists. Removing...', tmp_file
        )
        os.remove(tmp_file)
    return tmp_file


def nn_fill_array(array):
    """Fill any NaN values in an np.ndarray from the nearest non-nan values.

    Parameters
    ----------
    array : Union[np.ndarray, da.core.Array]
        Input array with NaN values

    Returns
    -------
    array : Union[np.ndarray, da.core.Array]
        Output array with NaN values filled
    """

    nan_mask = np.isnan(array)
    indices = nd.distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    if hasattr(array, 'vindex'):
        return array.vindex[tuple(indices)]
    return array[tuple(indices)]


def get_feature_basename(feature):
    """Get the base name of a feature, removing any height or pressure
    suffix"""
    height = re.findall(r'_\d+m', feature)
    press = re.findall(r'_\d+pa', feature)
    basename = (
        feature.replace(height[0], '')
        if height
        else feature.replace(press[0], '')
        if press
        else feature.split('_(.*)')[0]
        if '_(.*)' in feature
        else feature
    )
    return basename


def preprocess_datasets(dset):
    """Standardization preprocessing applied before datasets are concatenated
    by ``xr.open_mfdataset``"""
    if 'latitude' in dset.dims:
        dset = dset.swap_dims({'latitude': 'south_north'})
    if 'longitude' in dset.dims:
        dset = dset.swap_dims({'longitude': 'west_east'})
    if 'valid_time' in dset:
        dset = dset.rename({'valid_time': 'time'})
    if 'isobaricInhPa' in dset:
        dset = dset.rename({'isobaricInhPa': 'level'})
    if 'orog' in dset:
        dset = dset.rename({'orog': 'topography'})
    if 'time' in dset and dset.time.size > 1:
        if 'time' in dset.indexes and hasattr(
            dset.indexes['time'], 'to_datetimeindex'
        ):
            dset['time'] = dset.indexes['time'].to_datetimeindex()
        ti = dset['time'].astype('int64')
        dset['time'] = ti

    # sometimes these are included in prelim ERA5 data
    dset = dset.drop_vars(['expver', 'number'], errors='ignore')
    return dset


def xr_open_mfdataset(files, **kwargs):
    """Wrapper for xr.open_mfdataset with default opening options."""
    default_kwargs = {'engine': 'netcdf4'}
    default_kwargs.update(kwargs)
    if isinstance(files, str):
        files = [files]
    out = xr.open_mfdataset(
        files, preprocess=preprocess_datasets, **default_kwargs
    )
    bad_dims = (
        'latitude' in out
        and len(out['latitude'].dims) == 2
        and (out['latitude'].dims != out['longitude'].dims)
    )
    if bad_dims:
        out['longitude'] = (out['latitude'].dims, out['longitude'].values.T)
    return out


def safe_cast(o):
    """Cast to type safe for serialization."""
    if isinstance(o, tf.Tensor):
        o = o.numpy()
    if isinstance(o, (float, np.float64, np.float32)):
        return float(o)
    if isinstance(o, (int, np.int64, np.int32)):
        return int(o)
    if isinstance(o, (tuple, np.ndarray)):
        return list(o)
    if isinstance(o, (str, list)):
        return o
    return str(o)


def enforce_limits(features, data, nn_fill=False):
    """Enforce physical limits for feature data

    Parameters
    ----------
    features : list
        List of features with ordering corresponding to last channel of
        data array.
    data : ndarray
        Array of feature data
    nn_fill : bool
        Whether to fill values outside of limits with nearest neighbor
        interpolation. If False, values outside of limits are set to
        the limits.

    Returns
    -------
    data : ndarray
        Array of feature data with physical limits enforced
    """
    for fidx, fn in enumerate(features):
        dset_name = get_feature_basename(fn)
        if dset_name not in OUTPUT_ATTRS:
            msg = (
                f'Attempting to enforce physical limits for "{dset_name}" '
                'but it is not in the OUTPUT_ATTRS dict!'
            )
            logger.error(msg)
            raise KeyError(msg)

        max_val = OUTPUT_ATTRS[dset_name].get('max', np.inf)
        min_val = OUTPUT_ATTRS[dset_name].get('min', -np.inf)
        enforcing_msg = f'Enforcing range of ({min_val}, {max_val}) for "{fn}"'
        if nn_fill:
            enforcing_msg += ' with nearest neighbor interpolation.'
        else:
            enforcing_msg += ' with clipping.'

        f_max = data[..., fidx].max()
        f_min = data[..., fidx].min()
        max_frac = np.sum(data[..., fidx] > max_val) / data[..., fidx].size
        min_frac = np.sum(data[..., fidx] < min_val) / data[..., fidx].size
        msg = (
            f'{fn} has a max of {f_max} > {max_val}, with '
            f'{max_frac:.4e} of points above this max. {enforcing_msg}'
        )
        if f_max > max_val:
            logger.warning(msg)
            warn(msg)
        msg = (
            f'{fn} has a min of {f_min} < {min_val}, with '
            f'{min_frac:.4e} of points below this min. {enforcing_msg}'
        )
        if f_min < min_val:
            logger.warning(msg)
            warn(msg)

        if nn_fill:
            data[..., fidx] = np.where(
                data[..., fidx] > max_val, np.nan, data[..., fidx]
            )
            data[..., fidx] = np.where(
                data[..., fidx] < min_val, np.nan, data[..., fidx]
            )
            data[..., fidx] = nn_fill_array(data[..., fidx])
        else:
            data[..., fidx] = np.maximum(data[..., fidx], min_val)
            data[..., fidx] = np.minimum(data[..., fidx], max_val)
    return data.astype(np.float32)


def get_dset_attrs(feature):
    """Get attrributes for output feature

    Parameters
    ----------
    feature : str
        Name of feature to write

    Returns
    -------
    attrs : dict
        Dictionary of attributes for requested dset
    dtype : str
        Data type for requested dset. Defaults to float32
    """
    feat_base_name = get_feature_basename(feature)
    if feat_base_name in OUTPUT_ATTRS:
        attrs = OUTPUT_ATTRS[feat_base_name]
        dtype = attrs.get('dtype', 'float32')
    else:
        attrs = {}
        dtype = 'float32'
        msg = (
            'Could not find feature "{}" with base name "{}" in '
            'OUTPUT_ATTRS global variable. Writing with float32 and no '
            'chunking.'.format(feature, feat_base_name)
        )
        logger.warning(msg)
        warn(msg)

    return attrs, dtype


def safe_serialize(obj, **kwargs):
    """json.dumps with non-serializable object handling."""
    return json.dumps(obj, default=safe_cast, **kwargs)


class Timer:
    """Timer class for timing and storing function call times."""

    def __init__(self):
        self.log = {}
        self._elapsed = 0
        self._start = None
        self._stop = None

    def start(self):
        """Set start of timing period."""
        self._start = time.time()

    def stop(self):
        """Set stop time of timing period."""
        self._stop = time.time()

    @property
    def elapsed(self):
        """Elapsed time between start and stop."""
        if self._stop is None:
            return time.time() - self._start
        return self._stop - self._start

    @property
    def elapsed_str(self):
        """Elapsed time in string format."""
        return f'{round(self.elapsed, 5)} seconds'

    def __call__(self, func, call_id=None, log=False):
        """Time function call and store elapsed time in self.log.

        Parameters
        ----------
        func : function
            Function to time
        call_id: int | None
            ID to distingush calls with the same function name. For example,
            when runnning forward passes on multiple chunks.
        log : bool
            Whether to write to active logger

        Returns
        -------
        output of func
        """

        def wrapper(*args, **kwargs):
            """Wrapper with decorator pattern.

            Parameters
            ----------
            args : list
                positional arguments for fun
            kwargs : dict
                keyword arguments for fun
            """
            self.start()
            out = func(*args, **kwargs)
            self.stop()
            if call_id is not None:
                entry = self.log.get(call_id, {})
                entry[func.__name__] = self.elapsed
                self.log[call_id] = entry
            else:
                self.log[func.__name__] = self.elapsed
            if log:
                logger.debug(
                    'Call to %s finished in %s',
                    func.__name__,
                    self.elapsed_str,
                )
            return out

        return wrapper


def generate_random_string(length):
    """Generate random string with given length. Used for naming temporary
    files to avoid collisions."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def temporal_coarsening(data, t_enhance=4, method='subsample'):
    """Coarsen data according to t_enhance resolution

    Parameters
    ----------
    data : Union[np.ndarray, da.core.Array]
        5D array with dimensions
        (observations, spatial_1, spatial_2, temporal, features)
    t_enhance : int
        factor by which to coarsen temporal dimension
    method : str
        accepted options: [subsample, average, total, min, max]
        Subsample will take every t_enhance-th time step, average will average
        over t_enhance time steps, total will sum over t_enhance time steps

    Returns
    -------
    coarse_data : Union[np.ndarray, da.core.Array]
        5D array with same dimensions as data with new coarse resolution
    """

    if t_enhance is not None and len(data.shape) == 5:
        new_shape = (
            data.shape[0],
            data.shape[1],
            data.shape[2],
            -1,
            t_enhance,
            data.shape[4],
        )

        if method == 'subsample':
            coarse_data = data[:, :, :, ::t_enhance, :]

        elif method == 'average':
            coarse_data = np.nansum(np.reshape(data, new_shape), axis=4)
            coarse_data /= t_enhance

        elif method == 'max':
            coarse_data = np.max(np.reshape(data, new_shape), axis=4)

        elif method == 'min':
            coarse_data = np.min(np.reshape(data, new_shape), axis=4)

        elif method == 'total':
            coarse_data = np.nansum(np.reshape(data, new_shape), axis=4)

        else:
            msg = (
                f'Did not recognize temporal_coarsening method "{method}", '
                'can only accept one of: [subsample, average, total, max, min]'
            )
            logger.error(msg)
            raise KeyError(msg)

    else:
        coarse_data = data

    return coarse_data


def spatial_coarsening(data, s_enhance=2, obs_axis=True):
    """Coarsen data according to s_enhance resolution

    Parameters
    ----------
    data : Union[np.ndarray, da.core.Array]
        5D | 4D | 3D | 2D array with dimensions:
        (n_obs, spatial_1, spatial_2, temporal, features) (obs_axis=True)
        (n_obs, spatial_1, spatial_2, features) (obs_axis=True)
        (spatial_1, spatial_2, temporal, features) (obs_axis=False)
        (spatial_1, spatial_2, temporal_or_features) (obs_axis=False)
        (spatial_1, spatial_2) (obs_axis=False)
    s_enhance : int
        factor by which to coarsen spatial dimensions
    obs_axis : bool
        Flag for if axis=0 is the observation axis. If True (default)
        spatial axis=(1, 2) (zero-indexed), if False spatial axis=(0, 1)

    Returns
    -------
    data : Union[np.ndarray, da.core.Array]
        2D, 3D | 4D | 5D array with same dimensions as data with new coarse
        resolution
    """

    if len(data.shape) < 2:
        msg = (
            'Data must be 2D, 3D, 4D, or 5D to do spatial coarsening, but '
            f'received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    if obs_axis and len(data.shape) < 3:
        msg = (
            'Data must be 3D, 4D, or 5D to do spatial coarsening with '
            f'obs_axis=True, but received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    if s_enhance is not None and s_enhance > 1:
        bad1 = obs_axis and (
            data.shape[1] % s_enhance != 0 or data.shape[2] % s_enhance != 0
        )
        bad2 = not obs_axis and (
            data.shape[0] % s_enhance != 0 or data.shape[1] % s_enhance != 0
        )
        if bad1 or bad2:
            msg = (
                's_enhance must evenly divide grid size. '
                f'Received s_enhance: {s_enhance} with data shape: '
                f'{data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

        if obs_axis and len(data.shape) == 3:
            data = np.reshape(
                data,
                (
                    data.shape[0],
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2] // s_enhance,
                    s_enhance,
                ),
            )
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif obs_axis:
            data = np.reshape(
                data,
                (
                    data.shape[0],
                    data.shape[1] // s_enhance,
                    s_enhance,
                    data.shape[2] // s_enhance,
                    s_enhance,
                    *data.shape[3:],
                ),
            )
            data = data.sum(axis=(2, 4)) / s_enhance**2

        elif not obs_axis and len(data.shape) == 2:
            data = np.reshape(
                data,
                (
                    data.shape[0] // s_enhance,
                    s_enhance,
                    data.shape[1] // s_enhance,
                    s_enhance,
                ),
            )
            data = data.sum(axis=(1, 3)) / s_enhance**2

        elif not obs_axis:
            data = np.reshape(
                data,
                (
                    data.shape[0] // s_enhance,
                    s_enhance,
                    data.shape[1] // s_enhance,
                    s_enhance,
                    *data.shape[2:],
                ),
            )
            data = data.sum(axis=(1, 3)) / s_enhance**2

        else:
            msg = (
                'Data must be 2D, 3D, 4D, or 5D to do spatial coarsening, but '
                f'received: {data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

    return data


def pd_date_range(*args, **kwargs):
    """A simple wrapper on the pd.date_range() method that handles the closed
    vs. inclusive kwarg change in pd 1.4.0"""
    incl = version.parse(pd.__version__) >= version.parse('1.4.0')

    if incl and 'closed' in kwargs:
        kwargs['inclusive'] = kwargs.pop('closed')
    elif not incl and 'inclusive' in kwargs:
        kwargs['closed'] = kwargs.pop('inclusive')
        if kwargs['closed'] == 'both':
            kwargs['closed'] = None

    return pd.date_range(*args, **kwargs)


def camel_to_underscore(name):
    """Converts a camel case string to underscore (snake case)."""
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return name
