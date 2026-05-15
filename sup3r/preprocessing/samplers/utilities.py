"""Miscellaneous utilities for sampling"""

import logging
from warnings import warn

import dask.array as da
import numpy as np

from sup3r.preprocessing.utilities import _compute_chunks_if_dask
from sup3r.utilities.utilities import RANDOM_GENERATOR

logger = logging.getLogger(__name__)


def uniform_box_sampler(data_shape, sample_shape):
    """Returns a 2D spatial slice used to extract a sample from a data array.

    Parameters
    ----------
    data_shape : tuple
        (rows, cols) Size of full grid available for sampling
    sample_shape : tuple
        (rows, cols) Size of grid to sample from data

    Returns
    -------
    slices : list
        List of slices corresponding to row and col extent of arr sample
    """

    shape_1 = (
        data_shape[0] if data_shape[0] < sample_shape[0] else sample_shape[0]
    )
    shape_2 = (
        data_shape[1] if data_shape[1] < sample_shape[1] else sample_shape[1]
    )
    shape = (shape_1, shape_2)
    start_row = RANDOM_GENERATOR.integers(
        0, data_shape[0] - sample_shape[0] + 1
    )
    start_col = RANDOM_GENERATOR.integers(
        0, data_shape[1] - sample_shape[1] + 1
    )
    stop_row = start_row + shape[0]
    stop_col = start_col + shape[1]

    return [slice(start_row, stop_row), slice(start_col, stop_col)]


def weighted_box_sampler(data_shape, sample_shape, weights):
    """Extracts a temporal slice from data with selection weighted based on
    provided weights

    Parameters
    ----------
    data_shape : tuple
        (rows, cols) Size of full spatial grid available for sampling
    sample_shape : tuple
        (rows, cols) Size of grid to sample from data
    weights : ndarray
        Array of weights used to specify selection strategy. e.g. If weights is
        [0.2, 0.4, 0.1, 0.3] then the upper left quadrant of the spatial
        domain will be sampled 20 percent of the time, the upper right quadrant
        will be sampled 40 percent of the time, etc.

    Returns
    -------
    slices : list
        List of spatial slices [spatial_1, spatial_2]
    """
    max_cols = (
        data_shape[1] if data_shape[1] < sample_shape[1] else sample_shape[1]
    )
    max_rows = (
        data_shape[0] if data_shape[0] < sample_shape[0] else sample_shape[0]
    )
    max_cols = data_shape[1] - max_cols + 1
    max_rows = data_shape[0] - max_rows + 1
    indices = np.arange(0, max_rows * max_cols)
    chunks = np.array_split(indices, len(weights))
    weight_list = []
    for i, w in enumerate(weights):
        weight_list += [w] * len(chunks[i])
    weight_list /= np.sum(weight_list)
    msg = (
        'Must have a sample_shape with a number of elements greater than '
        'or equal to the number of spatial weights.'
    )
    assert len(indices) >= len(weight_list), msg
    start = RANDOM_GENERATOR.choice(indices, p=weight_list)
    row = start // max_cols
    col = start % max_cols
    stop_1 = row + np.min([sample_shape[0], data_shape[0]])
    stop_2 = col + np.min([sample_shape[1], data_shape[1]])

    slice_1 = slice(row, stop_1)
    slice_2 = slice(col, stop_2)

    return [slice_1, slice_2]


def weighted_time_sampler(data_shape, sample_shape, weights):
    """Returns a temporal slice with selection weighted based on
    provided weights used to extract temporal chunk from data

    Parameters
    ----------
    data_shape : tuple
        (rows, cols, n_steps) Size of full spatiotemporal data grid available
        for sampling
    shape : tuple
        (time_steps) Size of time slice to sample from data
    weights : list
        List of weights used to specify selection strategy. e.g. If weights
        is [0.2, 0.8] then the start of the temporal slice will be selected
        from the first half of the temporal extent with 0.8 probability and
        0.2 probability for the second half.

    Returns
    -------
    slice : slice
        time slice with size shape
    """

    shape = data_shape[2] if data_shape[2] < sample_shape else sample_shape
    t_indices = np.arange(0, data_shape[2] - shape + 1)
    t_chunks = np.array_split(t_indices, len(weights))

    weight_list = []
    for i, w in enumerate(weights):
        weight_list += [w] * len(t_chunks[i])
    weight_list /= np.sum(weight_list)

    start = RANDOM_GENERATOR.choice(t_indices, p=weight_list)
    stop = start + shape

    return slice(start, stop)


def uniform_time_sampler(data_shape, sample_shape, crop_slice=slice(None)):
    """Returns temporal slice used to extract temporal chunk from data.

    Parameters
    ----------
    data_shape : tuple
        (rows, cols, n_steps) Size of full spatiotemporal data grid available
        for sampling
    sample_shape : int
        (time_steps) Size of time slice to sample from data grid
    crop_slice : slice
        Optional slice used to restrict the sampling window.

    Returns
    -------
    slice : slice
        time slice with size shape
    """
    shape = data_shape[2] if data_shape[2] < sample_shape else sample_shape
    indices = np.arange(data_shape[2] + 1)[crop_slice]
    start = RANDOM_GENERATOR.integers(indices[0], indices[-1] - shape + 1)
    stop = start + shape
    return slice(start, stop)


def daily_time_sampler(data, shape, time_index):
    """Finds a random temporal slice from data starting at midnight

    Parameters
    ----------
    data : Union[np.ndarray, da.core.Array]
        Data array with dimensions
        (spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample from data, must be an integer
        less than or equal to 24.
    time_index : pd.DatetimeIndex
        Time index that matches the data axis=2

    Returns
    -------
    slice : slice
        time slice with size shape of data starting at the beginning of the day
    """

    msg = (
        f'data {data.shape} and time index ({len(time_index)}) '
        'shapes do not match, cannot sample daily data.'
    )
    assert data.shape[2] == len(time_index), msg

    ti_short = time_index[: -(shape - 1)]
    midnight_ilocs = np.where(
        (ti_short.hour == 0) & (ti_short.minute == 0) & (ti_short.second == 0)
    )[0]

    if not any(midnight_ilocs):
        msg = (
            'Cannot sample time index of shape {} with requested daily '
            'sample shape {}'.format(len(time_index), shape)
        )
        logger.error(msg)
        raise RuntimeError(msg)

    start = RANDOM_GENERATOR.integers(0, len(midnight_ilocs))
    start = midnight_ilocs[start]
    stop = start + shape

    return slice(start, stop)


def nsrdb_sub_daily_sampler(data, shape, time_index=None):
    """Finds a random sample during daylight hours of a day. Nightime is
    assumed to be marked as NaN in feature axis == csr_ind in the data input.

    Parameters
    ----------
    data : Union[Sup3rX, Sup3rDataset]
        Dataset object with 'clearsky_ratio' accessible as
        data['clearsky_ratio'] (spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample from data, must be an integer
        less than or equal to 24.
    time_index : pd.DatetimeIndex
        Time index corresponding the the time axis of `data`. If None then
        data.time_index will be used.

    Returns
    -------
    tslice : slice
        time slice with size shape of data starting at the beginning of the day
    """
    time_index = time_index if time_index is not None else data.time_index
    tslice = daily_time_sampler(data, 24, time_index)
    night_mask = da.isnan(data['clearsky_ratio'][..., tslice]).any(axis=(0, 1))

    if shape >= data.shape[2]:
        return tslice

    if (night_mask).all():
        msg = (
            f'No daylight data found for tslice {tslice} '
            f'{time_index[tslice]}'
        )
        logger.warning(msg)
        warn(msg)
        return tslice

    day_ilocs = np.where(np.asarray(~night_mask))[0]
    padding = shape - len(day_ilocs)
    half_pad = int(np.round(padding / 2))
    new_start = tslice.start + day_ilocs[0] - half_pad
    new_end = new_start + shape
    return slice(new_start, new_end)


def nsrdb_reduce_daily_data(data, shape, csr_ind=0):
    """Takes a 5D array and reduces the axis=3 temporal dim to daylight hours.

    Parameters
    ----------
    data : Union[np.ndarray, da.core.Array]
        5D data array, where [..., csr_ind] is assumed to be clearsky ratio
        with NaN at night.
        (n_obs, spatial_1, spatial_2, temporal, features)
    shape : int
        (time_steps) Size of time slice to sample from data. If this is
        greater than data.shape[-2] data won't be reduced.
    csr_ind : int
        Index of the feature axis where clearsky ratio is located and NaN's can
        be found at night.

    Returns
    -------
    data : Union[np.ndarray, da.core.Array]
        Same as input but with axis=3 reduced to dailylight hours with
        requested shape.
    """

    night_mask = da.isnan(data[:, :, :, :, csr_ind]).any(axis=(0, 1, 2))

    if shape >= data.shape[3]:
        return data

    if night_mask.all():
        msg = f'No daylight data found for data of shape {data.shape}'
        logger.warning(msg)
        warn(msg)
        return data

    day_ilocs = _compute_chunks_if_dask(np.where(~night_mask)[0])
    padding = shape - len(day_ilocs)
    half_pad = int(np.ceil(padding / 2))
    start = day_ilocs[0] - half_pad
    tslice = slice(start, start + shape)
    return data[..., tslice, :]
