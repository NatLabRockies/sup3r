"""Miscellaneous utilities shared across the batch_queues module"""

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, zoom

logger = logging.getLogger(__name__)


def temporal_simple_enhancing(data, t_enhance=4, mode='constant'):
    """Upsample data according to t_enhance resolution

    Parameters
    ----------
    data : Union[np.ndarray, da.core.Array]
        5D array with dimensions
        (observations, spatial_1, spatial_2, temporal, features)
    t_enhance : int
        factor by which to enhance temporal dimension
    mode : str
        interpolation method for enhancement.

    Returns
    -------
    enhanced_data : Union[np.ndarray, da.core.Array]
        5D array with same dimensions as data with new enhanced resolution
    """

    if t_enhance in {None, 1}:
        enhanced_data = data
    elif t_enhance not in {None, 1} and len(data.shape) == 5:
        if mode == 'constant':
            enhancement = [1, 1, 1, t_enhance, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )
        elif mode == 'linear':
            index_t_hr = np.arange(data.shape[3] * t_enhance)
            index_t_lr = index_t_hr[::t_enhance]
            enhanced_data = interp1d(
                index_t_lr, data, axis=3, fill_value='extrapolate'
            )(index_t_hr)
            enhanced_data = np.array(enhanced_data, dtype=np.float32)
    elif len(data.shape) != 5:
        msg = (
            'Data must be 5D to do temporal enhancing, but '
            f'received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    return enhanced_data


def smooth_data(low_res, training_features, smoothing_ignore, smoothing=None):
    """Smooth data using a gaussian filter

    Parameters
    ----------
    low_res : Union[np.ndarray, da.core.Array]
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    training_features : list | None
        Ordered list of training features input to the generative model
    smoothing_ignore : list | None
        List of features to ignore for the smoothing filter. None will
        smooth all features if smoothing kwarg is not None
    smoothing : float | None
        Standard deviation to use for gaussian filtering of the coarse
        data. This can be tuned by matching the kinetic energy of a low
        resolution simulation with the kinetic energy of a coarsened and
        smoothed high resolution simulation. If None no smoothing is
        performed.

    Returns
    -------
    low_res : Union[np.ndarray, da.core.Array]
        4D | 5D array
        (batch_size, spatial_1, spatial_2, features)
        (batch_size, spatial_1, spatial_2, temporal, features)
    """

    if smoothing is not None:
        feat_iter = [
            j
            for j in range(low_res.shape[-1])
            if training_features[j] not in smoothing_ignore
        ]
        for i in range(low_res.shape[0]):
            for j in feat_iter:
                if len(low_res.shape) == 5:
                    for t in range(low_res.shape[-2]):
                        low_res[i, ..., t, j] = gaussian_filter(
                            low_res[i, ..., t, j], smoothing, mode='nearest'
                        )
                else:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], smoothing, mode='nearest'
                    )
    return low_res


def spatial_simple_enhancing(data, s_enhance=2, obs_axis=True):
    """Simple enhancing according to s_enhance resolution

    Parameters
    ----------
    data : Union[np.ndarray, da.core.Array]
        5D | 4D | 3D array with dimensions:
        (n_obs, spatial_1, spatial_2, temporal, features) (obs_axis=True)
        (n_obs, spatial_1, spatial_2, features) (obs_axis=True)
        (spatial_1, spatial_2, temporal, features) (obs_axis=False)
        (spatial_1, spatial_2, temporal_or_features) (obs_axis=False)
    s_enhance : int
        factor by which to enhance spatial dimensions
    obs_axis : bool
        Flag for if axis=0 is the observation axis. If True (default)
        spatial axis=(1, 2) (zero-indexed), if False spatial axis=(0, 1)

    Returns
    -------
    enhanced_data : Union[np.ndarray, da.core.Array]
        3D | 4D | 5D array with same dimensions as data with new enhanced
        resolution
    """

    if len(data.shape) < 3:
        msg = (
            'Data must be 3D, 4D, or 5D to do spatial enhancing, but '
            f'received: {data.shape}'
        )
        logger.error(msg)
        raise ValueError(msg)

    if s_enhance is not None and s_enhance > 1:
        if obs_axis and len(data.shape) == 5:
            enhancement = [1, s_enhance, s_enhance, 1, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )

        elif obs_axis and len(data.shape) == 4:
            enhancement = [1, s_enhance, s_enhance, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )

        elif not obs_axis and len(data.shape) == 4:
            enhancement = [s_enhance, s_enhance, 1, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )

        elif not obs_axis and len(data.shape) == 3:
            enhancement = [s_enhance, s_enhance, 1]
            enhanced_data = zoom(
                data, enhancement, order=0, mode='nearest', grid_mode=True
            )
        else:
            msg = (
                'Data must be 3D, 4D, or 5D to do spatial enhancing, but '
                f'received: {data.shape}'
            )
            logger.error(msg)
            raise ValueError(msg)

    else:
        enhanced_data = data

    return enhanced_data
