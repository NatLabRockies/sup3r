"""Bias correction transformation functions.

TODO: These methods need to be refactored to use lazy calculations. They
currently slow down the forward pass runs when operating on full input data
volume.

We should write bc factor files in a format compatible with Loaders /
Rasterizers so we can use those class methods to match factors with locations
"""

import logging
from typing import Union
from warnings import warn

import dask.array as da
import numpy as np
import xarray as xr
from rex.utilities.bc_utils import QuantileDeltaMapping
from scipy.ndimage import gaussian_filter

from sup3r.preprocessing import Rasterizer
from sup3r.preprocessing.utilities import make_time_index_from_kws

logger = logging.getLogger(__name__)


def _get_factors(target, shape, var_names, bias_fp, threshold=0.1):
    """Get bias correction factors from sup3r's standard resource

    This was stripped without any change from original
    `_get_spatial_bc_factors` to allow re-use in other `*_bc_factors`
    functions.

    Parameters
    ----------
    target : tuple
        (lat, lon) lower left corner of raster. Either need target+shape or
        raster_file.
    shape : tuple
        (rows, cols) grid size. Either need target+shape or raster_file.
    var_names : dict
        A dictionary mapping the expected variable name in the `Resource`
        and the desired name to output. For instance the dictionary
        `{'base': 'base_ghi_params'}` would return a `params['base']` with
        a value equals to `base_ghi_params` from the `bias_fp` file.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets defined as values of the dictionary `var_names` given as an
        input argument, such as "{feature_name}_scalar" or
        "base_{base_dset}_params". Those are the full low-resolution shape of
        the forward pass input that will be sliced using lr_padded_slice for
        the current chunk.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error is
        raised.

    Return
    ------
    dict :
        A dictionary with the content from `bias_fp` as mapped by `var_names`,
        therefore, the keys here are the same keys in `var_names`.
        Also includes 'global_attrs' from Rasterizer.
    """
    res = Rasterizer(
        file_paths=bias_fp,
        target=np.asarray(target),
        shape=shape,
        threshold=threshold,
    )
    missing = [d for d in var_names.values() if d.lower() not in res.features]
    msg = f'Missing {" and ".join(missing)} in resource: {bias_fp}.'
    assert missing == [], msg
    # pylint: disable=E1136
    out = {k: res[var_names[k].lower()].data for k in var_names}
    out['cfg'] = res.attrs
    return out


def _get_spatial_bc_factors(lat_lon, feature_name, bias_fp, threshold=0.1):
    """Get bc factors (scalar/adder) for the given feature for the given
    domain (specified by lat_lon).

    Parameters
    ----------
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "{feature_name}_scalar" and "{feature_name}_adder" will be retrieved
        from bias_fp.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "{feature_name}_scalar" and "{feature_name}_adder" that are
        the full low-resolution shape of the forward pass input that will be
        sliced using lr_padded_slice for the current chunk.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error is
        raised.
    """
    var_names = {
        'scalar': f'{feature_name}_scalar',
        'adder': f'{feature_name}_adder',
    }
    target = lat_lon[-1, 0, :]
    shape = lat_lon.shape[:-1]
    return _get_factors(
        target=target,
        shape=shape,
        var_names=var_names,
        bias_fp=bias_fp,
        threshold=threshold,
    )


def _get_spatial_bc_quantiles(
    lat_lon: Union[np.ndarray, da.core.Array],
    base_dset: str,
    feature_name: str,
    bias_fp: str,
    threshold: float = 0.1,
):
    """Statistical distributions previously estimated for given lat/lon points

    Recover the parameters that describe the statistical distribution
    previously estimated with
    :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for three
    datasets: ``base`` (historical reference), ``bias`` (historical biased
    reference), and ``bias_fut`` (the future biased dataset, usually the data
    to correct).

    Parameters
    ----------
    lat_lon : Union[np.ndarray, da.core.Array]
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset : str
        Name of feature used as historical reference. A Dataset with name
        "base_{base_dset}_params" will be retrieved from ``bias_fp``.
    feature_name : str
        Name of the feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved from ``bias_fp``.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "base_{base_dset}_params", "bias_{feature_name}_params", and
        "bias_fut_{feature_name}_params" that define the statistical
        distributions.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error
        is raised.

    Returns
    -------
    params : dict
        A dictionary collecting the following parameters:
        - base : np.array
          Parameters used to define the statistical distribution estimated for
          the ``base_dset``. It has a shape of (I, J, P), where (I, J) are the
          same first two dimensions of the given `lat_lon` and P is the number
          of parameters and depends on the type of distribution. See
          :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for more
          details.
        - bias : np.array
          Parameters used to define the statistical distribution estimated for
          (historical) ``feature_name``. It has a shape of (I, J, P), where
          (I, J) are the same first two dimensions of the given `lat_lon` and P
          is the number of parameters and depends on the type of distribution.
          See :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for
          more details.
        - bias_fut : np.array
          Parameters used to define the statistical distribution estimated for
          (future) ``feature_name``. It has a shape of (I, J, P), where (I, J)
          are the same first two dimensions of the given `lat_lon` and P is the
          number of parameters used and depends on the type of distribution.
          See :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for more
          details.
        - global_attrs : dict
          Metadata used to guide how to use of the previous parameters on
          reconstructing the statistical distributions. For instance,
          `cfg['dist']` defines the type of distribution. See
          :class:`~sup3r.bias.qdm.QuantileDeltaMappingCorrection` for more
          details, including which metadata is saved.

    Warnings
    --------
    Be careful selecting which `bias_fp` to use. In particular, if
    "bias_fut_{feature_name}_params" is representative for the desired target
    period.

    See Also
    --------
    sup3r.bias.qdm.QuantileDeltaMappingCorrection
        Estimate the statistical distributions loaded here.

    Examples
    --------
    >>> lat_lon = np.array([
    ...              [39.649033, -105.46875 ],
    ...              [39.649033, -104.765625]])
    >>> params = _get_spatial_bc_quantiles(
    ...             lat_lon, "ghi", "rsds", "./dist_params.hdf")

    """
    var_names = {
        'base': f'base_{base_dset}_params',
        'bias': f'bias_{feature_name}_params',
        'bias_fut': f'bias_fut_{feature_name}_params',
    }
    target = lat_lon[-1, 0, :]
    shape = lat_lon.shape[:-1]
    return _get_factors(
        target=target,
        shape=shape,
        var_names=var_names,
        bias_fp=bias_fp,
        threshold=threshold,
    )


def global_linear_bc(data, scalar, adder, out_range=None):
    """Bias correct data using a simple global *scalar +adder method.

    Parameters
    ----------
    data : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    scalar : float
        Scalar (multiplicative) value to apply to input data.
    adder : float
        Adder value to apply to input data.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.

    Returns
    -------
    out : np.ndarray
        out = data * scalar + adder
    """
    out = data * scalar + adder
    if out_range is not None:
        out = np.maximum(out, np.min(out_range))
        out = np.minimum(out, np.max(out_range))
    return out


def local_linear_bc(
    data,
    lat_lon,
    feature_name,
    bias_fp,
    lr_padded_slice=None,
    out_range=None,
    smoothing=0,
    threshold=0.1,
):
    """Bias correct data using a simple annual (or multi-year) *scalar +adder
    method on a site-by-site basis.

    Parameters
    ----------
    data : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "{feature_name}_scalar" and "{feature_name}_adder" will be retrieved
        from bias_fp.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "{feature_name}_scalar" and "{feature_name}_adder" that are
        the full low-resolution shape of the forward pass input that will be
        sliced using lr_padded_slice for the current chunk.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included in the kwargs for the active chunk.
        If this is None, no slicing will be done and the full bias correction
        source shape will be used.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.
    smoothing : float
        Value to use to smooth the scalar/adder data. This can reduce the
        effect of extreme values within aggregations over large number of
        pixels.  This value is the standard deviation for the gaussian_filter
        kernel.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error is
        raised.

    Returns
    -------
    out : np.ndarray
        out = data * scalar + adder
    """

    out = _get_spatial_bc_factors(
        lat_lon, feature_name, bias_fp, threshold=threshold
    )
    scalar, adder = out['scalar'], out['adder']
    # 3D bias correction factors have seasonal/monthly correction in last axis
    if len(scalar.shape) == 3 and len(adder.shape) == 3:
        scalar = scalar.mean(axis=-1)
        adder = adder.mean(axis=-1)

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        scalar = scalar[spatial_slice]
        adder = adder[spatial_slice]

    if np.isnan(scalar).any() or np.isnan(adder).any():
        msg = (
            'Bias correction scalar/adder values had NaNs for '
            f'"{feature_name}" from: {bias_fp}'
        )
        logger.warning(msg)
        warn(msg)

    scalar = np.expand_dims(scalar, axis=-1)
    adder = np.expand_dims(adder, axis=-1)

    scalar = np.repeat(scalar, data.shape[-1], axis=-1)
    adder = np.repeat(adder, data.shape[-1], axis=-1)

    if smoothing > 0:
        for idt in range(scalar.shape[-1]):
            scalar[..., idt] = gaussian_filter(
                scalar[..., idt], smoothing, mode='nearest'
            )
            adder[..., idt] = gaussian_filter(
                adder[..., idt], smoothing, mode='nearest'
            )

    out = data * scalar + adder
    if out_range is not None:
        out = np.maximum(out, np.min(out_range))
        out = np.minimum(out, np.max(out_range))

    return out


def monthly_local_linear_bc(
    data,
    lat_lon,
    feature_name,
    bias_fp,
    date_range_kwargs,
    lr_padded_slice=None,
    temporal_avg=True,
    out_range=None,
    smoothing=0,
    scalar_range=None,
    adder_range=None,
    threshold=0.1,
):
    """Bias correct data using a simple monthly *scalar +adder method on a
    site-by-site basis.

    Parameters
    ----------
    data : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "{feature_name}_scalar" and "{feature_name}_adder" will be retrieved
        from bias_fp.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "{feature_name}_scalar" and "{feature_name}_adder" that are
        the full low-resolution shape of the forward pass input that will be
        sliced using lr_padded_slice for the current chunk.
    date_range_kwargs : dict
        Keyword args for pd.date_range to produce a DatetimeIndex object
        associated with the input data temporal axis (assumed 3rd axis e.g.
        axis=2). Note that if this method is called as part of a sup3r
        resolution forward pass, the date_range_kwargs will be included
        automatically for the current chunk.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    temporal_avg : bool
        Take the average scalars and adders for the chunk's time index, this
        will smooth the transition of scalars/adders from month to month if
        processing small chunks. If processing the full annual time index, set
        this to False.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.
    smoothing : float
        Value to use to smooth the scalar/adder data. This can reduce the
        effect of extreme values within aggregations over large number of
        pixels.  This value is the standard deviation for the gaussian_filter
        kernel.
    scalar_range : tuple | None
        Allowed range for the scalar term in the linear bias correction.
    adder_range : tuple | None
        Allowed range for the adder term in the linear bias correction.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error is
        raised.

    Returns
    -------
    out : np.ndarray
        out = data * scalar + adder
    """
    time_index = make_time_index_from_kws(date_range_kwargs)
    out = _get_spatial_bc_factors(
        lat_lon, feature_name, bias_fp, threshold=threshold
    )
    scalar, adder = out['scalar'], out['adder']

    assert len(scalar.shape) == 3, 'Monthly bias correct needs 3D scalars'
    assert len(adder.shape) == 3, 'Monthly bias correct needs 3D adders'

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        scalar = scalar[spatial_slice]
        adder = adder[spatial_slice]

    imonths = time_index.month.values - 1
    scalar = scalar[..., imonths]
    adder = adder[..., imonths]

    if temporal_avg:
        scalar = scalar.mean(axis=-1)
        adder = adder.mean(axis=-1)
        scalar = np.expand_dims(scalar, axis=-1)
        adder = np.expand_dims(adder, axis=-1)
        scalar = np.repeat(scalar, data.shape[-1], axis=-1)
        adder = np.repeat(adder, data.shape[-1], axis=-1)
        if len(time_index.month.unique()) > 2:
            msg = (
                'Bias correction method "monthly_local_linear_bc" was used '
                'with temporal averaging over a time index with >2 months.'
            )
            warn(msg)
            logger.warning(msg)

    if np.isnan(scalar).any() or np.isnan(adder).any():
        msg = (
            'Bias correction scalar/adder values had NaNs for '
            f'"{feature_name}" from: {bias_fp}'
        )
        logger.warning(msg)
        warn(msg)

    if smoothing > 0:
        for idt in range(scalar.shape[-1]):
            scalar[..., idt] = gaussian_filter(
                scalar[..., idt], smoothing, mode='nearest'
            )
            adder[..., idt] = gaussian_filter(
                adder[..., idt], smoothing, mode='nearest'
            )

    if scalar_range is not None:
        scalar = np.minimum(scalar, np.max(scalar_range))
        scalar = np.maximum(scalar, np.min(scalar_range))

    if adder_range is not None:
        adder = np.minimum(adder, np.max(adder_range))
        adder = np.maximum(adder, np.min(adder_range))

    out = data * scalar + adder
    if out_range is not None:
        out = np.maximum(out, np.min(out_range))
        out = np.minimum(out, np.max(out_range))

    return out


def _apply_qdm(
    subset,
    base_params,
    bias_params,
    bias_fut_params,
    dist='empirical',
    sampling='linear',
    log_base=10,
    relative=True,
    no_trend=False,
    delta_denom_min=None,
    delta_denom_zero=None,
    delta_range=None,
    max_workers=1,
):
    """Run QuantileDeltaMapping routine for the given time index. Used in local
    qdm correction and presrat.

    Parameters
    ----------
    subset : np.ndarray | da.core.array
        Subset of Sup3r input data to be bias corrected, assumed to be 3D with
        shape (spatial, spatial, temporal) for a single feature.
    base_params : np.ndarray
        4D array of **observed historical** distribution parameters created
        from a multi-year set of data where the shape is
        (space, space, time, N). This can be the
        output of a parametric distribution fit like
        ``scipy.stats.weibull_min.fit()`` where N is the number of parameters
        for that distribution, or this can define the x-values of N points from
        an empirical CDF that will be linearly interpolated between. If this is
        an empirical CDF, this must include the 0th and 100th percentile values
        and have even percentile spacing between values.
    bias_params : np.ndarray
        Same requirements as params_oh. This input arg is for the **modeled
        historical distribution**.
    bias_fut_params : np.ndarray | None
        Same requirements as params_oh. This input arg is for the **modeled
        future distribution**. If this is None, this defaults to params_mh
        (no future data, just corrected to modeled historical distribution)
    dist : str
        Probability distribution name to use to model the data which
        determines how the param args are used. This can "empirical" or any
        continuous distribution name from ``scipy.stats``.
    sampling : str | np.ndarray
        If dist="empirical", this is an option for how the quantiles were
        sampled to produce the params inputs, e.g., how to sample the
        y-axis of the distribution (see sampling functions in
        ``rex.utilities.bc_utils``). "linear" will do even spacing, "log"
        will concentrate samples near quantile=0, and "invlog" will
        concentrate samples near quantile=1. Can also be a 1D array of dist
        inputs if being used from reV, but they must all be the same
        option.
    log_base : int | float | np.ndarray
        Log base value if sampling is "log" or "invlog". A higher value
        will concentrate more samples at the extreme sides of the
        distribution. Can also be a 1D array of dist inputs if being used
        from reV, but they must all be the same option.
    relative : bool | np.ndarray
        Flag to preserve relative rather than absolute changes in
        quantiles. relative=False (default) will multiply by the change in
        quantiles while relative=True will add. See Equations 4-6 from
        Cannon et al., 2015 for more details. Can also be a 1D array of
        dist inputs if being used from reV, but they must all be the same
        option.
    no_trend : bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias by
        comparing the distributions of the biased dataset with a reference
        datasets, without reinforcing the zero rate or applying the k-factor.
        See ``params_mf`` of
        :class:`rex.utilities.bc_utils.QuantileDeltaMapping`. Note that this
        assumes that params_mh is the data distribution representative for the
        target data.
    delta_denom_min : float | None
        Option to specify a minimum value for the denominator term in the
        calculation of a relative delta value. This prevents division by a
        very small number making delta blow up and resulting in very large
        output bias corrected values. See equation 4 of Cannon et al., 2015
        for the delta term. If this is not set, the ``zero_rate_threshold``
        calculated as part of the presrat bias calculation will be used
    delta_range : tuple | None
        Option to set a (min, max) on the delta term in QDM. This can help
        prevent QDM from making non-realistic increases/decreases in
        otherwise physical values. See equation 4 of Cannon et al., 2015 for
        the delta term.
    delta_denom_zero : float | None
        Option to specify a value to replace zeros in the denominator term
        in the calculation of a relative delta value. This prevents
        division by a very small number making delta blow up and resulting
        in very large output bias corrected values. See equation 4 of
        Cannon et al., 2015 for the delta term.
    max_workers : int | None
        Max number of workers to use for QDM process pool
    """

    # This satisfies the rex's QDM design
    bias_fut_params = (
        None
        if no_trend
        else np.reshape(bias_fut_params, (-1, bias_fut_params.shape[-1]))
    )

    # The distributions at this point, after selected the respective
    # time window with `window_idx`, are 3D (space, space, N-params)
    # Collapse 3D (space, space, N) into 2D (space**2, N)
    QDM = QuantileDeltaMapping(
        params_oh=np.reshape(base_params, (-1, base_params.shape[-1])),
        params_mh=np.reshape(bias_params, (-1, bias_params.shape[-1])),
        params_mf=bias_fut_params,
        dist=dist,
        relative=relative,
        sampling=sampling,
        log_base=log_base,
        delta_denom_min=delta_denom_min,
        delta_denom_zero=delta_denom_zero,
        delta_range=delta_range,
    )
    # input 3D shape (spatial, spatial, temporal)
    # QDM expects input arr with shape (time, space)
    tmp = np.reshape(subset.data, (-1, subset.shape[-1])).T

    # Apply QDM correction
    logger.info(f'Applying QDM to data with shape {tmp.shape}...')
    tmp = QDM(tmp, max_workers=max_workers)
    logger.info(f'Finished QDM on data shape {tmp.shape}!')

    # Reorgnize array back from  (time, space)
    # to (spatial, spatial, temporal)
    return np.reshape(tmp.T, subset.shape)


def local_qdm_bc(
    data: xr.DataArray,
    lat_lon: np.ndarray,
    base_dset: str,
    feature_name: str,
    bias_fp: str,
    date_range_kwargs: dict,
    lr_padded_slice=None,
    threshold=0.1,
    relative=True,
    no_trend=False,
    delta_denom_min=None,
    delta_denom_zero=None,
    delta_range=None,
    out_range=None,
    max_workers=1,
):
    """Bias correction using QDM

    Apply QDM to correct bias on the given data. It assumes that the required
    statistical distributions were previously estimated and saved in
    ``bias_fp``.

    Assume CDF for the nearest day of year (doy) is representative.

    Parameters
    ----------
    data : xr.DataArray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : np.ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset :
        Name of feature that is used as (historical) reference. Dataset with
        names "base_{base_dset}_params" will be retrieved.
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved.
    date_range_kwargs : dict
        Keyword args for pd.date_range to produce a DatetimeIndex object
        associated with the input data temporal axis (assumed 3rd axis e.g.
        axis=2). Note that if this method is called as part of a sup3r
        resolution forward pass, the date_range_kwargs will be included
        automatically for the current chunk.
    bias_fp : str
        Filepath to statistical distributions file from the bias calc module.
        Must have datasets "bias_{feature_name}_params",
        "bias_fut_{feature_name}_params", and "base_{base_dset}_params" that
        are the parameters to define the statistical distributions to be used
        to correct the given `data`.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    no_trend: bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias by
        comparing the distributions of the biased dataset with a reference
        datasets. See
        ``params_mf`` of :class:`rex.utilities.bc_utils.QuantileDeltaMapping`.
        Note that this assumes that params_mh is the data distribution
        representative for the target data.
    delta_denom_min : float | None
        Option to specify a minimum value for the denominator term in the
        calculation of a relative delta value. This prevents division by a
        very small number making delta blow up and resulting in very large
        output bias corrected values. See equation 4 of Cannon et al., 2015
        for the delta term.
    delta_denom_zero : float | None
        Option to specify a value to replace zeros in the denominator term
        in the calculation of a relative delta value. This prevents
        division by a very small number making delta blow up and resulting
        in very large output bias corrected values. See equation 4 of
        Cannon et al., 2015 for the delta term.
    delta_range : tuple | None
        Option to set a (min, max) on the delta term in QDM. This can help
        prevent QDM from making non-realistic increases/decreases in
        otherwise physical values. See equation 4 of Cannon et al., 2015 for
        the delta term.
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.
    max_workers: int | None
        Max number of workers to use for QDM process pool

    Returns
    -------
    out : np.ndarray
        The input data corrected by QDM. Its shape is the same of the input
        (spatial, spatial, time_window, temporal). The dimension time_window
        aligns with the number of time windows defined in a year, while
        temporal aligns with the time of the data.

    See Also
    --------
    sup3r.bias.qdm.QuantileDeltaMappingCorrection :
        Estimate probability distributions required by QDM method
    rex.utilities.bc_utils.QuantileDeltaMapping :
        Core QDM transformation.

    Notes
    -----
    Be careful selecting `bias_fp`. Usually, the input `data` used here would
    be related to the dataset used to estimate
    "bias_fut_{feature_name}_params".

    Keeping arguments as consistent as possible with `local_linear_bc()`, thus
    a 4D data (spatial, spatial, time_window, temporal), and lat_lon (n_lats,
    n_lons, [lat, lon]). But `QuantileDeltaMapping()`, from rex library,
    expects an array, (time, space), thus we need to re-organize our input to
    match that, and in the end bring it back to (spatial, spatial, time_window,
    temporal). This is still better than maintaining the same functionality
    consistent in two libraries.

    Also, :class:`rex.utilities.bc_utils.QuantileDeltaMapping` expects params
    to be 2D (space, N-params).

    Examples
    --------
    >>> unbiased = local_qdm_bc(biased_array, lat_lon_array, "ghi", "rsds",
    ...                         "./dist_params.hdf")

    """
    # Confirm that the given time matches the expected data size
    msg = f'data was expected to be a 3D array but got shape {data.shape}'
    assert data.ndim == 3, msg
    time_index = make_time_index_from_kws(date_range_kwargs)
    msg = (
        f'Time should align with data 3rd dimension but got data '
        f'{data.shape} and time_index length '
        f'{time_index.size}: {time_index}'
    )
    assert data.shape[-1] == time_index.size, msg

    params = _get_spatial_bc_quantiles(
        lat_lon=lat_lon,
        base_dset=base_dset,
        feature_name=feature_name,
        bias_fp=bias_fp,
        threshold=threshold,
    )

    cfg = params['cfg']

    # params as dask arrays slows down QDM by several orders of magnitude
    base_params = np.array(params['base'])
    bias_params = np.array(params['bias'])
    bias_fut_params = None
    if 'bias_fut' in params:
        bias_fut_params = np.array(params['bias_fut'])

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        base_params = base_params[spatial_slice]
        bias_params = bias_params[spatial_slice]
        if bias_fut_params is not None:
            bias_fut_params = bias_fut_params[spatial_slice]

    data_unbiased = da.full_like(data, np.nan)
    closest_time_idx = [
        np.argmin(abs(d - cfg['time_window_center']))
        for d in time_index.day_of_year
    ]

    for nt in set(closest_time_idx):
        subset_idx = closest_time_idx == nt
        mf = None if bias_fut_params is None else bias_fut_params[:, :, nt]
        subset = _apply_qdm(
            subset=data[:, :, subset_idx],
            base_params=base_params[:, :, nt],
            bias_params=bias_params[:, :, nt],
            bias_fut_params=mf,
            dist=cfg.get('dist', 'empirical'),
            sampling=cfg.get('sampling', 'linear'),
            log_base=cfg.get('log_base', 10),
            relative=relative,
            no_trend=no_trend,
            delta_denom_min=delta_denom_min,
            delta_denom_zero=delta_denom_zero,
            delta_range=delta_range,
            max_workers=max_workers,
        )
        data_unbiased[:, :, subset_idx] = subset

    if out_range is not None:
        data_unbiased = np.maximum(data_unbiased, np.min(out_range))
        data_unbiased = np.minimum(data_unbiased, np.max(out_range))
    if not da.isfinite(data_unbiased).all():
        msg = (
            'QDM bias correction resulted in NaN / inf values! If this is a '
            'relative QDM, you may try setting ``delta_denom_min`` or '
            '``delta_denom_zero``'
        )
        logger.error(msg)
        raise RuntimeError(msg)
    return data_unbiased


def _get_spatial_bc_presrat(
    lat_lon: np.array,
    base_dset: str,
    feature_name: str,
    bias_fp: str,
    threshold: float = 0.1,
):
    """Statistical distributions previously estimated for given lat/lon points

    Recover the parameters that describe the statistical distribution
    previously estimated with :class:`~sup3r.bias.PresRat` for three datasets:
    ``base`` (historical reference), ``bias`` (historical biased reference),
    and ``bias_fut`` (the future biased dataset, usually the data to correct).

    Parameters
    ----------
    lat_lon : ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset : str
        Name of feature used as historical reference. A Dataset with name
        "base_{base_dset}_params" will be retrieved from ``bias_fp``.
    feature_name : str
        Name of the feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved from ``bias_fp``.
    bias_fp : str
        Filepath to bias correction file from the bias calc module. Must have
        datasets "base_{base_dset}_params", "bias_{feature_name}_params", and
        "bias_fut_{feature_name}_params" that define the statistical
        distributions.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error
        is raised.

    Returns
    -------
    dict :
        A dictionary containing:
        - base : np.array
          Parameters used to define the statistical distribution estimated for
          the ``base_dset``. It has a shape of (I, J, T, P), where (I, J) are
          the same first two dimensions of the given `lat_lon`; T is time in
          intervals equally spaced along a year. Check
          `cfg['time_window_center']` below to map each T to a day of the
          year; and P is the number of parameters and depends on the type of
          distribution. See :class:`~sup3r.bias.PresRat` for more details.
        - bias : np.array
          Parameters used to define the statistical distribution estimated for
          (historical) ``feature_name``. It has a shape of (I, J, T, P), where
          (I, J) are the same first two dimensions of the given `lat_lon`; T
          is time in intervals equally spaced along a year. Check
          `cfg['time_window_center']` to map each T to a day of the year; and P
          is the number of parameters and depends on the type of distribution.
          See :class:`~sup3r.bias.PresRat` for more details.
        - bias_fut : np.array
          Parameters used to define the statistical distribution estimated for
          (future) ``feature_name``. It has a shape of (I, J, T, P), where
          (I, J) are the same first two dimensions of the given `lat_lon`; T
          is time in intervals equally spaced along a year. Check
          `cfg['time_window_center']` to map each T to a day of the year; and
          P is the number of parameters used and depends on the type of
          distribution.
          See :class:`~sup3r.bias.PresRat` for more details.
        - bias_tau_fut : np.array
          The threshold for negligible magnitudes. Any value smaller than that
          should be replaced by zero to preserve the zero (precipitation)
          rate. I has dimension of (I, J, dummy), where (I, J) are the same
          first two dimensions of the given `lat_lon`; and a dummy 3rd
          dimension required due to the way sup3r saves data in an HDF.
        - k_factor : np.array
          The K factor used to preserve the mean rate of change from the model
          (see [Pierce2015]_). It has a shape of (I, J, T), where (I, J) are
          the same first two dimensions of the given `lat_lon`; T is time in
          intervals equally spaced along a year. Check
          `cfg['time_window_center']` to map each T to a day of the year.
        - cfg
          Metadata used to guide how to use of the previous parameters on
          reconstructing the statistical distributions. For instance,
          `cfg['dist']` defines the type of distribution, and
          `cfg['time_window_center']` maps the dimension T in days of the
          year for the dimension T of the parameters above. See
          :class:`~sup3r.bias.PresRat` for more details, including which
          metadata is saved.

    Warnings
    --------
    Be careful selecting which `bias_fp` to use. In particular, if
    "bias_fut_{feature_name}_params" is representative for the desired target
    period.

    See Also
    --------
    sup3r.bias.PresRat
        Estimate the statistical distributions loaded here.

    References
    ----------
    .. [Pierce2015] Pierce, D. W., Cayan, D. R., Maurer, E. P., Abatzoglou, J.
       T., & Hegewisch, K. C. (2015). Improved bias correction techniques for
       hydrological simulations of climate change. Journal of Hydrometeorology,
       16(6), 2421-2442.

    Examples
    --------
    >>> lat_lon = np.array([
    ...              [39.649033, -105.46875 ],
    ...              [39.649033, -104.765625]])
    >>> params = _get_spatial_bc_quantiles(
    ...             lat_lon, "ghi", "rsds", "./dist_params.hdf")

    """
    var_names = {
        'base': f'base_{base_dset}_params',
        'bias': f'bias_{feature_name}_params',
        'bias_fut': f'bias_fut_{feature_name}_params',
        'bias_tau_fut': f'{feature_name}_tau_fut',
        'k_factor': f'{feature_name}_k_factor',
    }
    target = lat_lon[-1, 0, :]
    shape = lat_lon.shape[:-1]
    return _get_factors(
        target=target,
        shape=shape,
        var_names=var_names,
        bias_fp=bias_fp,
        threshold=threshold,
    )


def local_presrat_bc(
    data: np.ndarray,
    lat_lon: np.ndarray,
    base_dset: str,
    feature_name: str,
    bias_fp,
    date_range_kwargs: dict,
    lr_padded_slice=None,
    threshold=0.1,
    relative=True,
    no_trend=False,
    delta_denom_min=None,
    delta_denom_zero=None,
    delta_range=None,
    k_range=None,
    out_range=None,
    max_workers=1,
):
    """Bias correction using PresRat

    Parameters
    ----------
    data : np.ndarray
        Sup3r input data to be bias corrected, assumed to be 3D with shape
        (spatial, spatial, temporal) for a single feature.
    lat_lon : np.ndarray
        Array of latitudes and longitudes for the domain to bias correct
        (n_lats, n_lons, 2)
    base_dset :
        Name of feature that is used as (historical) reference. Dataset with
        names "base_{base_dset}_params" will be retrieved.
    feature_name : str
        Name of feature that is being corrected. Datasets with names
        "bias_{feature_name}_params" and "bias_fut_{feature_name}_params" will
        be retrieved.
    bias_fp : str
        Filepath to statistical distributions file from the bias calc module.
        Must have datasets "bias_{feature_name}_params",
        "bias_fut_{feature_name}_params", and "base_{base_dset}_params" that
        are the parameters to define the statistical distributions to be used
        to correct the given `data`.
    date_range_kwargs : dict
        Keyword args for pd.date_range to produce a DatetimeIndex object
        associated with the input data temporal axis (assumed 3rd axis e.g.
        axis=2). Note that if this method is called as part of a sup3r
        resolution forward pass, the date_range_kwargs will be included
        automatically for the current chunk.
    lr_padded_slice : tuple | None
        Tuple of length four that slices (spatial_1, spatial_2, temporal,
        features) where each tuple entry is a slice object for that axes.
        Note that if this method is called as part of a sup3r forward pass, the
        lr_padded_slice will be included automatically in the kwargs for the
        active chunk. If this is None, no slicing will be done and the full
        bias correction source shape will be used.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the coordinates are
        more than this value away from the bias correction lat/lon, an error
        is raised.
    relative : bool
        Apply QDM correction as a relative factor (product), otherwise, it is
        applied as an offset (sum).
    no_trend : bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias by
        comparing the distributions of the biased dataset with a reference
        datasets, without reinforcing the zero rate or applying the k-factor.
        See ``params_mf`` of
        :class:`rex.utilities.bc_utils.QuantileDeltaMapping`. Note that this
        assumes that params_mh is the data distribution representative for the
        target data.
    delta_denom_min : float | None
        Option to specify a minimum value for the denominator term in the
        calculation of a relative delta value. This prevents division by a
        very small number making delta blow up and resulting in very large
        output bias corrected values. See equation 4 of Cannon et al., 2015
        for the delta term. If this is not set, the ``zero_rate_threshold``
        calculated as part of the presrat bias calculation will be used
    delta_denom_zero : float | None
        Option to specify a value to replace zeros in the denominator term
        in the calculation of a relative delta value. This prevents
        division by a very small number making delta blow up and resulting
        in very large output bias corrected values. See equation 4 of
        Cannon et al., 2015 for the delta term.
    delta_range : tuple | None
        Option to set a (min, max) on the delta term in QDM. This can help
        prevent QDM from making non-realistic increases/decreases in
        otherwise physical values. See equation 4 of Cannon et al., 2015 for
        the delta term.
    k_range : tuple | None
        Option to set a (min, max) value for the k-factor multiplier
    out_range : None | tuple
        Option to set floor/ceiling values on the output data.
    max_workers : int | None
        Max number of workers to use for QDM process pool
    """
    time_index = make_time_index_from_kws(date_range_kwargs)
    msg = f'data was expected to be a 3D array but got shape {data.shape}'
    assert data.ndim == 3, msg
    msg = (
        f'Time should align with data 3rd dimension but got data '
        f'{data.shape} and time_index length '
        f'{time_index.size}: {time_index}'
    )
    assert data.shape[-1] == time_index.size, msg

    params = _get_spatial_bc_presrat(
        lat_lon, base_dset, feature_name, bias_fp, threshold
    )
    cfg = params['cfg']
    base_params = params['base']
    bias_params = params['bias']
    bias_fut_params = params['bias_fut']
    bias_tau_fut = np.asarray(params['bias_tau_fut'])
    k_factor = params['k_factor']
    zero_rate_threshold = cfg['zero_rate_threshold']
    delta_denom_min = delta_denom_min or zero_rate_threshold

    if k_range is not None:
        k_factor = np.maximum(k_factor, np.min(k_range))
        k_factor = np.minimum(k_factor, np.max(k_range))

    logger.debug(
        f'Presrat K Factor has shape {k_factor.shape} and ranges '
        f'from {k_factor.min()} to {k_factor.max()}'
    )

    if lr_padded_slice is not None:
        spatial_slice = (lr_padded_slice[0], lr_padded_slice[1])
        base_params = base_params[spatial_slice]
        bias_params = bias_params[spatial_slice]
        if bias_fut_params is not None:
            bias_fut_params = bias_fut_params[spatial_slice]

    data_unbiased = da.full_like(data, np.nan)
    closest_time_idx = [
        np.argmin(abs(d - cfg['time_window_center']))
        for d in time_index.day_of_year
    ]

    for nt in set(closest_time_idx):
        subset_idx = closest_time_idx == nt
        mf = None if bias_fut_params is None else bias_fut_params[:, :, nt]
        subset = _apply_qdm(
            subset=data[:, :, subset_idx],
            base_params=base_params[:, :, nt],
            bias_params=bias_params[:, :, nt],
            bias_fut_params=mf,
            dist=cfg.get('dist', 'empirical'),
            sampling=cfg.get('sampling', 'linear'),
            log_base=cfg.get('log_base', 10),
            relative=relative,
            no_trend=no_trend,
            delta_denom_min=delta_denom_min,
            delta_denom_zero=delta_denom_zero,
            delta_range=delta_range,
            max_workers=max_workers,
        )
        # If no trend, it doesn't make sense to correct for zero rate or
        # apply the k-factor, but limit to QDM only.
        if not no_trend:
            subset = np.where(
                subset < bias_tau_fut, 0, subset * k_factor[:, :, nt : nt + 1]
            )

        data_unbiased[:, :, subset_idx] = subset

    if out_range is not None:
        data_unbiased = np.maximum(data_unbiased, np.min(out_range))
        data_unbiased = np.minimum(data_unbiased, np.max(out_range))

    if da.isnan(data_unbiased).any():
        msg = (
            'Presrat bias correction resulted in NaN values! If this is a '
            'relative QDM, you may try setting ``delta_denom_min`` or '
            '``delta_denom_zero``'
        )
        logger.error(msg)
        raise RuntimeError(msg)

    return data_unbiased
