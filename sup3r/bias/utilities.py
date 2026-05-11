"""Bias correction methods which can be applied to data handler data."""

import logging
import os
from inspect import signature
from warnings import warn

import numpy as np
from rex import Resource

import sup3r.bias.bias_transforms
from sup3r.bias.bias_transforms import _get_spatial_bc_factors, local_qdm_bc
from sup3r.preprocessing.utilities import (
    _parse_time_slice,
    get_date_range_kwargs,
)

logger = logging.getLogger(__name__)


def lin_bc(handler, bc_files, bias_feature=None, threshold=0.1):
    """Bias correct the data in this DataHandler in place using linear bias
    correction factors from files output by MonthlyLinearCorrection or
    LinearCorrection from sup3r.bias.bias_calc

    Parameters
    ----------
    handler : DataHandler
        DataHandler instance with `.data` attribute containing data to
        bias correct
    bc_files : list | tuple | str
        One or more filepaths to .h5 files output by
        MonthlyLinearCorrection or LinearCorrection. These should contain
        datasets named "{feature}_scalar" and "{feature}_adder" where
        {feature} is one of the features contained by this DataHandler and
        the data is a 3D array of shape (lat, lon, time) where time is
        length 1 for annual correction or 12 for monthly correction.
    bias_feature : str | None
        Name of the feature used as a reference. Dataset with
        name "base_{bias_feature}_scalar" and
        "base_{bias_feature}_adder" will be retrieved from ``bc_files``.
    threshold : float
        Nearest neighbor euclidean distance threshold. If the DataHandler
        coordinates are more than this value away from the bias correction
        lat/lon, an error is raised.
    """

    if isinstance(bc_files, str):
        bc_files = [bc_files]

    completed = []
    for feature in handler.features:
        for fp in bc_files:
            ref_feature = (
                feature if bias_feature is None else bias_feature
            )
            dset_scalar = f'{ref_feature}_scalar'
            dset_adder = f'{ref_feature}_adder'
            with Resource(fp) as res:
                dsets = [dset.lower() for dset in res.dsets]
                check = (
                    dset_scalar.lower() in dsets
                    and dset_adder.lower() in dsets
                )
            if feature not in completed and check:
                out = _get_spatial_bc_factors(
                    lat_lon=handler.lat_lon,
                    feature_name=ref_feature,
                    bias_fp=fp,
                    threshold=threshold,
                )
                scalar, adder = out['scalar'], out['adder']

                if scalar.shape[-1] == 1:
                    scalar = np.repeat(scalar, handler.shape[2], axis=2)
                    adder = np.repeat(adder, handler.shape[2], axis=2)
                elif scalar.shape[-1] == 12:
                    idm = handler.time_index.month.values - 1
                    scalar = scalar[..., idm]
                    adder = adder[..., idm]
                else:
                    msg = (
                        'Can only accept bias correction factors '
                        'with last dim equal to 1 or 12 but '
                        'received bias correction factors with '
                        'shape {}'.format(scalar.shape)
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)

                logger.info(
                    'Bias correcting "%s" with linear correction from "%s"',
                    feature,
                    os.path.basename(fp),
                )
                handler.data[feature] = (
                    scalar * handler.data[feature][...] + adder
                )
                completed.append(feature)


def qdm_bc(
    handler,
    bc_files,
    bias_feature,
    relative=True,
    threshold=0.1,
    no_trend=False,
    delta_denom_min=None,
    delta_denom_zero=None,
    delta_range=None,
    out_range=None,
    max_workers=1
):
    """Bias Correction using Quantile Delta Mapping

    Bias correct this DataHandler's data with Quantile Delta Mapping. The
    required statistical distributions should be pre-calculated using
    :class:`sup3r.bias.qdm.QuantileDeltaMappingCorrection`.

    Warning: There is no guarantee that the coefficients from ``bc_files``
    match the resource processed here. Be careful choosing ``bc_files``.

    Parameters
    ----------
    handler : DataHandler
        DataHandler instance with `.data` attribute containing data to
        bias correct
    bc_files : list | tuple | str
        One or more filepaths to .h5 files output by
        :class:`sup3r.bias.qdm.QuantileDeltaMappingCorrection`. These should
        contain datasets named "base_{bias_feature}_params",
        "bias_{feature}_params", and "bias_fut_{feature}_params" where
        {feature} is one of the features contained by this DataHandler and
        the data is a 3D array of shape (lat, lon, time) where time.
    bias_feature : str
        Name of the feature used as (historical) reference. Dataset with
        name "base_{bias_feature}_params" will be retrieved from
        ``bc_files``.
    relative : bool, default=True
        Switcher to apply QDM as a relative (use True) or absolute (use
        False) correction value.
    threshold : float, default=0.1
        Nearest neighbor euclidean distance threshold. If the DataHandler
        coordinates are more than this value away from the bias correction
        lat/lon, an error is raised.
    no_trend: bool, default=False
        An option to ignore the trend component of the correction, thus
        resulting in an ordinary Quantile Mapping, i.e. corrects the bias
        by comparing the distributions of the biased dataset with a
        reference datasets. See ``params_mf`` of
        :class:`rex.utilities.bc_utils.QuantileDeltaMapping`.
        Note that this assumes that "bias_{feature}_params"
        (``params_mh``) is the data distribution representative for the
        target data.
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
    """

    if isinstance(bc_files, str):
        bc_files = [bc_files]

    completed = []
    dr_kwargs = get_date_range_kwargs(handler.time_index)
    for feature in handler.features:
        dset_bc_hist = f'bias_{feature}_params'
        dset_bc_fut = f'bias_fut_{feature}_params'

        for fp in bc_files:
            with Resource(fp) as res:
                check = dset_bc_hist in res.dsets and dset_bc_fut in res.dsets

            if feature not in completed and check:
                logger.info(
                    'Bias correcting "%s" with QDM correction from "%s"',
                    feature,
                    os.path.basename(fp),
                )
                handler.data[feature] = local_qdm_bc(
                    handler.data[feature],
                    handler.lat_lon,
                    bias_feature,
                    feature,
                    bias_fp=fp,
                    date_range_kwargs=dr_kwargs,
                    threshold=threshold,
                    relative=relative,
                    no_trend=no_trend,
                    delta_denom_min=delta_denom_min,
                    delta_denom_zero=delta_denom_zero,
                    delta_range=delta_range,
                    out_range=out_range,
                    max_workers=max_workers

                )
                completed.append(feature)


def bias_correct_feature(
    source_feature,
    input_handler,
    bc_method,
    bc_kwargs,
    time_slice=None,
):
    """Bias correct data using a method defined by the bias_correct_method
    input to :class:`ForwardPassStrategy`

    Parameters
    ----------
    source_feature : str
        The source feature name corresponding to the output feature name
    input_handler : DataHandler
        DataHandler storing raw input data previously used as input for
        forward passes. This is assumed to have data with shape (lats, lons,
        time, features), which can be accessed through the handler with
        handler[feature, lat_slice, lon_slice, time_slice]
    bc_method : Callable
        Bias correction method from `bias_transforms.py`
    bc_kwargs : dict
        Dictionary of keyword arguments for bc_method
    time_slice : slice | None
        Optional time slice to restrict bias correction domain

    Returns
    -------
    data : Union[np.ndarray, da.core.Array]
        Data corrected by the bias_correct_method ready for input to the
        forward pass through the generative model.
    """

    time_slice = _parse_time_slice(time_slice)
    data = input_handler[source_feature][..., time_slice]

    lat_lon = input_handler.lat_lon
    if bc_method is not None:
        bc_method = getattr(sup3r.bias.bias_transforms, bc_method)
        logger.info('Running bias correction with: %s.', bc_method)
        feature_kwargs = bc_kwargs[source_feature]

        if 'date_range_kwargs' in signature(bc_method).parameters:
            ti = input_handler.time_index[time_slice]
            feature_kwargs['date_range_kwargs'] = get_date_range_kwargs(ti)

        use_lrps = 'lr_padded_slice' in signature(bc_method).parameters
        need_lrps = 'lr_padded_slice' not in feature_kwargs
        if use_lrps and need_lrps:
            feature_kwargs['lr_padded_slice'] = None

        use_tavg = 'temporal_avg' in signature(bc_method).parameters
        need_tavg = 'temporal_avg' not in feature_kwargs
        if use_tavg and need_tavg:
            msg = (
                'The kwarg "temporal_avg" was not provided in the bias '
                'correction kwargs but is present in the bias '
                'correction function "{}". If this is not set '
                'appropriately, especially for monthly bias '
                'correction, it could result in QA results that look '
                'worse than they actually are.'.format(bc_method)
            )
            logger.warning(msg)
            warn(msg)

        msg = (
            f'Bias correcting source_feature "{source_feature}" using '
            f'function: {bc_method} with kwargs: {feature_kwargs}'
        )
        logger.debug(msg)

        data = bc_method(data, lat_lon, **feature_kwargs)
    return data


def bias_correct_features(
    features,
    input_handler,
    bc_method,
    bc_kwargs,
    time_slice=None,
):
    """Bias correct all feature data using a method defined by the
    bias_correct_method input to :class:`ForwardPassStrategy`

    See Also
    --------
    :func:`bias_correct_feature`
    """

    time_slice = _parse_time_slice(time_slice)

    for feat in features:
        try:
            input_handler[feat][..., time_slice] = bias_correct_feature(
                source_feature=feat,
                input_handler=input_handler,
                time_slice=time_slice,
                bc_method=bc_method,
                bc_kwargs=bc_kwargs,
            )
        except Exception as e:
            msg = (
                f'Could not run bias correction method {bc_method} on '
                f'feature {feat} time slice {time_slice} with input '
                f'handler of class {type(input_handler)} with shape '
                f'{input_handler.shape}. Received error: {e}'
            )
            logger.exception(msg)
            raise RuntimeError(msg) from e

    return input_handler
