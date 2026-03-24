"""Methods used across container objects."""

import logging
import os
import pprint
from glob import glob
from importlib import import_module
from inspect import Parameter, Signature, getfullargspec, signature
from pathlib import Path
from typing import ClassVar, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
import psutil
import xarray as xr
from gaps.cli.documentation import CommandDocumentation

from .names import Dimension

logger = logging.getLogger(__name__)


def lower_names(data):
    """Set all fields / coords / dims to lower case."""
    rename_map = {
        f: f.lower()
        for f in [
            *list(data.data_vars),
            *list(data.dims),
            *list(data.coords),
        ]
        if f != f.lower()
    }
    return data.rename(rename_map)


def get_input_handler_class(input_handler_name: Optional[str] = None):
    """Get the :class:`~sup3r.preprocessing.data_handlers.DataHandler` or
    :class:`~sup3r.preprocessing.rasterizers.Rasterizer` object.

    Parameters
    ----------
    input_handler_name : str
        Class to use for input data. Provide a string name to match a class in
        `sup3r.preprocessing`. If None this will return
        :class:`~sup3r.preprocessing.rasterizers.Rasterizer`, which uses
        `LoaderX` or `LoaderH5` depending on file type. This is a simple
        handler object which does not derive new features from raw data.

    Returns
    -------
    HandlerClass : Rasterizer | DataHandler
        DataHandler or Rasterizer class from sup3r.preprocessing.
    """
    if input_handler_name is None:
        input_handler_name = 'DataHandler'

        logger.info(
            '"input_handler_name" arg was not provided. Using '
            f'"{input_handler_name}". If this is incorrect, please provide '
            'input_handler_name="DataHandlerName".'
        )

    HandlerClass = (
        getattr(import_module('sup3r.preprocessing'), input_handler_name, None)
        if isinstance(input_handler_name, str)
        else None
    )

    if HandlerClass is None:
        msg = (
            'Could not find requested data handler class '
            f'"{input_handler_name}" in sup3r.preprocessing.'
        )
        logger.error(msg)
        raise KeyError(msg)

    return HandlerClass


def _mem_check():
    mem = psutil.virtual_memory()
    return (
        f'Memory usage is {mem.used / 1e9:.3f} GB out of '
        f'{mem.total / 1e9:.3f} GB'
    )


def log_args(func):
    """Decorator to log annotations and args. This can be used to wrap __init__
    methods so we need to pass through the signature and docs"""

    def _get_args_dict(thing, fun, *args, **kwargs):
        """Get args dict from given object and object method."""

        ann_dict = {}
        if '__annotations__' in dir(thing):
            ann_dict = {
                name: getattr(thing, name)
                for name, val in thing.__annotations__.items()
                if val is not ClassVar
            }
        arg_spec = getfullargspec(fun)
        args = args or []
        names = (
            arg_spec.args if 'self' not in arg_spec.args else arg_spec.args[1:]
        )
        names = ['args', *names] if arg_spec.varargs is not None else names
        vals = [None] * len(names)
        defaults = arg_spec.defaults or []
        vals[-len(defaults) :] = defaults
        vals[: len(args)] = args
        args_dict = dict(zip(names, vals))
        args_dict.update(kwargs)
        args_dict.update(ann_dict)

        return args_dict

    def _log_args(thing, fun, *args, **kwargs):
        """Log annotated attributes and args."""

        args_dict = _get_args_dict(thing, fun, *args, **kwargs)
        name = thing.__class__.__name__
        logger.info(
            f'Initialized {name} with:\n{pprint.pformat(args_dict, indent=2)}'
        )
        logger.debug(_mem_check())

    def wrapper(self, *args, **kwargs):
        _log_args(self, func, *args, **kwargs)
        return func(self, *args, **kwargs)

    wrapper.__signature__, wrapper.__doc__ = (
        signature(func),
        getattr(func, '__doc__', ''),
    )
    return wrapper


def get_time_index_freqs(time_index):
    """Get nominal frequency and all unique frequencies for given time index.

    Note
    ----
    The nominal frequency is determined by the minimum time delta in the time
    index. Not seeing an edge case where the nominal frequency would not be the
    minimum frequency but it's possible we should use the most common frequency
    instead?

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Time index to get frequency for.

    Returns
    -------
    nominal_freq : pd.DateOffset
        Nominal frequency for given time index.
    unique_freqs : list
        List of unique frequencies in seconds for given time index.
    """
    unique_freqs = (
        list(set(np.diff(time_index)))
        if len(time_index) > 1
        else [np.timedelta64(1, 'D')]
    )
    unique_freqs /= np.timedelta64(1, 's')
    nominal_freq = pd.tseries.offsets.DateOffset(seconds=min(unique_freqs))
    return nominal_freq, unique_freqs


def get_date_range_kwargs(time_index):
    """Get kwargs for pd.date_range from a DatetimeIndex. This is used to
    provide a concise time_index representation which can be passed through
    the cli and avoid logging lengthly time indices.

    Note
    ----
    If the time index has leap days but is otherwise regular frequency,
    the kwarg ``drop_leap`` will be added to the output dict to indicate
    that leap days should be removed after generating the date range.

    Parameters
    ----------
    time_index : pd.DatetimeIndex
        Output time index.

    Returns
    -------
    kwargs : dict
        Dictionary to pass to pd.date_range(). Can also include kwarg
        ``drop_leap``
    """

    nominal_freq, unique_freqs = get_time_index_freqs(time_index)

    kwargs = {
        'start': time_index[0].strftime('%Y-%m-%d %H:%M:%S'),
        'end': time_index[-1].strftime('%Y-%m-%d %H:%M:%S'),
        'freq': nominal_freq,
    }

    nominal_ti = pd.date_range(**kwargs)

    if len(unique_freqs) > 1 and len(nominal_ti) > len(time_index):
        kwargs['drop_leap'] = True

    elif len(unique_freqs) > 1:
        msg = (
            f'Got more than one unique frequency ({unique_freqs} seconds) for '
            f'time index: {time_index}. This can occur if some input '
            'variables have different time steps. (e.g. daily and hourly '
            'data). Input data must have a consistent frequency.'
        )
        logger.error(msg)
        raise ValueError(msg)

    return kwargs


def make_time_index_from_kws(date_range_kwargs):
    """Function to make a pandas DatetimeIndex from the
    ``get_date_range_kwargs`` outputs

    Parameters
    ----------
    date_range_kwargs : dict
        Dictionary to pass to pd.date_range(), typically produced from
        ``get_date_range_kwargs()``. Can also include kwarg ``drop_leap``

    Returns
    -------
    time_index : pd.DatetimeIndex
        Output time index.
    """
    drop_leap = date_range_kwargs.pop('drop_leap', False)
    time_index = pd.date_range(**date_range_kwargs)

    if drop_leap:
        leap_mask = (time_index.month == 2) & (time_index.day == 29)
        time_index = time_index[~leap_mask]

    return time_index


def _compute_chunks_if_dask(arr):
    return (
        arr.compute_chunk_sizes()
        if hasattr(arr, 'compute_chunk_sizes')
        else arr
    )


def numpy_if_tensor(arr):
    """Cast array to numpy array if it is a tensor."""
    return arr.numpy() if hasattr(arr, 'numpy') else arr


def compute_if_dask(arr):
    """Apply compute method to input if it consists of a dask array or slice
    with dask elements."""
    if isinstance(arr, slice):
        return slice(
            compute_if_dask(arr.start),
            compute_if_dask(arr.stop),
            compute_if_dask(arr.step),
        )
    if isinstance(arr, (tuple, list)):
        return type(arr)(compute_if_dask(a) for a in arr)
    return arr.compute() if hasattr(arr, 'compute') else arr


def _rechunk_if_dask(arr, chunks='auto'):
    if hasattr(arr, 'rechunk'):
        return arr.rechunk(chunks)
    return arr


def _parse_time_slice(value):
    """Parses a value and returns a slice. Input can be a list, tuple, None, or
    a slice."""
    return (
        value
        if isinstance(value, slice)
        else slice(*value)
        if isinstance(value, (tuple, list))
        else slice(None)
    )


def expand_paths(fps):
    """Expand path(s)

    Parameter
    ---------
    fps : str or pathlib.Path or any Sequence of those
        One or multiple paths to file

    Returns
    -------
    list[str]
        A list of expanded unique and sorted paths as str

    Examples
    --------
    >>> expand_paths("myfile.h5")

    >>> expand_paths(["myfile.h5", "*.hdf"])
    """
    if isinstance(fps, (str, Path)):
        fps = (fps,)

    out = []
    for f in fps:
        files = glob(str(f))
        assert any(files), f'Unable to resolve file path: {f}'
        out.extend(files)

    return sorted(set(out))


def get_source_type(file_paths):
    """Get data source type

    Parameters
    ----------
    file_paths : list | str
        One or more paths to data files, can include a unix-style pat*tern

    Returns
    -------
    source_type : str
        "h5", "nc", or "zarr"
    """
    if file_paths is None:
        return None

    if isinstance(file_paths, str) and '*' in file_paths:
        temp = glob(file_paths)
        if any(temp):
            file_paths = temp

    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    _, source_type = os.path.splitext(file_paths[0])

    if source_type in {'.h5', '.hdf'}:
        return 'h5'
    if source_type in {'.nc'}:
        return 'nc'
    if source_type in {'.zarr'}:
        return 'zarr'
    msg = (
        f'Can only handle HDF, NETCDF, or ZARR files. Received unknown '
        f'extension "{source_type}" for files: {file_paths}. We will try to '
        'open this with xarray.'
    )
    logger.warning(msg)
    warn(msg)
    return 'nc'


def get_obj_params(obj):
    """Get available signature parameters for obj and obj bases"""
    objs = (obj, *getattr(obj, '_signature_objs', ()))
    return composite_sig(CommandDocumentation(*objs)).parameters.values()


def get_class_kwargs(obj, kwargs):
    """Get kwargs which match obj signature."""
    param_names = [p.name for p in get_obj_params(obj)]
    return {k: v for k, v in kwargs.items() if k in param_names}


def composite_sig(docs: CommandDocumentation):
    """Get composite signature from command documentation instance."""
    param_names = {
        p.name for sig in docs.signatures for p in sig.parameters.values()
    }
    config = {
        k: v for k, v in docs.template_config.items() if k in param_names
    }
    has_kwargs = config.pop('kwargs', False)
    kw_only = []
    pos_or_kw = []
    for k, v in config.items():
        if v != docs.REQUIRED_TAG:
            kw_only.append(Parameter(k, Parameter.KEYWORD_ONLY, default=v))
        else:
            pos_or_kw.append(Parameter(k, Parameter.POSITIONAL_OR_KEYWORD))

    params = pos_or_kw + kw_only
    if has_kwargs:
        params += [Parameter('kwargs', Parameter.VAR_KEYWORD)]
    return Signature(parameters=params)


def composite_info(objs, skip_params=None):
    """Get composite signature and doc string for given set of objects."""
    objs = objs if isinstance(objs, (tuple, list)) else [objs]
    docs = CommandDocumentation(*objs, skip_params=skip_params)
    return composite_sig(docs), docs.parameter_help


def check_signatures(objs, skip_params=None):
    """Make sure signatures of objects can be parsed for required arguments."""
    docs = CommandDocumentation(*objs, skip_params=skip_params)
    for i, sig in enumerate(docs.signatures):
        msg = (
            f'The signature of {objs[i]!r} cannot be resolved sufficiently. '
            'We need a detailed signature to determine how to distribute '
            'arguments.'
        )

        params = sig.parameters.values()
        assert {p.name for p in params} - {'args', 'kwargs'}, msg


def parse_features(features: Optional[Union[str, list]] = None, data=None):
    """Parse possible inputs for features (list, str, None, 'all'). If 'all'
    this returns all data_vars in data. If None this returns an empty list.

    Note
    ----
    Returns a string if input is a string and list otherwise. Need to manually
    get [features] if a list is required.

    Parameters
    ----------
    features : list | str | None
        Feature request to parse.
    data : T_Dataset
        Data containing available features
    """
    features = (
        list(data.data_vars)
        if features in ('all', ['all']) and data is not None
        else features
    )
    features = lowered(features) if features is not None else []
    return features


def parse_keys(
    keys, default_coords=None, default_dims=None, default_features=None
):
    """Return set of features and slices for all dimensions contained in
    dataset that can be passed to isel and transposed to standard dimension
    order. If keys is empty then we just want to return the coordinate
    data, so features will be set to just the coordinate names."""

    keys = list(keys) if isinstance(keys, set) else keys
    keys = keys if isinstance(keys, tuple) else (keys,)
    has_feats = is_type_of(keys[0], str)

    # return just coords if an empty feature set is requested
    just_coords = (
        isinstance(keys[0], (tuple, list, np.ndarray)) and len(keys[0]) == 0
    )

    if just_coords:
        features = list(default_coords)
    elif has_feats:
        features = lowered(keys[0]) if keys[0] != 'all' else default_features
    else:
        features = []

    if len(features) > 0:
        dim_keys = () if len(keys) == 1 else keys[1:]
    else:
        dim_keys = keys

    dim_keys = parse_ellipsis(dim_keys, dim_num=len(default_dims))
    ordd = ordered_dims(default_dims)

    if len(dim_keys) > len(default_dims):
        msg = (
            'Received keys = %s which are incompatible with the '
            'dimensions = %s. If trying to access features by integer '
            'index instead use feature names.'
        )
        logger.error(msg, keys, ordd)
        raise ValueError(msg % (str(keys), ordd))

    if len(features) > 0 and len(dim_keys) > 0:
        msg = (
            'Received keys = %s which includes both features and '
            'dimension indexing. The correct access pattern is '
            'ds[features][indices]'
        )
        logger.error(msg, keys)
        raise ValueError(msg % str(keys))

    return features, dict(zip(ordd, dim_keys))


def parse_to_list(features=None, data=None):
    """Parse features and return as a list, even if features is a string."""
    features = (
        np.array(
            list(features)
            if isinstance(features, (set, tuple))
            else features
            if isinstance(features, list)
            else [features]
        )
        .flatten()
        .tolist()
    )
    return parse_features(features=features, data=data)


def parse_ellipsis(vals, dim_num):
    """
    Replace ellipsis with N slices where N is dim_num - len(vals) + 1

    Parameters
    ----------
    vals : list | tuple
        Entries that will be used to index an array with dim_num dimensions.
    dim_num : int
        Number of dimensions of array that will be indexed with given vals.
    """
    new_vals = []
    for v in vals:
        if v is Ellipsis:
            needed = dim_num - len(vals) + 1
            new_vals.extend([slice(None)] * needed)
        else:
            new_vals.append(v)
    return new_vals


def contains_ellipsis(vals):
    """Check if vals contain an ellipse. This is used to correctly parse keys
    for ``Sup3rX.__getitem__``"""
    return vals is Ellipsis or (
        isinstance(vals, (tuple, list)) and any(v is Ellipsis for v in vals)
    )


def is_type_of(vals, vtype):
    """Check if vals is an instance of type or group of that type."""
    return isinstance(vals, vtype) or (
        isinstance(vals, (set, tuple, list))
        and all(isinstance(v, vtype) for v in vals)
    )


def _get_strings(vals):
    vals = [vals] if isinstance(vals, str) else vals
    return [v for v in vals if is_type_of(v, str)]


def _lowered(features):
    return (
        features.lower()
        if isinstance(features, str)
        else [f.lower() if isinstance(f, str) else f for f in features]
    )


def lowered(features):
    """Return a lower case version of the given str or list of strings. Used to
    standardize storage and lookup of features."""

    feats = _lowered(features)
    if _get_strings(features) != _get_strings(feats):
        msg = (
            f'Received some upper case features: {_get_strings(features)}. '
            f'Using {_get_strings(feats)} instead.'
        )
        logger.warning(msg)
        warn(msg)
    return feats


def ordered_dims(dims: tuple):
    """Return the order of dims that follows the ordering of Dimension.order()
    for the common dim names. e.g dims = ('time', 'south_north', 'dummy',
    'west_east') will return ('south_north', 'west_east', 'time',
    'dummy')."""
    standard = [dim for dim in Dimension.order() if dim in dims]
    non_standard = [dim for dim in dims if dim not in standard]
    if Dimension.VARIABLE in standard:
        return tuple(standard[:-1] + non_standard + standard[-1:])
    return tuple(standard + non_standard)


def ordered_array(data: xr.DataArray):
    """Transpose arrays so they have a (space, time, ...) or (space, time,
    ..., feature) ordering.

    Parameters
    ----------
    data : xr.DataArray
        xr.DataArray with `.dims` attribute listing all contained dimensions
    """
    return data.transpose(*ordered_dims(data.dims))


def dims_array_tuple(arr):
    """Return a tuple of (dims, array) with dims equal to the ordered slice
    of Dimension.order() with the same len as arr.shape. This is used to set
    xr.Dataset entries. e.g. dset[var] = (dims, array)"""
    if len(arr.shape) > 1:
        arr = (Dimension.order()[1 : len(arr.shape) + 1], arr)
    return arr
