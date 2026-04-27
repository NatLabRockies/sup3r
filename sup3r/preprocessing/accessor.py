"""Accessor for xarray. This defines the basic data object contained by all
``Container`` objects."""

import logging
from typing import Union
from warnings import warn

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import mode
from typing_extensions import Self

from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import (
    _lowered,
    _mem_check,
    compute_if_dask,
    dims_array_tuple,
    is_type_of,
    ordered_array,
    ordered_dims,
    parse_keys,
)
from sup3r.utilities.utilities import nn_fill_array

logger = logging.getLogger(__name__)


@xr.register_dataset_accessor('sx')
class Sup3rX:
    """Accessor for xarray - the suggested way to extend xarray functionality.

    References
    ----------
    https://docs.xarray.dev/en/latest/internals/extending-xarray.html

    Note
    ----
    (1) This is an ``xr.Dataset`` style object with all ``xr.Dataset``
    methods, plus more. The way to access these methods is either through
    appending ``.sx.<method>`` on an ``xr.Dataset`` or by wrapping an
    ``xr.Dataset`` with ``Sup3rX``, e.g. ``Sup3rX(xr.Dataset(...)).<method>``.
    Throughout the ``sup3r`` codebase we prefer to use the latter. The
    most important part of this interface is parsing ``__getitem__`` calls of
    the form ``ds.sx[keys]``.

        (i) ``keys`` can be a single feature name, list of features, or numpy
            style indexing for dimensions. e.g. ``ds.sx['u'][slice(0, 10),
            ...]`` or ``ds.sx[['u', 'v']][..., slice(0, 10)]``.

        (ii) If ``ds[keys]`` returns an ``xr.Dataset`` object then
             ``ds.sx[keys]`` will return a ``Sup3rX`` object. e.g.
            ``ds.sx[['u','v']]``) will return a :class:`Sup3rX` instance but
            ``ds.sx['u']`` will return an ``xr.DataArray``

        (ii) Providing only numpy style indexing without features will return
            an array with all contained features in the last dimension with a
            spatiotemporal shape corresponding to the indices. e.g.
            ``ds[slice(0, 10), 0, 1]`` will return an array of shape ``(10, 1,
            1, n_features)``. This array will be a dask.array or numpy.array,
            depending on whether data is still on disk or loaded into memory.

    (2) The ``__getitem__`` and ``__getattr__`` methods will cast back to
    ``type(self)`` if ``self._ds.__getitem__`` or ``self._ds.__getattr__``
    returns an instance of ``type(self._ds)`` (e.g. an ``xr.Dataset``). This
    means we do not have to constantly append ``.sx`` for successive calls to
    accessor methods.

    Examples
    --------
    >>> # To use as an accessor:
    >>> ds = xr.Dataset(...)
    >>> feature_data = ds.sx[features]
    >>> ti = ds.sx.time_index
    >>> lat_lon_array = ds.sx.lat_lon

    >>> # Use as wrapper:
    >>> ds = Sup3rX(xr.Dataset(data_vars={'windspeed': ...}, ...))
    >>> np_array = ds['windspeed'].values
    >>> dask_array = ds['windspeed'][...] == ds['windspeed'].as_array()
    """

    def __init__(self, ds: Union[xr.Dataset, Self]):
        """Initialize accessor.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            xarray Dataset instance to access with the following methods
        """
        self._ds = ds
        self._features = None
        self._meta = None
        self._shape = None
        self._loaded = None
        self._as_array_cache = None
        self._feature_inds = None
        self._feature_sel_cache = None
        self.time_slice = None

    def _clear_array_cache(self):
        """Clear cached eager array views after mutating the dataset."""
        self._features = None
        self._shape = None
        self._loaded = None
        self._as_array_cache = None
        self._feature_inds = None
        self._feature_sel_cache = None

    def _get_feature_inds(self, features):
        """Get cached feature indices for the ordered eager array."""
        if self._feature_inds is None:
            self._feature_inds = {
                feature: i for i, feature in enumerate(self.features)
            }
        return [self._feature_inds[feature] for feature in features]

    def _get_feature_sel(self, features):
        """Get cached feature selector for eager ordered array slicing."""
        if self._feature_sel_cache is None:
            self._feature_sel_cache = {}

        cache_key = tuple(features)
        if cache_key not in self._feature_sel_cache:
            feature_inds = self._get_feature_inds(features)
            self._feature_sel_cache[cache_key] = (
                slice(None)
                if feature_inds == list(range(len(self.features)))
                else feature_inds
            )

        return self._feature_sel_cache[cache_key]

    def __getitem__(
        self, keys
    ) -> Union[Union[np.ndarray, da.core.Array], Self]:
        """Method for accessing variables. keys can optionally include a
        feature name or list of feature names as the first entry of a keys
        tuple.

        Notes
        -----
        This returns a ``Sup3rX`` object when keys is only an iterable of
        features. e.g. an instance of ``(set, list, tuple)``. When keys is
        only a single feature (a string) this returns an ``xr.DataArray``.
        Otherwise keys must be dimension indices or slices and this will return
        an np.ndarray or dask.array, depending on whether data is loaded into
        memory or not. This array will have features stacked over the last
        dimension.
        """

        features, slices = parse_keys(
            keys,
            default_coords=self.coords,
            default_dims=self._ds.dims,
            default_features=self.features,
        )
        single_feat = isinstance(features, str)

        out = self._ds[features] if len(features) > 0 else self._ds
        out = self.ordered(out) if single_feat else type(self)(out)

        if len(features) > 0:
            return out

        slices = {k: v for k, v in slices.items() if k in out.dims}
        is_fancy = self._needs_fancy_indexing(slices.values())

        if not is_fancy:
            out = out.isel(**slices)

        out = out.as_array()

        if is_fancy and self.loaded:
            # DataArray coord or Numpy indexing
            return out[tuple(slices.values())]

        if is_fancy:
            # DataArray + Dask indexing
            return out.vindex[tuple(slices.values())]

        return out

    def __getattr__(self, attr):
        """Get attribute and cast to ``type(self)`` if an ``xr.Dataset`` is
        returned first."""
        out = getattr(self._ds, attr)
        return type(self)(out) if isinstance(out, xr.Dataset) else out

    def __setitem__(self, keys, data):
        """
        Parameters
        ----------
        keys : str | list | tuple
            keys to set. This can be a string like 'temperature' or a list
            like ``['u', 'v']``. ``data`` will be iterated over in the latter
            case.
        data : Union[np.ndarray, da.core.Array] | xr.DataArray
            array object used to set variable data. If ``variable`` is a list
            then this is expected to have a trailing dimension with length
            equal to the length of the list.
        """
        if is_type_of(keys, str):
            if isinstance(keys, (list, tuple)) and hasattr(data, 'data_vars'):
                data_dict = {v: data[v] for v in keys}
            elif isinstance(keys, (list, tuple)):
                data_dict = {v: data[..., i] for i, v in enumerate(keys)}
            else:
                data_dict = {keys.lower(): data}
            _ = self.assign(data_dict)
        else:
            msg = f'Cannot set values for keys {keys}'
            logger.error(msg)
            raise KeyError(msg)

    def __contains__(self, vals):
        """Check if ``self._ds`` contains ``vals``.

        Parameters
        ----------
        vals : str | list
            Values to check. Can be a list of strings or a single string.

        Examples
        --------
        >>> bool(['u', 'v'] in self)
        >>> bool('u' in self)
        """
        feature_check = isinstance(vals, (list, tuple)) and all(
            isinstance(s, str) for s in vals
        )
        if feature_check:
            return all(s.lower() in self._ds for s in vals)
        return self._ds.__contains__(vals)

    @property
    def values(self):
        """Return numpy values in standard dimension order ``(lats, lons, time,
        ..., features)``"""
        out = self.as_array()
        if not self.loaded:
            return np.asarray(out)
        return out

    def to_dataarray(self) -> Union[np.ndarray, da.core.Array]:
        """Return xr.DataArray for the contained xr.Dataset."""
        if not self.features:
            coords = [self._ds[f] for f in Dimension.coords_2d()]
            return da.stack(coords, axis=-1)
        return self.ordered(self._ds.to_array())

    def as_array(self):
        """Return ``.data`` attribute of an xarray.DataArray with our standard
        dimension order ``(lats, lons, time, ..., features)``"""
        if self.loaded and self._as_array_cache is not None:
            return self._as_array_cache

        out = self.to_dataarray()
        out = getattr(out, 'data', out)

        if self.loaded:
            self._as_array_cache = out

        return out

    def _stack_features(self, arrs):
        if self.loaded:
            return np.stack(arrs, axis=-1)
        return da.stack(arrs, axis=-1)

    def compute(self, **kwargs):
        """Load `._ds` into memory. This updates the internal `xr.Dataset` if
        it has not been loaded already."""
        self._clear_array_cache()
        if not self.loaded:
            logger.debug(f'Loading dataset into memory: {self._ds}')
            logger.debug(f'Pre-loading: {_mem_check()}')

            for f in list(self._ds.data_vars) + list(self._ds.coords):
                if hasattr(self._ds[f], 'compute'):
                    self._ds[f] = self._ds[f].compute(**kwargs)
                logger.debug(
                    f'Loaded {f} into memory with shape '
                    f'{self._ds[f].shape}. {_mem_check()}'
                )
            logger.debug(f'Loaded dataset into memory: {self._ds}')
            logger.debug(f'Post-loading: {_mem_check()}')
            self._loaded = True
        return self

    @property
    def loaded(self):
        """Check if data has been loaded as numpy arrays."""
        if self._loaded is None:
            self._loaded = all(
                isinstance(self._ds[f].data, np.ndarray)
                for f in self._ds.data_vars
            )
        return self._loaded

    @property
    def flattened(self):
        """Check if the contained data is flattened 2D data or 3D rasterized
        data."""
        return Dimension.FLATTENED_SPATIAL in self.dims

    @property
    def time_independent(self):
        """Check if the contained data is time independent."""
        return Dimension.TIME not in self.dims

    def update_ds(self, new_dset, attrs=None):
        """Update `self._ds` with coords and data_vars replaced with those
        provided. These are both provided as dictionaries {name: dask.array}.

        Parameters
        ----------
        new_dset : dict[str, dask.array]
            Can contain any existing or new variable / coordinate as long as
            they all have a consistent shape.

        Returns
        -------
        _ds : xr.Dataset
            Updated dataset with provided coordinates and data_vars with
            variables in our standard dimension order.
        """
        coords = dict(self._ds.coords)
        data_vars = dict(self._ds.data_vars)
        new_coords = {
            k: dims_array_tuple(v) for k, v in new_dset.items() if k in coords
        }
        coords.update(new_coords)
        new_data = {
            k: dims_array_tuple(v)
            for k, v in new_dset.items()
            if k not in coords
        }
        data_vars.update(new_data)

        self._ds = xr.Dataset(coords=coords, data_vars=data_vars, attrs=attrs)
        self._clear_array_cache()
        return self

    @property
    def name(self):
        """Name of dataset. Used to label datasets when grouped in
        :class:`Data` objects. e.g. for low / high res pairs or daily / hourly
        data."""
        return self._ds.attrs.get('name', None)

    def ordered(self, data):
        """Return data with dimensions in standard order ``(lats, lons, time,
        ..., features)``"""
        if data.dims != ordered_dims(data.dims):
            return data.transpose(*ordered_dims(data.dims), ...)
        return data

    def sample(self, idx):
        """Get sample from ``self._ds``. The idx should be a tuple of slices
        for the dimensions ``(south_north, west_east, time)`` and a list of
        feature names. e.g.
        ``(slice(0, 3), slice(1, 10), slice(None), ['u_10m', 'v_10m'])``"""
        isel_kwargs = dict(zip(Dimension.dims_3d(), idx[:-1]))
        features = (
            _lowered(idx[-1]) if is_type_of(idx[-1], str) else self.features
        )

        if self.loaded:
            feature_sel = self._get_feature_sel(features)
            return self.as_array()[(*idx[:-1], feature_sel)]

        out = self._ds[features].isel(**isel_kwargs)
        return self.ordered(out.to_array()).data

    @name.setter
    def name(self, value):
        """Set name of dataset."""
        self._ds.attrs['name'] = value

    def isel(self, *args, **kwargs):
        """Override xr.Dataset.isel to cast back to Sup3rX object."""
        return type(self)(self._ds.isel(*args, **kwargs))

    def coarsen(self, *args, **kwargs):
        """Override xr.Dataset.coarsen to cast back to Sup3rX object."""
        return type(self)(self._ds.coarsen(*args, **kwargs))

    def mean(self, **kwargs):
        """Get mean directly from dataset object."""
        features = kwargs.pop('features', None)
        out = (
            self._ds[features].mean(**kwargs)
            if features is not None
            else self._ds.mean(**kwargs)
        )
        return type(self)(out) if isinstance(out, xr.Dataset) else out

    def std(self, **kwargs):
        """Get std directly from dataset object."""
        features = kwargs.pop('features', None)
        out = (
            self._ds[features].std(**kwargs)
            if features is not None
            else self._ds.std(**kwargs)
        )
        return type(self)(out) if isinstance(out, xr.Dataset) else out

    def normalize(self, means, stds):
        """Normalize dataset using given means and stds. These are provided as
        dictionaries."""
        self._clear_array_cache()
        feats = set(self._ds.data_vars).intersection(means).intersection(stds)
        for f in feats:
            self._ds[f] = (self._ds[f] - means[f]) / stds[f]

    def interpolate_na(self, **kwargs):
        """Use `xr.DataArray.interpolate_na` to fill NaN values with a dask
        compatible method."""
        features = kwargs.pop('features', list(self.data_vars))
        kwargs['fill_value'] = kwargs.get('fill_value', 'extrapolate')
        for feat in features:
            if 'dim' in kwargs:
                if kwargs['dim'] == Dimension.TIME:
                    kwargs['use_coordinate'] = kwargs.get(
                        'use_coordinate', False
                    )
                self._ds[feat] = self._ds[feat].interpolate_na(**kwargs)
            elif np.isnan(self._ds[feat]).any():
                msg = (
                    'No dim given for interpolate_na. This will use nearest '
                    f'neighbor fill to interpolate {feat}, which could take '
                    'some time.'
                )
                logger.warning(msg)
                warn(msg)
                self._ds[feat] = (
                    self._ds[feat].dims,
                    nn_fill_array(self._ds[feat].values),
                )
        return self

    @staticmethod
    def _needs_fancy_indexing(keys) -> Union[np.ndarray, da.core.Array]:
        """We use `.vindex` if keys require fancy indexing."""
        where_list = [
            ind for ind in keys if isinstance(ind, np.ndarray) and ind.ndim > 0
        ]
        return len(where_list) > 1

    def add_dims_to_data_vars(self, vals):
        """Add dimensions to vals entries if needed. This is used to set values
        of `self._ds` which can require dimensions to be explicitly specified
        for the data being set. e.g. self._ds['u_100m'] = (('south_north',
        'west_east', 'time'), data). We make guesses on the correct dims if
        they are missing and give a warning. We add attributes if available in
        vals, as well

        Parameters
        ----------
        vals : dict[Str, Union]
            Dictionary of feature names and arrays to use for setting feature
            data. When arrays are >2 dimensions xarray needs explicit dimension
            info, so we need to add these if not provided.
        """
        new_vals = {}
        for k, v in vals.items():
            if isinstance(v, tuple):
                new_vals[k] = v
            elif isinstance(v, (xr.DataArray, xr.Dataset)):
                dat = v if isinstance(v, xr.DataArray) else v[k]
                data = (
                    ordered_array(dat).squeeze(dim='variable').data
                    if 'variable' in dat.dims
                    else ordered_array(dat).data
                )
                new_vals[k] = (
                    ordered_dims(dat.dims),
                    data,
                    getattr(dat, 'attrs', {}),
                )
            elif k in self._ds.data_vars or len(v.shape) > 1:
                if k in self._ds.data_vars:
                    val = (ordered_dims(self._ds[k].dims), v)
                else:
                    val = dims_array_tuple(v)
                msg = (
                    f'Setting data for variable "{k}" without explicitly '
                    f'providing dimensions. Using dims = {tuple(val[0])}.'
                )
                logger.warning(msg)
                warn(msg)
                new_vals[k] = val
            else:
                new_vals[k] = v
        return new_vals

    def assign(
        self, vals: dict[str, Union[Union[np.ndarray, da.core.Array], tuple]]
    ):
        """Override xarray assign and assign_coords methods to enable update
        without explicitly providing dimensions if variable already exists.

        Parameters
        ----------
        vals : dict
            Dictionary of variable names and either arrays or tuples of (dims,
            array). If dims are not provided this will try to use stored dims
            of the variable, if it exists already.
        """
        self._clear_array_cache()
        data_vars = self.add_dims_to_data_vars(vals)
        if all(f in self.coords for f in vals):
            self._ds = self._ds.assign_coords(data_vars)
        else:
            self._ds = self._ds.assign(data_vars)
        return self

    @property
    def features(self):
        """Features in this container."""
        if self._features is None:
            self._features = list(self._ds.data_vars)
        return self._features

    @property
    def dtype(self):
        """Get dtype of underlying array."""
        return self.to_array().dtype

    @property
    def shape(self):
        """Get shape of underlying xr.DataArray, using our standard dimension
        order."""
        if self._shape is None:
            dim_dict = dict(self._ds.sizes)
            dim_vals = [
                dim_dict[k] for k in Dimension.order() if k in dim_dict
            ]
            self._shape = (*dim_vals, len(self.features))
        return self._shape

    @property
    def size(self):
        """Get size of data contained to use in weight calculations."""
        return np.prod(self.shape)

    @property
    def time_index(self):
        """Base time index for contained data."""
        return (
            pd.to_datetime(self._ds.indexes[Dimension.TIME])
            if Dimension.TIME in self._ds.indexes
            else None
        )

    @time_index.setter
    def time_index(self, value):
        """Update the time_index attribute with given index."""
        self._ds.indexes['time'] = value

    @property
    def time_step(self):
        """Get time step in seconds."""
        sec_diff = (self.time_index[1:] - self.time_index[:-1]).total_seconds()
        return float(mode(sec_diff, keepdims=False).mode)

    @property
    def lat_lon(self) -> Union[np.ndarray, da.core.Array]:
        """Base lat lon for contained data."""
        coords = [self._ds[d] for d in Dimension.coords_2d()]
        lat_lon = self._stack_features(coords)

        # only one coordinate but this property is assumed to be a 2D array
        if len(lat_lon.shape) == 1:
            lat_lon = lat_lon.reshape((1, 1, 2))

        return lat_lon

    @lat_lon.setter
    def lat_lon(self, lat_lon):
        """Update the lat_lon attribute with array values."""
        self[Dimension.coords_2d()] = lat_lon

    @property
    def target(self):
        """Return the value of the lower left hand coordinate."""
        return np.asarray(self.lat_lon[-1, 0])

    @property
    def grid_shape(self):
        """Return the shape of the spatial dimensions."""
        return self.lat_lon.shape[:-1]

    @property
    def meta(self):
        """Return dataframe of flattened lat / lon values. Can also be set to
        include additional data like elevation, country, state, etc"""
        if self._meta is None:
            self._meta = pd.DataFrame(
                columns=Dimension.coords_2d(),
                data=self.lat_lon.reshape((-1, 2)),
            )
        return self._meta

    @meta.setter
    def meta(self, meta):
        """Set meta data. Used to update meta with additional info from
        datasets like WTK and NSRDB."""
        self._meta = meta

    def unflatten(self, grid_shape):
        """Convert flattened dataset into rasterized dataset with the given
        grid shape."""
        if self.flattened:
            ind = pd.MultiIndex.from_product(
                (np.arange(grid_shape[0]), np.arange(grid_shape[1])),
                names=Dimension.dims_2d(),
            )
            self._ds = self._ds.assign({Dimension.FLATTENED_SPATIAL: ind})
            self._ds = self._ds.unstack(Dimension.FLATTENED_SPATIAL)
        else:
            msg = 'Dataset is already unflattened'
            logger.warning(msg)
            warn(msg)
        return self

    def flatten(self):
        """Flatten rasterized dataset so that there is only a single spatial
        dimension."""
        if not self.flattened:
            dims = {Dimension.FLATTENED_SPATIAL: Dimension.dims_2d()}
            self._ds = self._ds.stack(dims)
            index = np.arange(len(self._ds[Dimension.FLATTENED_SPATIAL]))
            self._ds = self._ds.drop_vars(Dimension.dims_2d())
            self._ds = self._ds.assign({Dimension.FLATTENED_SPATIAL: index})
        else:
            msg = 'Dataset is already flattened'
            logger.warning(msg)
            warn(msg)
        return self

    def set_regular_grid(self):
        """In the case of a regular grid, use this to set latitude and
        and longitude as 1D arrays which enables calls to ``self.sel``
        and combining chunks with different coordinates through
        ``xr.combine_by_coords``"""

        lat_lon_2d = (
            len(self._ds[Dimension.LATITUDE].dims) == 2
            and len(self._ds[Dimension.LONGITUDE].dims) == 2
        )
        if not lat_lon_2d:
            msg = (
                'lat/lon data is not 2D. Either the regular grid has '
                'already been set or the data is non-regular'
            )
            logger.warning(msg)
            warn(msg)
            return self

        same_lats = np.allclose(
            np.diff(self._ds[Dimension.LATITUDE].values, axis=1), 0
        )
        same_lons = np.allclose(
            np.diff(self._ds[Dimension.LONGITUDE].values, axis=0), 0
        )
        if not (lat_lon_2d and same_lats and same_lons):
            msg = ('Cannot set regular grid for non-regular data. Latitude '
                   f'check = {same_lats}, Longitude check = {same_lons}.')
            logger.warning(msg)
            warn(msg)
        else:
            self._ds[Dimension.LATITUDE] = self._ds[
                Dimension.LATITUDE
            ].isel(**{Dimension.WEST_EAST: 0})
            self._ds[Dimension.LONGITUDE] = self._ds[
                Dimension.LONGITUDE
            ].isel(**{Dimension.SOUTH_NORTH: 0})
            self._ds = self._ds.swap_dims({
                Dimension.SOUTH_NORTH: Dimension.LATITUDE,
                Dimension.WEST_EAST: Dimension.LONGITUDE,
            })
        return self

    def _qa(self, feature, stats=None):
        """Get qa info for given feature."""
        info = {}
        stats = stats or ['nan_perc', 'std', 'mean', 'min', 'max', 'units']
        logger.info('Running qa on feature: %s', feature)
        nan_count = 100 * np.isnan(self[feature].data).sum()
        nan_perc = nan_count / self[feature].size

        for stat in stats:
            logger.info('Running QA method %s on feature: %s', stat, feature)
            if stat == 'nan_perc':
                info['nan_perc'] = compute_if_dask(nan_perc)
            elif stat == 'units':
                info['units'] = self[feature].attrs.get('units', None)
            else:
                msg = f'Unknown QA method requested: {stat}'
                assert hasattr(self[feature], stat), msg
                qa_data = getattr(self[feature], stat)().data
                info[stat] = compute_if_dask(qa_data)
        return info

    def qa(self, stats=None):
        """Check NaNs and the given stats for all features."""
        qa_info = {}
        for f in self.features:
            qa_info[f] = self._qa(f, stats=stats)
        return qa_info

    def __mul__(self, other):
        """Multiply ``Sup3rX`` object by other. Used to compute weighted means
        and stdevs."""
        try:
            return type(self)(other * self._ds)
        except Exception as e:
            msg = f'Multiplication not supported for type {type(other)}.'
            raise NotImplementedError(msg) from e

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, other):
        """Raise ``Sup3rX`` object to an integer power. Used to compute
        weighted standard deviations."""
        try:
            return type(self)(self._ds**other)
        except Exception as e:
            msg = f'Exponentiation not supported for type {type(other)}.'
            raise NotImplementedError(msg) from e
