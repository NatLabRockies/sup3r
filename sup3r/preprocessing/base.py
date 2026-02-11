"""Base classes - fundamental dataset objects and the base :class:`Container`
object, which just contains dataset objects. All objects that interact with
data are containers. e.g. loaders, rasterizers, data handlers, samplers, batch
queues, batch handlers.

TODO: https://github.com/xarray-contrib/datatree might be a better approach
for Sup3rDataset concept. Consider migrating once datatree has been fully
integrated into xarray (in progress as of 8/8/2024)
"""

import logging
import pprint
from abc import ABCMeta
from collections.abc import Mapping
from typing import Union
from warnings import warn

import numpy as np
import xarray as xr

import sup3r.preprocessing.accessor  # noqa: F401 # pylint: disable=W0611
from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.utilities import (
    composite_info,
    is_type_of,
)
from sup3r.utilities.utilities import Timer

logger = logging.getLogger(__name__)


def _get_class_info(namespace):
    sig_objs = namespace.get('_signature_objs', None)
    skips = namespace.get('_skip_params', None)
    sig = doc = None
    if sig_objs:
        sig, doc = composite_info(sig_objs, skip_params=skips)
    return sig, doc


class Sup3rMeta(ABCMeta, type):
    """Meta class to define ``__name__``, ``__signature__``, and
    ``__subclasscheck__`` of composite and derived classes. This allows us to
    still resolve a signature for classes which pass through parent args /
    kwargs as ``*args`` / ``**kwargs`` or those built through factory
    composition, for example."""

    def __new__(mcs, name, bases, namespace, **kwargs):
        """Define __name__ and __signature__"""
        sig, doc = _get_class_info(namespace)
        name = namespace.get('__name__', name)
        if sig:
            namespace['__signature__'] = sig
        if '__init__' in namespace and sig:
            namespace['__init__'].__signature__ = sig
        if '__init__' in namespace and doc:
            namespace['__init__'].__doc__ = doc
        return super().__new__(mcs, name, bases, namespace, **kwargs)

    def __subclasscheck__(cls, subclass):
        """Check if factory built class shares base classes."""
        if super().__subclasscheck__(subclass):
            return True
        if hasattr(subclass, '_signature_objs'):
            return {obj.__name__ for obj in cls._signature_objs} == {
                obj.__name__ for obj in subclass._signature_objs
            }
        return False

    def __repr__(cls):
        return f"<class '{cls.__module__}.{cls.__name__}'>"


class DsetTuple:
    """A simple class to mimic namedtuple behavior with dynamic attributes
    while being serializable"""

    def __init__(self, **kwargs):
        self.dset_names = list(kwargs)
        self.__dict__.update(kwargs)

    @property
    def dsets(self):
        """Dictionary with only dset names and associated values."""
        return {k: v for k, v in self.__dict__.items() if k in self.dset_names}

    def __iter__(self):
        return iter(self.dsets.values())

    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.dsets)[key]
        return self.dsets[key]

    def __len__(self):
        return len(self.dsets)

    def __repr__(self):
        return f'DsetTuple({self.dsets})'


class Sup3rDataset:
    """Interface for interacting with one or two ``xr.Dataset`` instances.
    This is a wrapper around one or two ``Sup3rX`` objects so they work well
    with Dual objects like ``DualSampler``, ``DualRasterizer``,
    ``DualBatchHandler``, etc...)

    Examples
    --------
    >>> # access high_res or low_res:
    >>> hr = xr.Dataset(...)
    >>> lr = xr.Dataset(...)
    >>> ds = Sup3rDataset(low_res=lr, high_res=hr)
    >>> ds.high_res; ds.low_res  # returns Sup3rX objects
    >>> ds[feature]  # returns a tuple of dataarray (low_res, high_res)

    >>> # access hourly or daily:
    >>> daily = xr.Dataset(...)
    >>> hourly = xr.Dataset(...)
    >>> ds = Sup3rDataset(daily=daily, hourly=hourly)
    >>> ds.hourly; ds.daily  # returns Sup3rX objects
    >>> ds[feature]  # returns a tuple of dataarray (daily, hourly)

    >>> # single resolution data access:
    >>> xds = xr.Dataset(...)
    >>> ds = Sup3rDataset(hourly=xds)
    >>> ds.hourly  # returns Sup3rX object
    >>> ds[feature]  # returns a single dataarray

    Note
    ----
    (1) This may seem similar to
    :class:`~sup3r.preprocessing.collections.base.Collection`, which also can
    contain multiple data members, but members of
    :class:`~sup3r.preprocessing.collections.base.Collection` objects are
    completely independent while here there are at most two members which are
    related as low / high res versions of the same underlying data.

    (2) Here we make an important choice to use high_res members to compute
    means / stds. It would be reasonable to instead use the average of high_res
    and low_res means / stds for aggregate stats but we want to preserve the
    relationship between coarsened variables after normalization (e.g.
    temperature_2m, temperature_max_2m, temperature_min_2m). This means all
    these variables should have the same means and stds, which ultimately come
    from the high_res non coarsened variable.
    """

    DSET_NAMES = ('low_res', 'high_res', 'obs')

    def __init__(
        self,
        **dsets: Mapping[str, Union[xr.Dataset, Sup3rX]],
    ):
        """
        Parameters
        ----------
        dsets : Mapping[str, xr.Dataset | Sup3rX | Sup3rDataset]
            ``Sup3rDataset`` is initialized from a flexible kwargs input. The
            keys will be used as names in a named tuple and the values will be
            the dataset members. These names will also be used to define
            attributes which point to these dataset members. You can provide
            ``name=data`` or ``name1=data1, name2=data2`` and then access these
            datasets as ``.name1`` or ``.name2``. If dsets values are
            xr.Dataset objects these will be cast to ``Sup3rX`` objects first.
            We also check if dsets values are ``Sup3rDataset`` objects and if
            they only include one data member we use those to reinitialize a
            ``Sup3rDataset``
        """

        for name, dset in dsets.items():
            if isinstance(dset, xr.Dataset):
                dsets[name] = Sup3rX(dset)
            elif isinstance(dset, type(self)):
                msg = (
                    'Initializing Sup3rDataset with Sup3rDataset objects '
                    'which contain more than one member is not allowed.'
                )
                assert len(dset) == 1, msg
                dsets[name] = dset._ds[0]

        self._ds = DsetTuple(**dsets)

    def __iter__(self):
        yield from self._ds

    @property
    def dtype(self):
        """Get datatype of first member. Assumed to be constant for all
        members."""
        return self._ds[0].dtype

    def __len__(self):
        return len(self._ds)

    def __getattr__(self, attr):
        """Get attribute through accessor if available. Otherwise use standard
        xarray interface."""
        if hasattr(self._ds, attr):
            return getattr(self._ds, attr)
        out = [self._getattr(ds, attr) for ds in self._ds]
        if len(self._ds) == 1:
            out = out[0]
        return out

    @classmethod
    def _getattr(cls, dset, attr):
        """Get attribute from single data member."""
        return getattr(dset.sx, attr, getattr(dset, attr))

    @classmethod
    def _getitem(cls, dset, item):
        """Get item from single data member."""
        return dset.sx[item] if hasattr(dset, 'sx') else dset[item]

    def rewrap(self, data):
        """Rewrap data as ``Sup3rDataset`` after calling parent method."""
        if isinstance(data, type(self)):
            return data
        if len(data) == 1:
            return type(self)(high_res=data[0])
        return type(self)(**dict(zip(self.DSET_NAMES, data)))

    def sample(self, idx):
        """Get samples from ``self._ds`` members. idx should be either a tuple
        of slices for the dimensions (south_north, west_east, time) and a list
        of feature names or a tuple of the same, for multi-member datasets
        (dual datasets and dual with observations datasets)."""
        if len(self._ds) > 1:
            return tuple(d.sample(idx[i]) for i, d in enumerate(self))
        return self._ds[-1].sample(idx)

    def isel(self, *args, **kwargs):
        """Return new Sup3rDataset with isel applied to each member."""
        return self.rewrap(tuple(d.isel(*args, **kwargs) for d in self))

    def __getitem__(self, keys):
        """If keys is an int this is interpreted as a request for that member
        of ``self._ds``. Otherwise, if there's only a single member of
        ``self._ds`` we get self._ds[-1][keys]. If there's two members we get
        ``(self._ds[0][keys], self._ds[1][keys])`` and cast this back to a
        ``Sup3rDataset`` if each of ``self._ds[i][keys]`` is a ``Sup3rX``
        object"""
        if isinstance(keys, int):
            return self._ds[keys]

        out = tuple(self._getitem(d, keys) for d in self._ds)
        if len(self._ds) == 1:
            return out[-1]
        if all(isinstance(o, Sup3rX) for o in out):
            return type(self)(**dict(zip(self._ds.dset_names, out)))
        return out

    @property
    def shape(self):
        """We use the shape of the largest data member. These are assumed to be
        ordered as (low-res, high-res) if there are two members."""
        return self._ds[-1].shape

    @property
    def features(self):
        """The features are determined by the set of features from all data
        members."""
        if len(self._ds) == 1:
            return self._ds[0].features
        feats = [
            f for f in self._ds[0].features if f not in self._ds[1].features
        ]
        feats += self._ds[1].features
        return feats

    @property
    def size(self):
        """Return number of elements in the largest data member."""
        return np.prod(self.shape)

    def __contains__(self, vals):
        """Check for vals in all of the dset members."""
        return any(d.sx.__contains__(vals) for d in self._ds)

    def __setitem__(self, keys, data):
        """Set dset member values. Check if values is a tuple / list and if
        so interpret this as sending a tuple / list element to each dset
        member. e.g. ``vals[0] -> dsets[0]``, ``vals[1] -> dsets[1]``, etc"""
        if len(self._ds) == 1:
            self._ds[-1].__setitem__(keys, data)
        else:
            for i, self_i in enumerate(self):
                dat = data[i] if isinstance(data, (tuple, list)) else data
                self_i.__setitem__(keys, dat)

    def mean(self, **kwargs):
        """Use the high_res members to compute the means. These are used for
        normalization during training."""
        kwargs['skipna'] = kwargs.get('skipna', True)
        return self._ds[1 if len(self._ds) > 1 else 0].mean(**kwargs)

    def std(self, **kwargs):
        """Use the high_res members to compute the stds. These are used for
        normalization during training."""
        kwargs['skipna'] = kwargs.get('skipna', True)
        return self._ds[1 if len(self._ds) > 1 else 0].std(**kwargs)

    def normalize(self, means, stds):
        """Normalize dataset using the given mean and stds. These are provided
        as dictionaries."""
        _ = [d.normalize(means=means, stds=stds) for d in self._ds]

    def compute(self, **kwargs):
        """Load data into memory for each data member."""
        _ = [d.compute(**kwargs) for d in self._ds]

    @property
    def loaded(self):
        """Check if all data members have been loaded into memory."""
        return all(d.loaded for d in self._ds)


class Container(metaclass=Sup3rMeta):
    """Basic fundamental object used to build preprocessing objects. Contains
    an xarray-like Dataset (:class:`~.accessor.Sup3rX`), wrapped tuple of
    ``Sup3rX`` objects (:class:`.Sup3rDataset`), or a tuple of such objects.
    """

    __slots__ = ['_data', 'timer']

    def __init__(
        self,
        data: Union[
            Sup3rX,
            Sup3rDataset,
            tuple[Sup3rX, ...],
            tuple[Sup3rDataset, ...],
            dict[str, Sup3rX],
            dict[str, Sup3rDataset],
        ] = None,
    ):
        """
        Parameters
        ----------
        data: Union[Sup3rX, Sup3rDataset, tuple[Sup3rX, ...],
                    tuple[Sup3rDataset, ...]
            Can be an ``xr.Dataset``, a :class:`~.accessor.Sup3rX` object, a
            :class:`.Sup3rDataset` object, or a tuple of such objects.

            Note
            ----
            ``.data`` will return a :class:`~.Sup3rDataset` object or tuple of
            such. This is a tuple when the `.data` attribute belongs to a
            :class:`~.collections.base.Collection` object like
            :class:`~.batch_handlers.factory.BatchHandler`. Otherwise this is
            :class:`~.Sup3rDataset` object, which is either a wrapped 3-tuple,
            2-tuple, or 1-tuple (e.g. ``len(data) == 3``, ``len(data) == 2`` or
            ``len(data) == 1)``. This is a 3-tuple when ``.data`` belongs to a
            container object like :class:`~.samplers.DualSamplerWithObs`, a
            2-tuple when ``.data`` belongs to a dual container object like
            :class:`~.samplers.DualSampler`, and a 1-tuple otherwise.
        """
        self.data = data
        self.timer = Timer()

    @property
    def data(self):
        """Return underlying data.

        Returns
        -------
        :class:`.Sup3rDataset`

        See Also
        --------
        :py:meth:`.wrap`
        """
        return self._data

    @data.setter
    def data(self, data):
        """Set data value. Wrap given value depending on type.

        See Also
        --------
        :py:meth:`.wrap`"""
        self._data = self.wrap(data)

    def wrap(self, data):
        """
        Return a :class:`~.Sup3rDataset` object or tuple of such. This is a
        tuple when the ``.data`` attribute belongs to a
        :class:`~.collections.base.Collection` object like
        :class:`~.batch_handlers.factory.BatchHandler`. Otherwise this is
        :class:`~.Sup3rDataset` object, which is either a wrapped 3-tuple,
        2-tuple, or 1-tuple (e.g. ``len(data) == 3``, ``len(data) == 2`` or
        ``len(data) == 1)``. This is a 3-tuple when ``.data`` belongs to a
        container object like :class:`~.samplers.DualSamplerWithObs`, a 2-tuple
        when ``.data`` belongs to a dual container object like
        :class:`~.samplers.DualSampler`, and a 1-tuple otherwise.
        """
        if data is None:
            return data

        if hasattr(data, 'data'):
            data = data.data

        if is_type_of(data, Sup3rDataset):
            return data

        if isinstance(data, dict):
            data = Sup3rDataset(**data)

        if isinstance(data, tuple) and len(data) > 1:
            msg = (
                f'{self.__class__.__name__}.data is being set with a '
                f'{len(data)}-tuple without explicit dataset names. We will '
                f'assume name ordering: {Sup3rDataset.DSET_NAMES[: len(data)]}'
            )
            logger.warning(msg)
            warn(msg)
            data = Sup3rDataset(**dict(zip(Sup3rDataset.DSET_NAMES, data)))
        elif not isinstance(data, Sup3rDataset):
            name = getattr(data, 'name', None) or 'high_res'
            data = Sup3rDataset(**{name: data})
        return data

    def post_init_log(self, args_dict=None):
        """Log additional arguments after initialization."""
        if args_dict is not None:
            logger.info(
                f'Finished initializing {self.__class__.__name__} with:\n'
                f'{pprint.pformat(args_dict, indent=2)}'
            )

    @property
    def shape(self):
        """Get shape of underlying data."""
        return self.data.shape

    def __len__(self):
        return len(self.data)

    def __contains__(self, vals):
        return vals in self.data

    def __getitem__(self, keys):
        """Get item from underlying data. ``.data`` is a ``Sup3rX`` or
        ``Sup3rDataset`` object, so this uses those ``__getitem__`` methods.

        See Also
        --------
        :py:meth:`.accessor.Sup3rX.__getitem__`,
        :py:meth:`.Sup3rDataset.__getitem__`
        """
        return self.data[keys]

    def __setitem__(self, keys, data):
        """Set item in underlying data."""
        self.data.__setitem__(keys, data)

    def __getattr__(self, attr):
        """Check if attribute is available from ``.data``"""
        try:
            return getattr(self._data, attr)
        except Exception as e:
            msg = f'{self.__class__.__name__} object has no attribute "{attr}"'
            raise AttributeError(msg) from e

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
