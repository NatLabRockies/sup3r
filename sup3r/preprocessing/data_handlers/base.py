"""DataHandler objects, which inherit from the
:class:`~sup3r.preprocessing.derivers.Deriver` class and compose
:class:`~sup3r.preprocessing.rasterizers.Rasterizer`,
:class:`~sup3r.preprocessing.loaders.Loader`,
:class:`~sup3r.preprocessing.cachers.Cacher` classes.

DataHandlers are used to load, rasterize, derive, and cache features from
data. They can be used to load data from various sources, such as HDF5,
NetCDF, or other formats, and to derive features from the loaded data. For
example, you can provide u / v at isobars, geopotential, and surface elevation,
to derive u / v at a specific height. All variables (e.g. temperaute_500m) can
be interpolated from isobars as long as geopotential and surface elevation are
also provided.

TODO: If ``.data`` is a ``Sup3rDataset`` with more than one member only the
high resolution data is cached. We could extend caching and loading to write /
load both with groups
"""

import logging
from typing import Callable, Optional, Union

from rex import MultiFileNSRDBX, MultiFileWindX

from sup3r.preprocessing.base import (
    Sup3rDataset,
)
from sup3r.preprocessing.derivers import Deriver
from sup3r.preprocessing.derivers.methods import (
    RegistryH5SolarCC,
    RegistryH5WindCC,
)
from sup3r.preprocessing.loaders import Loader
from sup3r.preprocessing.rasterizers import Rasterizer
from sup3r.preprocessing.utilities import (
    expand_paths,
    log_args,
    parse_to_list,
)
from sup3r.writers import Cacher
from sup3r.writers.utilities import _check_for_cache

logger = logging.getLogger(__name__)


class DataHandler(Deriver):
    """Base DataHandler. Composes
    :class:`~sup3r.preprocessing.rasterizers.Rasterizer`,
    :class:`~sup3r.preprocessing.loaders.Loader`, and
    :class:`~sup3r.preprocessing.cachers.Cacher` classes.

    Examples
    --------
    Extract wind speed at 40m and 80m above the ground from files for u/v
    at 10m and 100m. Windspeed will be interpolated from surrounding
    levels using a log profile. ``dh`` will contain dask arrays of this
    data with 10x10x50 chunk sizes, which can be accessed with
    ``dh['windspeed_40m']`` or ``dh['windspeed_80m']``. Data will be
    cached to files named 'windspeed_40m.h5' and 'windspeed_80m.h5' in
    './cache_dir' with 5x5x10 chunks on disk.

    >>> cache_chunks = {'south_north': 5, 'west_east': 5, 'time': 10}
    >>> load_chunks = {'south_north': 10, 'west_east': 10, 'time': 50}
    >>> grid_size = (50, 50)
    >>> lower_left_coordinate = (39.7, -105.2)
    >>> dh = DataHandler(
    ...     file_paths=['./data_dir/u_10m.nc', './data_dir/u_100m.nc',
    ...                 './data_dir/v_10m.nc', './data_dir/v_100m.nc'],
    ...     features=['windspeed_40m', 'windspeed_80m'],
    ...     shape=grid_size, time_slice=slice(0, 100),
    ...     target=lower_left_coordinate, hr_spatial_coarsen=2,
    ...     chunks=load_chunks, interp_kwargs={'method': 'log'},
    ...     cache_kwargs={'cache_pattern': './cache_dir/{feature}.h5',
    ...                   'chunks': cache_chunks})

    >>> # Derive more features from already initialized data handler:
    >>> dh['windspeed_60m'] = dh.derive('windspeed_60m')

    Derive wind speed and direction at 200m above the ground from files
    for geopotential height (zg), surface elevation (orog), and u/v at 10m,
    100m, and isobars. Both zg and orog are assumed to be in meters.

    >>> dh = DataHandler(
    ...     file_paths=['./data_dir/u_10m.nc', './data_dir/u_100m.nc',
    ...                 './data_dir/v_10m.nc', './data_dir/v_100m.nc',
    ...                 './data_dir/u_isobars.nc',
    ...                 './data_dir/v_isobars.nc',
    ...                 './data_dir/zg.nc',
    ...                 './data_dir/orog.nc'],
    ...     features=['windspeed_200m', 'winddirection_200m'],
    ...     shape=grid_size, time_slice=slice(0, 100),
    ...     target=lower_left_coordinate,
    ...     chunks=load_chunks, interp_kwargs={'method': 'log'},
    ...     cache_kwargs={'cache_pattern': './cache_dir/{feature}.h5',
    ...                   'chunks': cache_chunks})
    """

    @log_args
    def __init__(
        self,
        file_paths,
        features='all',
        load_features='all',
        res_kwargs: Optional[dict] = None,
        chunks: Union[str, dict[str, int]] = 'auto',
        target: Optional[tuple] = None,
        shape: Optional[tuple] = None,
        time_slice: Union[slice, tuple, list, None] = slice(None),
        threshold: Optional[float] = None,
        time_roll: int = 0,
        time_shift: Optional[int] = None,
        hr_spatial_coarsen: int = 1,
        nan_method_kwargs: Optional[dict] = None,
        feature_aliases: Optional[dict] = None,
        BaseLoader: Optional[Callable] = None,
        FeatureRegistry: Optional[dict] = None,
        interp_kwargs: Optional[dict] = None,
        cache_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        file_paths : str | list | pathlib.Path
            file_paths input to LoaderClass
        features : list | str
            Features to derive. If 'all' then all available raw features will
            just be loaded. Specify explicit feature names for derivations.
        load_features : list | str
            Features to load and make available for derivations. If 'all' then
            all available raw features will be loaded and made available for
            derivations. This can be used to restrict features used for
            derivations. For example, to derive 'temperature_100m' from only
            temperature isobars, from data that includes single level values as
            well (like temperature_2m), don't include 'temperature_2m' in the
            ``load_features`` list.
        res_kwargs : dict
            Additional keyword arguments passed through to the ``BaseLoader``.
            BaseLoader is usually xr.open_mfdataset for NETCDF files and
            MultiFileResourceX for H5 files.
        chunks : dict | str
            Dictionary of chunk sizes to pass through to
            ``dask.array.from_array()`` or ``xr.Dataset().chunk()``. Will be
            converted to a tuple when used in ``from_array()``. These are the
            methods for H5 and NETCDF data, respectively. This argument can
            be "auto" or None in addition to a dictionary. None will not do
            any chunking and load data into memory as ``np.array``
        target : tuple
            (lat, lon) lower left corner of raster. Either need target+shape
            or raster_file.
        shape : tuple
            (rows, cols) grid size. Either need target+shape or raster_file.
        time_slice : slice | list
            Slice specifying extent and step of temporal extraction. e.g.
            slice(start, stop, step). If equal to slice(None, None, 1) the full
            time dimension is selected. Can be also be a list ``[start, stop,
            step]``
        threshold : float
            Nearest neighbor euclidean distance threshold. If the coordinates
            are more than this value away from the target lat/lon, an error is
            raised.
        time_roll : int
            Number of steps to roll along the time axis. `Passed to
            xr.Dataset.roll()`
        time_shift : int | None
            Number of minutes to shift time axis. This can be used, for
            example, to shift the time index for daily data so that the time
            stamp for a given day starts at the zeroth minute instead of at
            noon, as is the case for most GCM data.
        hr_spatial_coarsen : int
            Spatial coarsening factor. Passed to `xr.Dataset.coarsen()`
        nan_method_kwargs : str | dict | None
            Keyword arguments for nan handling. If 'mask', time steps with nans
            will be dropped. Otherwise this should be a dict of kwargs which
            will be passed to
            :py:meth:`sup3r.preprocessing.accessor.Sup3rX.interpolate_na`. e.g.
            {'method': 'linear', 'dim': 'time'}
        feature_aliases : dict
            Optional dictionary of feature aliases to use when loading data.
            This is useful for renaming features to expected sup3r names.
            For example, {'sp': 'pressure_0m', 'u10': u_10m'}.
        BaseLoader : Callable
            Base level file loader wrapped by
            :class:`~sup3r.preprocessing.loaders.Loader`. This is usually
            xr.open_mfdataset for NETCDF files and MultiFileResourceX for H5
            files.
        FeatureRegistry : dict
            Dictionary of
            :class:`~sup3r.preprocessing.derivers.methods.DerivedFeature`
            objects used for derivations
        interp_kwargs : dict | None
            Dictionary of kwargs for level interpolation. Can include "method"
            and "run_level_check" keys. Method specifies how to perform height
            interpolation. e.g. Deriving u_20m from u_10m and u_100m. Options
            are "linear" and "log". See
            :py:meth:`sup3r.preprocessing.derivers.Deriver.do_level_interpolation`
        cache_kwargs : dict | None
            Dictionary with kwargs for caching wrangled data. This should at
            minimum include a `cache_pattern` key, value. This pattern must
            have a {feature} format key and either a h5 or nc file extension,
            based on desired output type. See class:`Cacher` for description
            of more arguments.
        kwargs : dict
            Dictionary of additional keyword args for
            :class:`~sup3r.preprocessing.rasterizers.Rasterizer`, used
            specifically for rasterizing flattened data
        """  # pylint: disable=line-too-long

        features = parse_to_list(features=features)
        source_files = expand_paths(file_paths)
        cached_files, cached_features, _, missing_features = _check_for_cache(
            features=features, cache_kwargs=cache_kwargs
        )

        logger.debug(
            'DataHandler preparing %s requested features from %s source files '
            '(%s cached, %s via rasterizer).',
            len(features),
            len(source_files),
            len(cached_features),
            len(missing_features),
        )

        just_coords = not features
        if just_coords:
            logger.info('Rasterizing source data for coordinate-only access.')
        raster_feats = load_features if missing_features else []
        self.rasterizer = self.loader = self.cache = None
        if cached_features:
            self.cache = Loader(
                file_paths=cached_files,
                features=load_features,
                chunks=chunks,
                feature_aliases=feature_aliases,
                BaseLoader=BaseLoader,
            )
            self.rasterizer = self.loader = self.cache
        if any(missing_features) or just_coords:
            logger.debug('%s not found in cache', missing_features)
            self.rasterizer = Rasterizer(
                file_paths=file_paths,
                res_kwargs=res_kwargs,
                features=raster_feats,
                chunks=chunks,
                target=target,
                shape=shape,
                time_slice=time_slice,
                threshold=threshold,
                feature_aliases=feature_aliases,
                BaseLoader=BaseLoader,
                **kwargs,
            )
            self.loader = self.rasterizer.loader

        self.time_slice = self.rasterizer.time_slice
        self.lat_lon = self.rasterizer.lat_lon
        self._rasterizer_hook()
        self.data = self.rasterizer

        if missing_features:
            super().__init__(
                data=self.data,
                features=features,
                time_roll=time_roll,
                time_shift=time_shift,
                hr_spatial_coarsen=hr_spatial_coarsen,
                nan_method_kwargs=nan_method_kwargs,
                FeatureRegistry=FeatureRegistry,
                interp_kwargs=interp_kwargs,
            )

        if self.cache is not None:
            self.data[cached_features] = self.cache.data[cached_features]
            self.rasterizer.file_paths = (
                expand_paths(file_paths) + cached_files
            )

        should_cache = cache_kwargs is not None and (
            bool(missing_features) or cache_kwargs.get('overwrite', False)
        )
        if should_cache and 'cache_pattern' in cache_kwargs:
            self.cacher = Cacher(data=self.data, cache_kwargs=cache_kwargs)
        self._deriver_hook()

    def _rasterizer_hook(self):
        """Hook in after rasterizer initialization. Implement this to
        extend class functionality with operations after default rasterizer
        initialization. e.g. If special methods are required to add more
        data to the rasterized data or to perform some pre-processing
        before derivations.

        Examples
        --------
         - adding a special method to extract / regrid clearsky_ghi from an
         nsrdb source file prior to derivation of clearsky_ratio.
         - apply bias correction to rasterized data before deriving new
         features
        """

    def _deriver_hook(self):
        """Hook in after deriver initialization. Implement this to extend
        class functionality with operations after default deriver
        initialization. e.g. If special methods are required to derive
        additional features which might depend on non-standard inputs (e.g.
        other source files than those used by the loader)."""


class DailyDataHandler(DataHandler):
    """General data handler class with daily data as an additional attribute.
    xr.Dataset coarsen method employed to compute averages / mins / maxes over
    daily windows. Special treatment of clearsky_ratio, which requires
    derivation from total clearsky_ghi and total ghi.

    TODO:
    (1) Not a fan of manually adding cs_ghi / ghi and then removing. Maybe
    this could be handled through a derivation instead

    (2) We assume daily and hourly data here but we could generalize this to
    go from daily -> any time step. This would then enable the CC models to do
    arbitrary temporal enhancement.
    """

    def __init__(self, file_paths, features, **kwargs):
        """Add features required for daily cs ratio derivation if not
        requested."""

        features = parse_to_list(features=features)
        self.requested_features = features.copy()
        if 'clearsky_ratio' in features:
            needed = [
                f
                for f in self.FEATURE_REGISTRY['clearsky_ratio'].inputs
                if f not in features
            ]
            features.extend(needed)
        super().__init__(file_paths=file_paths, features=features, **kwargs)

    _signature_objs = (__init__, DataHandler)

    def _deriver_hook(self):
        """Hook to run daily coarsening calculations after derivations of
        hourly variables. Replaces data with daily averages / maxes / mins
        / sums"""
        msg = (
            'Data needs to be hourly with at least 24 hours, but data '
            'shape is {}.'.format(self.data.shape)
        )

        day_steps = int(24 * 3600 / self.time_step)
        assert len(self.time_index) % day_steps == 0, msg
        assert len(self.time_index) > day_steps, msg

        n_data_days = int(len(self.time_index) / day_steps)

        logger.info(
            'Calculating daily average datasets for %s training data days.',
            n_data_days,
        )
        daily_data = self.data.coarsen(time=day_steps).mean()
        feats = [f for f in self.features if 'clearsky_ratio' not in f]
        feats = (
            feats
            if 'clearsky_ratio' not in self.features
            else [*feats, 'total_clearsky_ghi', 'total_ghi']
        )
        for fname in feats:
            if '_max_' in fname:
                daily_data[fname] = (
                    self.data[fname].coarsen(time=day_steps).max()
                )
            if '_min_' in fname:
                daily_data[fname] = (
                    self.data[fname].coarsen(time=day_steps).min()
                )
            if 'total_' in fname:
                daily_data[fname] = (
                    self.data[fname.split('total_')[-1]]
                    .coarsen(time=day_steps)
                    .sum()
                )

        if 'clearsky_ratio' in self.features:
            daily_data['clearsky_ratio'] = (
                daily_data['total_ghi'] / daily_data['total_clearsky_ghi']
            )

        logger.info(
            'Finished calculating daily average datasets for %s training '
            'data days.',
            n_data_days,
        )
        hourly_data = self.data[self.requested_features]
        daily_data = daily_data[self.requested_features]
        hourly_data.attrs.update({'name': 'hourly'})
        daily_data.attrs.update({'name': 'daily'})
        self.data = Sup3rDataset(daily=daily_data, hourly=hourly_data)


class DataHandlerH5SolarCC(DailyDataHandler):
    """Extended ``DailyDataHandler`` specifically for handling H5 data for
    SolarCC applications"""

    BASE_LOADER = MultiFileNSRDBX
    FEATURE_REGISTRY = RegistryH5SolarCC


class DataHandlerH5WindCC(DailyDataHandler):
    """Extended ``DailyDataHandler`` specifically for handling H5 data for
    WindCC applications"""

    BASE_LOADER = MultiFileWindX
    FEATURE_REGISTRY = RegistryH5WindCC
