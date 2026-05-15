"""Basic objects that can perform derivations of new features from loaded /
rasterized features."""

import logging
import re
from inspect import signature
from typing import Union
from warnings import warn

import dask.array as da
import numpy as np
import xarray as xr

from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import (
    _rechunk_if_dask,
    parse_to_list,
)
from sup3r.utilities.interpolation import Interpolator

from .methods import DerivedFeature, RegistryBase
from .utilities import parse_feature

logger = logging.getLogger(__name__)


class BaseDeriver(Container):
    """Container subclass with additional methods for transforming / deriving
    data exposed through an :class:`Rasterizer` object."""

    FEATURE_REGISTRY = RegistryBase

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        features,
        FeatureRegistry=None,
        interp_kwargs=None,
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Data to use for derivations. Usually comes from the `.data`
            attribute of a :class:`Rasterizer` object.
        features : list
            List of feature names to derive from the :class:`Rasterizer` data.
            The :class:`Rasterizer` object contains the features available to
            use in the derivation. e.g. rasterizer.features = ['windspeed',
            'winddirection'] with self.features = ['U', 'V']
        FeatureRegistry : Dict
            Optional FeatureRegistry dictionary to use for derivation method
            lookups. When the :class:`Deriver` is asked to derive a feature
            that is not found in the :class:`Rasterizer` data it will look for
            a method to derive the feature in the registry.
        interp_kwargs : dict | None
            Dictionary of kwargs for level interpolation. Can include "method"
            and "run_level_check" keys. Method specifies how to perform height
            interpolation. e.g. Deriving u_20m from u_10m and u_100m. Options
            are "linear" and "log". See
            :py:meth:`sup3r.preprocessing.derivers.Deriver.do_level_interpolation`
        """  # pylint: disable=line-too-long
        if FeatureRegistry is not None:
            self.FEATURE_REGISTRY = FeatureRegistry

        super().__init__(data=data)
        self.interp_kwargs = interp_kwargs
        features = parse_to_list(data=data, features=features)
        new_features = [f for f in features if f not in self.data]
        # Persist derived dependencies back onto self.data so downstream
        # derivations can reuse them instead of recomputing the same feature.
        for f in new_features:
            self.data[f] = self.derive(f)
            logger.info('Finished deriving %s.', f)
        self.data = (
            self.data[list(self.data.coords)]
            if not features
            else self.data
            if features == 'all'
            else self.data[features]
        )

    def _check_registry(self, feature) -> Union[type[DerivedFeature], None]:
        """Check if feature or matching pattern is in the feature registry
        keys. Return the corresponding value if found."""
        if feature.lower() in self.FEATURE_REGISTRY:
            return self.FEATURE_REGISTRY[feature.lower()]
        for pattern, method in self.FEATURE_REGISTRY.items():
            if re.match(pattern.lower(), feature.lower()):
                return method
        return None

    def _get_inputs(self, feature, method=None):
        """Get method inputs and map any wildcards to height or pressure
        (depending on the name of "feature")"""
        method = method or self._check_registry(feature)
        fstruct = parse_feature(feature)
        return [fstruct.map_wildcard(i) for i in getattr(method, 'inputs', [])]

    def get_inputs(self, feature):
        """Get inputs for the given feature and inputs for those inputs."""
        inputs = self._get_inputs(feature)
        more_inputs = []
        for inp in inputs:
            more_inputs.extend(self._get_inputs(inp))
        return inputs + more_inputs

    def no_overlap(self, feature):
        """Check if any of the nested inputs for 'feature' contain 'feature'"""
        return feature not in self.get_inputs(feature)

    def check_registry(
        self, feature
    ) -> Union[np.ndarray, da.core.Array, str, None]:
        """Get compute method from the registry if available. Will check for
        pattern feature match in feature registry. e.g. if u_100m matches a
        feature registry entry of u_(.*)m
        """
        method = self._check_registry(feature)
        if isinstance(method, str):
            return method
        if hasattr(method, 'inputs'):
            inputs = self._get_inputs(feature, method)
            missing = [f for f in inputs if f not in self.data]
            # Recursive derivation is only safe when the dependency chain does
            # not loop back to the requested feature, unless interpolation can
            # break that cycle from nearby levels.
            can_derive = all(
                (self.no_overlap(m) or self.has_interp_variables(m))
                for m in missing
            )
            logger.debug('Found compute method (%s) for %s.', method, feature)
            msg = 'Missing required features %s. Trying to derive these first.'
            if any(missing) and can_derive:
                logger.debug(msg, missing)
                for f in missing:
                    self.data[f] = self.derive(f)
            msg = 'All required features %s found. Proceeding.'
            if not missing or all(f in self.data for f in missing):
                logger.debug(msg, inputs)
                out = self._run_compute(feature, method)
                out.attrs.update(self.collect_input_attrs(feature, inputs))
                return out
            msg = (
                'Some of the method inputs reference %s itself. We will '
                'try height interpolation instead.'
            )
            if not can_derive:
                logger.debug(msg, feature)
        return None

    def _run_compute(self, feature, method):
        """If we have all the inputs we can run the compute method."""
        compute = method.compute
        params = signature(compute).parameters
        fstruct = parse_feature(feature)
        kwargs = {
            k: getattr(fstruct, k) for k in params if hasattr(fstruct, k)
        }
        return compute(self.data, **kwargs)

    def map_new_name(self, feature, pattern):
        """If the search for a derivation method first finds an alternative
        name for the feature we want to derive, by matching a wildcard pattern,
        we need to replace the wildcard with the specific height or pressure we
        want and continue the search for a derivation method with this new
        name."""
        fstruct = parse_feature(feature)
        pstruct = parse_feature(pattern)
        # Registry aliases can point from one naming convention to another,
        # including wildcard patterns like basename_*m.
        if '*' not in pattern:
            new_feature = pattern
        elif fstruct.height is not None:
            new_feature = pstruct.basename + f'_{fstruct.height}m'
        elif fstruct.pressure is not None:
            new_feature = pstruct.basename + f'_{fstruct.pressure}pa'
        else:
            msg = (
                'Found matching pattern "%s" for feature "%s" but could not '
                'construct a valid new feature name'
            )
            logger.error(msg, pattern, feature)
            raise RuntimeError(msg)
        logger.debug(
            'Found alternative name "%s" for "%s". Continuing derivation '
            'for %s.',
            feature,
            new_feature,
            new_feature,
        )
        return new_feature

    def has_interp_variables(self, feature):
        """Check if the given feature can be interpolated from values at nearby
        heights or from pressure level data. e.g. If ``u_10m`` and ``u_50m``
        exist then ``u_30m`` can be interpolated from these. If a pressure
        level array ``u`` is available this can also be used, in conjunction
        with height data."""
        fstruct = parse_feature(feature)
        count = 0
        for feat in self.data.features:
            fstruct_check = parse_feature(feat)
            height = fstruct_check.height

            if (
                fstruct_check.basename == fstruct.basename
                and height is not None
            ):
                count += 1
        return count > 1 or fstruct.basename in self.data

    def derive(
        self, feature, strict=True
    ) -> Union[np.ndarray, da.core.Array, None]:
        """Resolve a feature from contained data or available derivations.

        Parameters
        ----------
        feature : str
            Feature to resolve from the contained data or available compute
            methods.
        strict : bool
            Whether to raise if the feature cannot be resolved. If ``False``,
            return ``None`` instead.
        """
        if feature not in self.data:
            compute_check = self.check_registry(feature)
            if compute_check is not None and isinstance(compute_check, str):
                new_feature = self.map_new_name(feature, compute_check)
                return self.derive(new_feature, strict=strict)

            if compute_check is not None:
                return compute_check

            # If no direct compute method exists, fall back to vertical
            # interpolation from nearby heights or pressure levels.
            if self.has_interp_variables(feature):
                logger.debug(
                    'Attempting level interpolation for "%s"', feature
                )
                return self.do_level_interpolation(
                    feature, interp_kwargs=self.interp_kwargs
                )

            if not strict:
                return None

            msg = (
                'Could not find "%s" in contained data or in the available '
                'compute methods.'
            )
            raise RuntimeError(msg % feature)

        if np.isnan(self.data[feature]).any():
            msg = (
                f'Feature "{feature}" contains NaN values. Either use '
                '`nan_method_kwargs` to handle NaN values or clean the data.'
            )
            logger.warning(msg)
            warn(msg)
        return self.data[feature]

    def get_single_level_data(self, feature):
        """When doing level interpolation we should include the single level
        data available. e.g. If we have u_100m already and want to interpolate
        u_40m from multi-level data U we should add u_100m at height 100m
        before doing interpolation, since 100 could be a closer level to 40m
        than those available in U."""
        fstruct = parse_feature(feature)
        pattern = fstruct.basename + '_(.*)'
        var_list = []
        lev_list = []
        lev_array = None
        var_array = None
        for f in list(self.data.data_vars):
            if re.match(pattern.lower(), f):
                var_list.append(self.data[f])
                pstruct = parse_feature(f)
                lev = (
                    pstruct.height
                    if pstruct.height is not None
                    else pstruct.pressure
                )
                lev_list.append(np.float32(lev))

        if len(var_list) > 0:
            var_array = da.stack(var_list, axis=-1)
            # Broadcast the scalar levels to match the stacked data shape so
            # interpolation can treat data values and their levels uniformly.
            sl_shape = (*var_array.shape[:-1], len(lev_list))
            lev_array = da.broadcast_to(da.from_array(lev_list), sl_shape)

        return var_array, lev_array

    def get_multi_level_data(self, feature):
        """Get data stored in multi-level arrays, like u stored on pressure
        levels."""
        fstruct = parse_feature(feature)
        var_array = None
        lev_array = None

        if fstruct.basename in self.data:
            var_array = self.data[fstruct.basename].data.astype(np.float32)

        if fstruct.height is not None and var_array is not None:
            msg = (
                f'To interpolate {fstruct.basename} to {feature} the loaded '
                'data needs to include "zg" and "topography" or have a '
                f'"{Dimension.HEIGHT}" dimension.'
            )
            can_calc_height = (
                'zg' in self.data.features
                and 'topography' in self.data.features
            )
            have_height = Dimension.HEIGHT in self.data.dims
            assert can_calc_height or have_height, msg

            if can_calc_height:
                lev_array = self.data[['zg', 'topography']].as_array()
                lev_array = lev_array[..., 0] - lev_array[..., 1]
            else:
                lev_array = da.broadcast_to(
                    self.data[Dimension.HEIGHT].astype(np.float32),
                    var_array.shape,
                )
        elif var_array is not None:
            msg = (
                f'To interpolate {fstruct.basename} to {feature} the loaded '
                'data needs to include "level" (a.k.a pressure at multiple '
                'levels).'
            )
            assert Dimension.PRESSURE_LEVEL in self.data, msg
            lev_array = da.broadcast_to(
                self.data[Dimension.PRESSURE_LEVEL].astype(np.float32),
                var_array.shape,
            )
        return var_array, lev_array

    def collect_input_attrs(self, feature, inputs=None):
        """Collect attributes from the input features for the given feature."""
        fstruct = parse_feature(feature)
        attrs = {}
        if inputs is None:
            inputs = [
                feat
                for feat in self.data.features
                if parse_feature(feat).basename == fstruct.basename
            ]
        for feat in inputs:
            fattrs = self.data[feat].attrs
            if fattrs is not None:
                for k, v in fattrs.items():
                    attrs.setdefault(k, [])
                    if isinstance(v, (list, np.ndarray)):
                        attrs[k].extend(v)
                    else:
                        attrs[k].append(v)
        for k, v in attrs.items():
            if len(set(v)) == 1:
                attrs[k] = v[0]
        attrs.setdefault('inputs', inputs)
        return attrs

    def do_level_interpolation(
        self, feature, interp_kwargs=None
    ) -> xr.DataArray:
        """Interpolate over height or pressure to derive the given feature."""
        ml_var, ml_levs = self.get_multi_level_data(feature)
        sl_var, sl_levs = self.get_single_level_data(feature)

        fstruct = parse_feature(feature)
        attrs = self.collect_input_attrs(feature)
        level = (
            fstruct.height if fstruct.height is not None else fstruct.pressure
        )

        if ml_var is not None and sl_var is None:
            var_array = ml_var
            lev_array = ml_levs
        elif sl_var is not None and ml_var is None:
            var_array = sl_var
            lev_array = sl_levs
        elif ml_var is not None and sl_var is not None:
            # Prefer using every available level by combining explicit
            # single-level fields with multi-level arrays before interpolation.
            var_array = np.concatenate([ml_var, sl_var], axis=-1)
            lev_array = np.concatenate([ml_levs, sl_levs], axis=-1)
        else:
            msg = 'Neither single level nor multi level data was found for %s'
            logger.error(msg, feature)
            raise RuntimeError(msg % feature)

        if np.isnan(lev_array).any():
            msg = (
                'NaN values found in the interpolation level array for '
                f'{feature}.'
            )
            logger.warning(msg)
            warn(msg)
        if np.isnan(var_array).any():
            msg = (
                'NaN values found in the interpolation variable array for '
                f'{feature}.'
            )
            logger.warning(msg)
            warn(msg)

        out = Interpolator.interp_to_level(
            lev_array=lev_array,
            var_array=var_array,
            level=np.float32(level),
            interp_kwargs=interp_kwargs,
        )

        assert not np.isnan(out).any(), (
            f'NaN values found in the interpolated output for {feature}. '
            'Make sure there are no NaN values in the level or variable data.'
        )

        return xr.DataArray(
            data=_rechunk_if_dask(out),
            dims=Dimension.dims_3d()[: len(out.shape)],
            attrs=attrs,
        )


class Deriver(BaseDeriver):
    """Extends base :class:`BaseDeriver` class with time_roll and
    hr_spatial_coarsen args."""

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        features,
        time_roll=0,
        time_shift=None,
        hr_spatial_coarsen=1,
        nan_method_kwargs=None,
        FeatureRegistry=None,
        interp_kwargs=None,
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset]
            Data used for derivations
        features: list
            List of features to derive
        time_roll: int
            Number of steps to roll along the time axis. `Passed to
            xr.Dataset.roll()`
        time_shift: int | None
            Number of minutes to shift time axis. This can be used, for
            example, to shift the time index for daily data so that the time
            stamp for a given day starts at the zeroth minute instead of at
            noon, as is the case for most GCM data.
        hr_spatial_coarsen: int
            Spatial coarsening factor. Passed to `xr.Dataset.coarsen()`
        nan_method_kwargs: str | dict | None
            Keyword arguments for nan handling. If 'mask', time steps with nans
            will be dropped. Otherwise this should be a dict of kwargs which
            will be passed to :meth:`Sup3rX.interpolate_na`.
        FeatureRegistry : dict
            Dictionary of :class:`DerivedFeature` objects used for derivations
        interp_kwargs : dict | None
            Dictionary of kwargs for level interpolation. Can include "method"
            and "run_level_check" keys. Method specifies how to perform height
            interpolation. e.g. Deriving u_20m from u_10m and u_100m. Options
            are "linear" and "log". See
            :py:meth:`sup3r.preprocessing.derivers.Deriver.do_level_interpolation`
        """  # pylint: disable=line-too-long

        super().__init__(
            data=data,
            features=features,
            FeatureRegistry=FeatureRegistry,
            interp_kwargs=interp_kwargs,
        )

        if time_roll != 0:
            logger.debug('Applying time_roll=%s to data array', time_roll)
            self.data = self.data.roll(**{Dimension.TIME: time_roll})

        if time_shift is not None:
            logger.debug('Applying time_shift=%s to time index', time_shift)
            self.data.time_index = self.data.time_index.shift(
                time_shift, freq='min'
            )

        if hr_spatial_coarsen > 1:
            logger.debug(
                'Applying hr_spatial_coarsen=%s to data.', hr_spatial_coarsen
            )
            self.data = self.data.coarsen(
                {
                    Dimension.SOUTH_NORTH: hr_spatial_coarsen,
                    Dimension.WEST_EAST: hr_spatial_coarsen,
                },
                boundary='trim',
            ).mean()

        if nan_method_kwargs is not None:
            if nan_method_kwargs['method'] == 'mask':
                dim = nan_method_kwargs.get('dim', Dimension.TIME)
                arr = self.data.to_dataarray()
                dims = set(arr.dims) - {dim}
                mask = np.isnan(arr).any(dims).data
                self.data = self.data.drop_isel(**{dim: mask})

            elif np.isnan(self.data.as_array()).any():
                logger.info(
                    'Filling nan values with nan_method_kwargs=%s.',
                    nan_method_kwargs,
                )
                self.data = self.data.interpolate_na(**nan_method_kwargs)
