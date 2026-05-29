"""Basic ``Sampler`` objects. These are containers which also can sample from
the underlying data. These interface with ``BatchQueues`` so they also have
additional information about how different features are used by models."""

import logging
from fnmatch import fnmatch
from functools import cached_property
from typing import Optional
from warnings import warn

import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.samplers.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
)
from sup3r.preprocessing.utilities import compute_if_dask, lowered
from sup3r.utilities.utilities import (
    OUTPUT_ATTRS,
    RANDOM_GENERATOR,
    get_feature_basename,
)

logger = logging.getLogger(__name__)


class Sampler(Container):
    """Basic Sampler class for iterating through batches of samples from the
    contained data."""

    def __init__(
        self,
        data,
        sample_shape: Optional[tuple] = None,
        batch_size: int = 16,
        feature_sets: Optional[dict] = None,
        proxy_obs_kwargs: Optional[dict] = None,
        mode: str = 'lazy',
    ):
        """
        Parameters
        ----------
        data: Union[Sup3rX, Sup3rDataset],
            Object with data that will be sampled from. Usually the ``.data``
            attribute of various :class:`~sup3r.preprocessing.base.Container`
            objects.  i.e. :class:`~sup3r.preprocessing.loaders.Loader`,
            :class:`~sup3r.preprocessing.rasterizers.Rasterizer`,
            :class:`~sup3r.preprocessing.derivers.Deriver`, as long as the
            spatial dimensions are not flattened.
        sample_shape : tuple
            Size of arrays to sample from the contained data.
        batch_size : int
            Number of samples to get to build a single batch. A sample of
            ``(sample_shape[0], sample_shape[1], batch_size *
            sample_shape[2])`` is first selected from underlying dataset and
            then reshaped into ``(batch_size, *sample_shape)`` to get a single
            batch. This is more efficient than getting ``N = batch_size``
            samples and then stacking.
        feature_sets : Optional[dict]
            Optional dictionary describing how the full set of features is
            split between ``lr_features``, ``hr_exo_features``, and
            ``hr_out_features``.

            lr_features : list | tuple
                List of feature names or patt*erns to use as low-resolution
                model inputs. If no entry is provided then all available
                features from the data will be used.
            hr_out_features : list | tuple
                List of feature names or patt*erns that should be output
                by the generative model and available as ground truth targets.
                If no entry is provided then all features in lr_features will
                be used.
            hr_exo_features : list | tuple
                List of feature names or patt*erns that should be available as
                high-resolution model inputs (like topography or observations)
                or for bespoke loss functions. Features used as inputs are
                injected into the model mid-network to condition output on
                high-resolution information. The model configuration should
                have the appropriate layers to use these features. e.g.
                ``Sup3rConcat`` for topography injection, ``Sup3rObsModel`` or
                ``Sup3rCrossAttention`` for obs injection.  If no entry is
                provided then hr_exo_features will be empty.
            *To include sparse features as inputs or targets the features
            must have an "_obs" suffix.
        proxy_obs_kwargs : dict | None
            Optional dictionary of keyword arguments to pass to the proxy
            observation generator. This is only used when training with proxy
            observations. Top-level keys (``onshore_obs_frac``,
            ``offshore_obs_frac``, ``perturbation_scale``) apply to all obs
            features as defaults. A source-feature-named sub-dict (keyed by
            the gridded feature name, e.g. ``u_100m`` for ``u_100m_obs``)
            overrides any of those keys for that specific feature::

                proxy_obs_kwargs = {
                    'onshore_obs_frac': {'spatial': [0.3, 0.7], 'temporal': 1},
                    'perturbation_scale': 0.01,
                    'u_100m': {
                        'onshore_obs_frac': {'spatial': 0.9},
                        'perturbation_scale': 0.05,
                    },
                }

            perturbation_scale : float
                If non-zero, gaussian noise scaled by this value times the
                per-feature batch standard deviation is added to proxy obs.
            onshore_obs_frac : dict
                Fraction of onshore observations per batch. Keys are
                'spatial' and 'temporal'. Each value is a float (fixed
                fraction) or a [min, max] list to sample uniformly per batch.
            offshore_obs_frac : dict
                Same as ``onshore_obs_frac`` but applied where topography
                <= 0. Ignored when topography is not a source feature.
        mode : str
            Mode for sampling data. Options are 'lazy' or 'eager'. 'eager' mode
            pre-loads all data into memory as numpy arrays for faster access.
            'lazy' mode samples directly from the underlying data object, which
            could be backed by dask arrays or on-disk netCDF files.
        """
        super().__init__(data=data)
        feature_sets = feature_sets or {}
        self._lr_features = feature_sets.get('lr_features', self.data.features)
        self._hr_exo_features = feature_sets.get('hr_exo_features', [])
        self._hr_out_features = feature_sets.get('hr_out_features', [])
        self.proxy_obs_kwargs = proxy_obs_kwargs or {}
        self.mode = mode
        self.sample_shape = sample_shape or (10, 10, 1)
        self.batch_size = batch_size
        self.preflight()
        self.check_feature_consistency()

    @property
    def use_proxy_obs(self):
        """Whether to use proxy observations. When True, proxy observation
        features are generated by masking the corresponding gridded ground
        truth data and are appended to the samples. The obs features are
        specified by the ``obs_features`` argument and should have a
        corresponding source feature in the data features that is used for
        sampling. For example, an obs feature named ``temperature_obs`` would
        be generated from the gridded ground truth feature named
        ``temperature``.
        """
        return bool(self.proxy_obs_kwargs)

    def get_sample_index(self, n_obs=None):
        """Randomly gets spatiotemporal sample index.

        Returns
        -------
        sample_index : tuple
            Tuple of latitude slice, longitude slice, time slice, and features.
            Used to get single observation like ``self.data[sample_index]``

        Notes
        -----
        If ``n_obs > 1`` this will get a time slice with ``n_obs *
        self.sample_shape[2]`` time steps, which will then be reshaped into
        ``n_obs`` samples each with ``self.sample_shape[2]`` time steps. This
        is a much more efficient way of getting batches of samples but only
        works if there are enough continuous time steps to sample.
        """
        n_obs = n_obs or self.batch_size
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        time_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2] * n_obs
        )
        return (*spatial_slice, time_slice, self.hr_sample_features)

    def preflight(self):
        """Perform shape and feature checks."""
        good_shape = (
            self.sample_shape[0] <= self.data.shape[0]
            and self.sample_shape[1] <= self.data.shape[1]
        )
        msg = (
            f'spatial_sample_shape {self.sample_shape[:2]} is '
            f'larger than the raster size {self.data.shape[:2]}'
        )
        assert good_shape, msg

        msg = (
            f'sample_shape[2] ({self.sample_shape[2]}) cannot be larger '
            'than the number of time steps in the raw data '
            f'({self.data.shape[2]}).'
        )

        assert self.data.shape[2] >= self.sample_shape[2], msg

        msg = (
            f'sample_shape[2] * batch_size ({self.sample_shape[2]} * '
            f'{self.batch_size}) is larger than the number of time steps in '
            f'the raw data ({self.data.shape[2]}). This prevents us from '
            'building batches with a single sample with n_time_steps = '
            'sample_shape[2] * batch_size, which is far more performant than '
            'building batches with n_samples = batch_size, each with '
            'n_time_steps = sample_shape[2].'
        )
        if (
            self.data.shape[2] < self.sample_shape[2] * self.batch_size
            and self.data.shape[2] > 1
        ):
            logger.warning(msg)
            warn(msg)

        if self.mode == 'eager':
            logger.debug('Received mode = "eager".')
            _ = self.compute()

    def check_proxy_obs_consistency(self):
        """Check that the obs features are configured correctly for proxy
        observations."""
        all_feats = lowered(set(self.data.features))

        assert set(self.obs_features).issubset(set(self.hr_source_features)), (
            'When using proxy observations, all obs features must be '
            'included either in hr_out_features or hr_exo_features.'
        )
        assert not set(all_feats).intersection(self.obs_features), (
            f'Obs features {self.obs_features} cannot be in the data '
            f'features {self.data.features} when using proxy '
            'observations.'
        )
        feats = set(self.hr_exo_features) - set(self.obs_features)
        assert feats.issubset(all_feats), (
            f'All non-obs hr_exo_features {feats} must be in the data '
            f'features {self.data.features} when using proxy observations.'
        )
        base_feats = [f.replace('_obs', '') for f in self.obs_features]
        assert set(base_feats).issubset(all_feats), (
            f'All obs features {self.obs_features} must have a '
            'corresponding source feature in the data features '
            f'{self.data.features} when using proxy observations.'
        )
        assert set(base_feats).issubset(self.hr_source_features), (
            f'All obs features {self.obs_features} must have a '
            'corresponding source feature listed in hr_out_features when '
            'using proxy observations.'
        )

    def check_feature_consistency(self):
        """Check that the feature sets are consistent with each other and the
        obs features are configured correctly."""
        all_feats = lowered(set(self.data.features))
        assert set(self.lr_features).issubset(all_feats), (
            f'All lr_features {self.lr_features} must be in the data features '
            f'{self.data.features}.'
        )
        assert set(self.hr_out_features).issubset(all_feats), (
            f'All hr_out_features {self.hr_out_features} must be in the data '
            f'features {self.data.features}.'
        )
        if not self.use_proxy_obs:
            assert set(self.hr_exo_features).issubset(all_feats), (
                f'All hr_exo_features {self.hr_exo_features} must be in the '
                f'data features {self.data.features} when not using proxy '
                'observations.'
            )
        else:
            self.check_proxy_obs_consistency()

        if len(self.obs_features) > 0 and set(self.obs_features).intersection(
            self.hr_exo_features
        ):
            msg = (
                f'Obs features {self.obs_features} must come at the end of '
                f'the hr_exo_features {self.hr_exo_features}'
            )
            assert list(self.obs_features) == list(
                self.hr_exo_features[-len(self.obs_features) :]
            ), msg

        if len(self.hr_exo_features) > 0:
            msg = (
                f'hr_exo_features {self.hr_exo_features} must come at the end '
                f'of the full high-res feature set: {self.hr_source_features}'
            )
            assert list(self.hr_exo_features) == list(
                self.hr_source_features[-len(self.hr_exo_features) :]
            ), msg

    @property
    def sample_shape(self) -> tuple:
        """Shape of the data sample to select when ``__next__()`` is called."""
        return self._sample_shape

    @sample_shape.setter
    def sample_shape(self, sample_shape):
        """Set the shape of the data sample to select when ``__next__()`` is
        called."""
        self._sample_shape = sample_shape
        if len(self._sample_shape) == 2:
            logger.info(
                'Found 2D sample shape of %s. Adding temporal dim of 1',
                self._sample_shape,
            )
            self._sample_shape = (*self._sample_shape, 1)

    @property
    def hr_sample_shape(self) -> tuple:
        """Shape of the data sample to select when `__next__()` is called. Same
        as sample_shape"""
        return self._sample_shape

    @hr_sample_shape.setter
    def hr_sample_shape(self, hr_sample_shape):
        """Set the sample shape to select when `__next__()` is called. Same
        as sample_shape"""
        self._sample_shape = hr_sample_shape

    def _reshape_samples(self, samples):
        """Reshape samples into batch shapes, with shape = (batch_size,
        *sample_shape, n_features). Samples start out with a time dimension of
        shape = batch_size * sample_shape[2] so we need to split this and
        reorder the dimensions.

        Parameters
        ----------
        samples : Union[np.ndarray, da.core.Array]
            Selection from `self.data` with shape:
            (samp_shape[0], samp_shape[1], batch_size * samp_shape[2], n_feats)
            This is reshaped to:
            (batch_size, samp_shape[0], samp_shape[1], samp_shape[2], n_feats)

        Returns
        -------
        batch: np.ndarray
            Reshaped sample array, with shape:
            (batch_size, samp_shape[0], samp_shape[1], samp_shape[2], n_feats)

        """
        new_shape = list(samples.shape)
        new_shape = [
            *new_shape[:2],
            self.batch_size,
            new_shape[2] // self.batch_size,
            new_shape[-1],
        ]
        # (lats, lons, batch_size, times, feats)
        out = np.reshape(samples, new_shape)
        # (batch_size, lats, lons, times, feats)
        return np.transpose(out, axes=(2, 0, 1, 3, 4))

    @classmethod
    def _stack_samples(cls, samples):
        """Used to build batch arrays in the case of independent time samples
        (e.g. slow batching)

        Note
        ----
        Tuples are in the case of dual datasets. e.g. This sampler is for a
        :class:`~sup3r.preprocessing.batch_handlers.DualBatchHandler`

        Parameters
        ----------
        samples : tuple[list[np.ndarray | da.core.Array], ...] |
                  list[np.ndarray | da.core.Array]
            Each list has length = batch_size and each array has shape:
            (samp_shape[0], samp_shape[1], samp_shape[2], n_feats)

        Returns
        -------
        batch: tuple[np.ndarray, np.ndarray] | np.ndarray
            Stacked sample array(s), each with shape:
            (batch_size, samp_shape[0], samp_shape[1], samp_shape[2], n_feats)
        """
        if isinstance(samples[0], tuple):
            lr = np.stack([s[0] for s in samples], axis=0)
            hr = np.stack([s[1] for s in samples], axis=0)
            return (lr, hr)
        return np.stack(samples, axis=0)

    def _compute_samples(self, samples):
        """Cast samples to numpy arrays. This only does something when samples
        are dask arrays.

        Parameters
        ----------
        samples : tuple[np.ndarray | da.core.Array, ...] |
                  np.ndarray | da.core.Array
            Samples retrieved from the underlying data. Could be a tuple
            in the case of dual datasets.
        """
        if self.mode == 'eager':
            return samples
        return compute_if_dask(samples)

    def _fast_batch(self):
        """Get batch of samples with adjacent time slices."""
        idx = self.get_sample_index(n_obs=self.batch_size)
        out = self.data.sample(idx)
        out = self._compute_samples(out)
        if isinstance(out, tuple):
            out = tuple(self._reshape_samples(o) for o in out)
        else:
            out = self._reshape_samples(out)
        return self._append_obs_features(out)

    def _slow_batch(self):
        """Get batch of samples with random time slices."""
        out = [
            self.data.sample(self.get_sample_index(n_obs=1))
            for _ in range(self.batch_size)
        ]
        out = self._compute_samples(out)
        out = self._stack_samples(out)
        return self._append_obs_features(out)

    def _fast_batch_possible(self):
        return self.batch_size * self.sample_shape[2] <= self.data.shape[2]

    def _get_proxy_obs(self, hi_res):
        """Generate proxy observation data by masking the gridded high-res
        data. Optionally adds a perturbation to the proxy observations sampled
        from a gaussian distribution with mean 0 and standard deviation equal
        to the standard deviation of the unmasked values for each feature
        multiplied by perturbation_scale. This is done to prevent the model
        from learning to ignore the obs features because they are exactly the
        same as the gridded features at the observed locations. This can also
        encourage the model to condition on obs that differ significantly from
        the gridded data. Unobserved locations are set to NaN.

        Parameters
        ----------
        hi_res : np.ndarray
            High resolution batch data with shape:
            (batch_size, spatial_1, spatial_2, temporal, n_features)

        Returns
        -------
        obs : np.ndarray
            Observation data with NaN for unobserved locations. Shape:
            (batch_size, spatial_1, spatial_2, temporal, n_obs_features)
        """
        obs_mask = self._get_full_obs_mask(hi_res)
        obs = hi_res[..., self.obs_features_ind].copy()
        stds = np.std(obs, axis=(1, 2, 3), keepdims=True)
        obs[obs_mask[..., : obs.shape[-1]]] = np.nan
        for i, feat in enumerate(self.obs_features):
            scale = self._get_proxy_kwarg('perturbation_scale', feat, 0)
            if scale > 0:
                srange = stds[..., i] * scale
                obs[..., i] += np.random.normal(scale=srange)
                base = get_feature_basename(feat.replace('_obs', ''))
                attrs = OUTPUT_ATTRS.get(base, {})
                lo = attrs.get('min', -np.inf)
                hi = attrs.get('max', np.inf)
                obs[..., i] = np.where(
                    np.isnan(obs[..., i]),
                    obs[..., i],
                    np.clip(obs[..., i], lo, hi),
                )
        return obs

    def _append_obs_features(self, samples):
        """Append proxy observation features to the batch samples when
        ``use_proxy_obs=True``. The obs features are generated by masking
        the corresponding gridded ground truth features.

        Parameters
        ----------
        samples : np.ndarray | tuple[np.ndarray, ...]
            Batch samples from the data source. For single datasets, shape
            is (batch_size, s1, s2, t, n_features). For dual datasets,
            this is a tuple of arrays.

        Returns
        -------
        samples : np.ndarray | tuple[np.ndarray, ...]
            Same as input, but with obs features appended to the last
            dimension if proxy obs are enabled.
        """
        if not self.use_proxy_obs:
            return samples

        if isinstance(samples, tuple):
            # For dual datasets, obs features are appended to the high-res
            # member (last element)
            hr = samples[-1]
            obs = self._get_proxy_obs(hr)
            hr = np.concatenate([hr, obs], axis=-1)
            return (*samples[:-1], hr)

        obs = self._get_proxy_obs(samples)
        return np.concatenate([samples, obs], axis=-1)

    def __next__(self):
        """Get next batch of samples. This retrieves n_samples = batch_size
        with shape = sample_shape from the `.data` (a xr.Dataset or
        Sup3rDataset) through the Sup3rX accessor.

        When ``use_proxy_obs=True`` and ``obs_features`` are configured, proxy
        observation features are generated by masking the corresponding
        gridded ground truth data and are appended to the samples.

        Returns
        -------
        samples : tuple(np.ndarray | da.core.Array) | np.ndarray | da.core.Array
            Either a tuple or single array of samples. This is a tuple when
            this method is sampling from a ``Sup3rDataset`` with two data
            members. When proxy obs are enabled, obs features are appended
            to the feature dimension.
        """  # noqa: E501
        if self._fast_batch_possible():
            return self._fast_batch()
        return self._slow_batch()

    def _parse_features(self, unparsed_feats):
        """Return a list of parsed feature names without wildcards."""
        if isinstance(unparsed_feats, str):
            parsed_feats = [unparsed_feats]
        elif isinstance(unparsed_feats, tuple):
            parsed_feats = list(unparsed_feats)
        elif unparsed_feats is None:
            parsed_feats = []
        else:
            parsed_feats = unparsed_feats

        if any('*' in fn for fn in parsed_feats):
            out = []
            for feature in self.features:
                match = any(
                    fnmatch(feature.lower(), pattern.lower())
                    for pattern in parsed_feats
                )
                if match:
                    out.append(feature)
            parsed_feats = out
        return lowered(parsed_feats)

    @cached_property
    def lr_features(self):
        """List of feature names or patt*erns to use as low-resolution model
        inputs. If no entry is provided then all available features from the
        data will be used."""
        return self._parse_features(self._lr_features)

    @cached_property
    def hr_source_features(self):
        """List of feature names or patt*erns that should be available natively
        as high-resolution.  For a non-dual sampler this is all features, since
        even features only provided to the model as low-resolution still need
        to be coarsened from the high-resolution data. This is in contrast to
        dual samplers
        (:class:`~sup3r.preprocessing.samplers.dual.DualSampler`), where there
        are separate high-resolution and low-resolution data members."""
        feats = [f for f in self.lr_features if f not in self.hr_exo_features]
        feats += [
            f
            for f in self.hr_out_features
            if f not in feats and f not in self.hr_exo_features
        ]
        feats += [f for f in self.hr_exo_features if f not in feats]
        return feats

    @cached_property
    def hr_features(self):
        """List of feature names or patt*erns that the model is shown at
        high-resolution. This does not include features that are only shown to
        the model after coarsening. Thus, this includes hr_out_features and
        and hr_exo_features but not lr_features."""
        out = [
            f for f in self.hr_out_features if f not in self.hr_exo_features
        ]
        out += self.hr_exo_features
        return out

    @cached_property
    def hr_sample_features(self):
        """List of feature names used in the sample index for the
        high-resolution training data."""
        return (
            [f for f in self.hr_source_features if f not in self.obs_features]
            if self.use_proxy_obs
            else self.hr_source_features
        )

    @cached_property
    def hr_out_features(self):
        """List of feature names or patt*erns that should be output by the
        generative model. If no entry is provided then all features in
        hr_features will be used."""
        hr_out = self._parse_features(self._hr_out_features)
        return self.lr_features if len(hr_out) == 0 else hr_out

    @cached_property
    def hr_exo_features(self):
        """Get a list of exogenous high-resolution features that are only used
        for training e.g., mid-network high-res topo injection. These must come
        at the end of the high-res feature set. These can also be input to the
        model as low-res features."""
        return self._parse_features(self._hr_exo_features)

    @cached_property
    def obs_features(self):
        """List of feature names or patt*erns that should be treated as
        observations. These features will be included in the high-res data but
        not the low-res data and won't necessarily be expected to be output by
        the generative model. These are different from other `hr_exo_features`
        in that they are intended to be used as observation features with NaN
        values where observations are not available."""
        return [f for f in self.hr_source_features if '_obs' in f]

    @cached_property
    def hr_features_ind(self):
        """Get the high-resolution feature channel indices that should be
        included for loss calculations. This includes hr_out_features and
        hr_exo_features, Any high-resolution features that are only included in
        the data handler to be coarsened for the low-res input are removed.
        """
        return [self.hr_source_features.index(f) for f in self.hr_features]

    @cached_property
    def lr_features_ind(self):
        """Get the low-resolution feature channel indices that should be
        included for training. This includes lr_features.
        """
        return [self.hr_source_features.index(f) for f in self.lr_features]

    @cached_property
    def obs_features_ind(self):
        """Get the source feature indices in ``features`` for each obs
        feature. Each obs feature named ``<feature>_obs`` maps to the
        corresponding ``<feature>`` in the features.

        Returns
        -------
        list[int]
            Indices into ``features`` for each obs feature source.
        """
        check_feats = (
            [f.replace('_obs', '') for f in self.obs_features]
            if self.use_proxy_obs
            else self.obs_features
        )
        return [self.hr_source_features.index(f) for f in check_feats]

    def _get_proxy_kwarg(self, key, feat, default):
        """Get a proxy obs kwarg value for a specific obs feature, with
        fallback to the global default in ``proxy_obs_kwargs``.

        Parameters
        ----------
        key : str
            The kwarg name, e.g. ``'onshore_obs_frac'`` or
            ``'perturbation_scale'``.
        feat : str
            The obs feature name (e.g. ``'u_100m_obs'``). The ``'_obs'``
            suffix is stripped to look up the source-feature override key.
        default :
            Value returned when neither a feature-level nor a global entry
            exists in ``proxy_obs_kwargs``.
        """
        src = feat.replace('_obs', '')
        feat_overrides = self.proxy_obs_kwargs.get(src, {})
        if key in feat_overrides:
            return feat_overrides[key]
        return self.proxy_obs_kwargs.get(key, default)

    def _get_obs_mask(self, hi_res, spatial_frac, time_frac=1.0):
        """Get observation mask for a given spatial and time obs fraction for
        an entire batch. This is divided between spatial and time fractions
        because often the spatial fraction is significantly lower than the time
        fraction in practice, e.g. for a given spatial location there might be
        observations for most of the time period but only a small fraction of
        the spatial domain is observed.

        Parameters
        ----------
        hi_res : np.ndarray
            True high resolution data for the entire batch.
        spatial_frac : float | list
            Fraction of the spatial domain that should be treated as
            observations. This is a value between 0 and 1 or a list with lower
            and upper bounds for the spatial fraction.
        time_frac : float | list, optional
            Fraction of the time domain that should be treated as observations.
            This is a value between 0 and 1 or a list with lower and upper
            bounds for the time fraction. Default is 1.0

        Returns
        -------
        np.ndarray
            Mask which is True for locations that are not observed and False
            for locations that are observed. Shape:
            (n_obs, spatial_1, spatial_2, n_temporal, 1)

        Notes
        -----
        The output mask has a trailing singleton feature dimension. Callers
        are responsible for repeating or concatenating across features.
        Each sample in the batch has an independent mask.
        """
        s_range = (
            spatial_frac
            if isinstance(spatial_frac, (list, tuple))
            else [spatial_frac, spatial_frac]
        )
        t_range = (
            time_frac
            if isinstance(time_frac, (list, tuple))
            else [time_frac, time_frac]
        )
        n_obs, n_spatial_1, n_spatial_2, n_temporal = hi_res.shape[:-1]

        s_fracs = RANDOM_GENERATOR.uniform(*s_range, size=n_obs)
        t_fracs = RANDOM_GENERATOR.uniform(*t_range, size=n_obs)
        s_fracs = np.clip(s_fracs, 0, 1)
        t_fracs = np.clip(t_fracs, 0, 1)

        s_mask = RANDOM_GENERATOR.uniform(
            size=(n_obs, n_spatial_1, n_spatial_2)
        )
        s_mask = s_mask <= s_fracs[:, None, None]
        s_mask = s_mask[..., None, None]

        t_mask = RANDOM_GENERATOR.uniform(size=(n_obs, n_temporal))
        t_mask = t_mask <= t_fracs[:, None]
        t_mask = t_mask[:, None, None, :, None]

        mask = ~(s_mask & t_mask)
        return mask

    def _get_topo(self, hi_res):
        """Return the topography slice from ``hi_res``, or ``None`` if
        topography is not in the source features."""
        if 'topography' not in self.hr_source_features:
            return None
        topo_idx = self.hr_source_features.index('topography')
        return hi_res[..., topo_idx]

    def _get_feat_obs_mask(self, hi_res, feat, topo):
        """Build the observation mask for a single obs feature, applying the
        offshore mask where topography is non-positive when ``topo`` is
        provided.

        Parameters
        ----------
        hi_res : np.ndarray
            High-resolution batch data.
        feat : str
            Obs feature name (e.g. ``'u_100m_obs'``).
        topo : np.ndarray | None
            Topography array with shape ``(n_obs, s1, s2, n_temporal)``, or
            ``None`` when topography is unavailable.

        Returns
        -------
        np.ndarray
            Boolean mask with shape ``(n_obs, s1, s2, n_temporal, 1)``.
        """
        on_frac = self._get_proxy_kwarg('onshore_obs_frac', feat, {})
        on_sf = on_frac.get('spatial', 0.0)
        on_tf = on_frac.get('temporal', 1.0)
        feat_mask = self._get_obs_mask(hi_res, on_sf, on_tf)
        if topo is None:
            return feat_mask
        off_frac = self._get_proxy_kwarg('offshore_obs_frac', feat, {})
        if not off_frac:
            return feat_mask
        off_sf = off_frac.get('spatial', 0.0)
        off_tf = off_frac.get('temporal', 1.0)
        offshore_mask = self._get_obs_mask(hi_res, off_sf, off_tf)
        return np.where(topo[..., None] > 0, feat_mask, offshore_mask)

    def _get_full_obs_mask(self, hi_res):
        """Define observation mask for the current batch. Builds a per-feature
        composite mask that applies separate onshore and offshore fractions and
        supports per-feature ``proxy_obs_kwargs`` overrides."""
        topo = self._get_topo(hi_res)
        per_feat_masks = [
            self._get_feat_obs_mask(hi_res, feat, topo)
            for feat in self.obs_features
        ]
        return np.concatenate(per_feat_masks, axis=-1)
