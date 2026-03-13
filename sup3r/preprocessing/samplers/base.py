"""Basic ``Sampler`` objects. These are containers which also can sample from
the underlying data. These interface with ``BatchQueues`` so they also have
additional information about how different features are used by models."""

import logging
from fnmatch import fnmatch
from typing import Optional
from warnings import warn

import numpy as np

from sup3r.preprocessing.base import Container
from sup3r.preprocessing.samplers.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
)
from sup3r.preprocessing.utilities import compute_if_dask, lowered
from sup3r.utilities.utilities import RANDOM_GENERATOR

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
                List of feature names or patt*erns that should be available
                as high-resolution model inputs (like topography or
                observations). These are injected into the model mid-network
                to condition output on high-resolution information. The model
                configuration should have the appropriate layers to use these
                features. e.g. ``Sup3rConcat`` for topography injection,
                ``Sup3rObsModel`` or ``Sup3rCrossAttention`` for obs injection.
                If no entry is provided then hr_exo_features will be empty.

            *To include sparse features as inputs or targets the features
            must have an "_obs" suffix.
        proxy_obs_kwargs : dict | None
            Optional dictionary of keyword arguments to pass to the proxy
            observation generator. This is only used when training with proxy
            observations. Keys can include ``onshore_obs_frac`` and
            ``offshore_obs_frac`` which specify the fraction of the batch that
            should be treated as onshore and offshore observations,
            respectively. For example, ``proxy_obs_kwargs={'onshore_obs_frac':
            {'spatial': 0.1, 'temporal': 0.2}, 'offshore_obs_frac': {'spatial':
            0.05, 'temporal': 0.1}}`` would specify that for the onshore
            region observations cover 10% of the spatial domain and 20% of the
            temporal domain, while for the offshore region observations cover
            5% of the spatial domain and 10% of the temporal domain. Instead of
            a single float, these can also be lists to specify a lower and
            upper bound for the spatial and temporal fractions, in which case
            the actual fraction for each batch will be sampled uniformly
            between these bounds.
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
        check = bool(self.proxy_obs_kwargs)
        check = check or (
            len(self.obs_features) > 0
            and all(f not in self.features for f in self.obs_features)
        )
        return check

    @property
    def onshore_obs_frac(self):
        """Fraction of onshore observations to include in each batch when using
        proxy observations. This can be a single float or a dictionary with
        keys 'spatial' and 'temporal' to specify the fraction for each domain.
        If a dictionary is provided, the actual fraction for each batch will be
        sampled uniformly between the specified spatial and temporal fractions.
        """
        return self.proxy_obs_kwargs.get('onshore_obs_frac', {})

    @property
    def offshore_obs_frac(self):
        """Fraction of offshore observations to include in each batch when
        using proxy observations. This can be a single float or a dictionary
        with keys 'spatial' and 'temporal' to specify the fraction for each
        domain.  If a dictionary is provided, the actual fraction for each
        batch will be sampled uniformly between the specified spatial and
        temporal fractions.
        """
        return self.proxy_obs_kwargs.get('offshore_obs_frac', {})

    def get_sample_index(self, n_obs=None):
        """Randomly gets spatiotemporal sample index.

        Notes
        -----
        If ``n_obs > 1`` this will get a time slice with ``n_obs *
        self.sample_shape[2]`` time steps, which will then be reshaped into
        ``n_obs`` samples each with ``self.sample_shape[2]`` time steps. This
        is a much more efficient way of getting batches of samples but only
        works if there are enough continuous time steps to sample.

        Returns
        -------
        sample_index : tuple
            Tuple of latitude slice, longitude slice, time slice, and features.
            Used to get single observation like ``self.data[sample_index]``
        """
        n_obs = n_obs or self.batch_size
        spatial_slice = uniform_box_sampler(self.shape, self.sample_shape[:2])
        time_slice = uniform_time_sampler(
            self.shape, self.sample_shape[2] * n_obs
        )
        feats = (
            self.features
            if not self.use_proxy_obs
            else self.features[: -len(self.obs_features)]
        )
        return (*spatial_slice, time_slice, feats)

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
        if self.data.shape[2] < self.sample_shape[2] * self.batch_size:
            logger.warning(msg)
            warn(msg)
        if self.mode == 'eager':
            logger.info('Received mode = "eager".')
            _ = self.compute()

    def check_feature_consistency(self):
        """Check that the feature sets are consistent with each other and the
        obs features are configured correctly."""
        if self.use_proxy_obs and not all(
            f in self.hr_features for f in self.obs_features
        ):
            msg = (
                'When using proxy observations, all obs features must be '
                'included either in hr_out_features or hr_exo_features.'
            )
            raise ValueError(msg)

        if self.use_proxy_obs and any(
            f in self.data.features for f in self.obs_features
        ):
            msg = (
                f'Obs features {self.obs_features} cannot be in the data '
                f'features {self.data.features} when using proxy observations.'
            )
            raise ValueError(msg)

        if len(self.obs_features) > 0 and any(
            f in self.hr_exo_features for f in self.obs_features
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
                f'of the full high-res feature set: {self.hr_features}'
            )
            assert list(self.hr_exo_features) == list(
                self.hr_features[-len(self.hr_exo_features) :]
            ), msg

        assert all(
            f in lowered(self.data.features) for f in self.lr_features
        ), (
            f'All lr_features {self.lr_features} must be in the data features '
            f'{self.data.features}.'
        )
        assert all(
            f in lowered(self.data.features) for f in self.hr_out_features
        ), (
            f'All hr_out_features {self.hr_out_features} must be in the data '
            f'features {self.data.features}.'
        )
        if not self.use_proxy_obs:
            assert all(
                f in lowered(self.data.features) for f in self.hr_exo_features
            ), (
                f'All hr_exo_features {self.hr_exo_features} must be in the '
                f'data features {self.data.features} when not using proxy '
                'observations.'
            )
        else:
            feats = set(self.hr_exo_features) - set(self.obs_features)
            assert all(f in lowered(self.data.features) for f in feats), (
                f'All non-obs hr_exo_features {feats} must be in the data '
                f'features {self.data.features} when using proxy observations.'
            )

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
                'Found 2D sample shape of {}. Adding temporal dim of 1'.format(
                    self._sample_shape
                )
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
        out = self.data.sample(self.get_sample_index(n_obs=self.batch_size))
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

    @property
    def obs_features_ind(self):
        """Get the source feature indices in ``features`` for each obs
        feature. Each obs feature named ``<feature>_obs`` maps to the
        corresponding ``<feature>`` in the features.

        Returns
        -------
        list[int]
            Indices into ``features`` for each obs feature source.
        """
        if len(self.obs_features) == 0:
            return []

        if self.use_proxy_obs:
            return [
                self.hr_features.index(f.replace('_obs', ''))
                for f in self.obs_features
            ]
        else:
            return [self.hr_features.index(f) for f in self.obs_features]

    def _get_proxy_obs(self, hi_res):
        """Generate proxy observation data by masking the gridded high-res
        data. Unobserved locations are set to NaN.

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
        obs[obs_mask[..., : obs.shape[-1]]] = np.nan
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

    @property
    def lr_features(self):
        """List of feature names or patt*erns to use as low-resolution model
        inputs. If no entry is provided then all available features from the
        data will be used."""
        return self._parse_features(self._lr_features)

    @property
    def hr_features(self):
        """List of feature names or patt*erns that should be available as
        either high-resolution model inputs (like topography or observations)
        or as ground truth targets. If no entry is provided then all available
        features from data will be used."""
        out = [
            f for f in self.hr_out_features if f not in self.hr_exo_features
        ]
        out += self.hr_exo_features
        return out

    @property
    def hr_out_features(self):
        """List of feature names or patt*erns that should be output by the
        generative model. If no entry is provided then all features in
        hr_features will be used."""
        hr_out = self._parse_features(self._hr_out_features)
        return self.lr_features if len(hr_out) == 0 else hr_out

    @property
    def hr_exo_features(self):
        """Get a list of exogenous high-resolution features that are only used
        for training e.g., mid-network high-res topo injection. These must come
        at the end of the high-res feature set. These can also be input to the
        model as low-res features."""
        return self._parse_features(self._hr_exo_features)

    @property
    def obs_features(self):
        """List of feature names or patt*erns that should be treated as
        observations. These features will be included in the high-res data but
        not the low-res data and won't necessarily be expected to be output by
        the generative model. These are different from the `hr_exo_features` in
        that they are intended to be used as observation features with NaN
        values where observations are not available."""
        return [f for f in self.hr_features if '_obs' in f]

    @property
    def hr_features_ind(self):
        """Get the high-resolution feature channel indices that should be
        included for training. This includes hr_out_features and
        hr_exo_features, Any high-resolution features that are only included in
        the data handler to be coarsened for the low-res input are removed.
        """
        return [self.features.index(f) for f in self.hr_features]

    @property
    def lr_features_ind(self):
        """Get the low-resolution feature channel indices that should be
        included for training. This includes lr_features.
        """
        return [self.features.index(f) for f in self.lr_features]

    @property
    def features(self):
        """Get the full set of features that should be included for training.
        This is the union of lr_features, hr_out_features and hr_exo_features.
        """
        feats = self.lr_features
        feats += [f for f in self.hr_out_features if f not in feats]
        feats += [f for f in self.hr_exo_features if f not in feats]
        return feats

    def _get_single_obs_mask(self, hi_res, spatial_frac, time_frac=1.0):
        """Get observation mask for a given spatial and temporal obs
        fraction for a single batch entry.

        Parameters
        ----------
        hi_res : np.ndarray
            True high resolution data for a single batch entry.
        spatial_frac : float
            Fraction of the spatial domain that should be treated as
            observations. This is a value between 0 and 1.
        time_frac : float, optional
            Fraction of the temporal domain that should be treated as
            observations. This is a value between 0 and 1. Default is 1.0

        Returns
        -------
        np.ndarray
            Mask which is True for locations that are not observed and False
            for locations that are observed.
            (spatial_1, spatial_2, n_features)
            (spatial_1, spatial_2, n_temporal, n_features)
        """
        mask_shape = [*hi_res.shape[:-1], len(self.hr_out_features)]
        s_mask = RANDOM_GENERATOR.uniform(size=mask_shape[1:3]) <= spatial_frac
        s_mask = s_mask[..., None, None]
        t_mask = RANDOM_GENERATOR.uniform(size=mask_shape[-2]) <= time_frac
        t_mask = t_mask[None, None, ..., None]
        mask = ~(s_mask & t_mask)
        return np.repeat(mask, mask_shape[-1], axis=-1)

    def _get_obs_mask(self, hi_res, spatial_frac, time_frac=1.0):
        """Get observation mask for a given spatial and temporal obs
        fraction for an entire batch. This is divided between spatial and
        temporal fractions because often the spatial fraction is significantly
        lower than the temporal fraction in practice, e.g. for a given spatial
        location there might be observations for most of the time period but
        only a small fraction of the spatial domain is observed.

        Parameters
        ----------
        hi_res : np.ndarray
            True high resolution data for the entire batch.
        spatial_frac : float | list
            Fraction of the spatial domain that should be treated as
            observations. This is a value between 0 and 1 or a list with
            lower and upper bounds for the spatial fraction.
        time_frac : float | list, optional
            Fraction of the temporal domain that should be treated as
            observations. This is a value between 0 and 1 or a list with
            lower and upper bounds for the temporal fraction. Default is 1.0

        Returns
        -------
        np.ndarray
            Mask which is True for locations that are not observed and False
            for locations that are observed.
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
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
        s_fracs = RANDOM_GENERATOR.uniform(*s_range, size=hi_res.shape[0])
        t_fracs = RANDOM_GENERATOR.uniform(*t_range, size=hi_res.shape[0])
        s_fracs = np.clip(s_fracs, 0, 1)
        t_fracs = np.clip(t_fracs, 0, 1)
        mask = np.stack(
            [
                self._get_single_obs_mask(hi_res, s, t)
                for s, t in zip(s_fracs, t_fracs)
            ],
            axis=0,
        )
        return mask

    def _get_full_obs_mask(self, hi_res):
        """Define observation mask for the current batch. This differs from
        ``_get_obs_mask`` by defining a composite mask based on separate
        onshore and offshore masks. This is because there is often more
        observation data available onshore than offshore."""
        on_sf = self.onshore_obs_frac.get('spatial', 0.0)
        on_tf = self.onshore_obs_frac.get('time', 1.0)
        obs_mask = self._get_obs_mask(hi_res, on_sf, on_tf)
        if 'topography' in self.hr_features and self.offshore_obs_frac:
            topo_idx = self.hr_features.index('topography')
            topo = hi_res[..., topo_idx]
            off_sf = self.offshore_obs_frac.get('spatial', 0.0)
            off_tf = self.offshore_obs_frac.get('time', 1.0)
            offshore_mask = self._get_obs_mask(hi_res, off_sf, off_tf)
            obs_mask = np.where(topo[..., None] > 0, obs_mask, offshore_mask)
        return obs_mask
