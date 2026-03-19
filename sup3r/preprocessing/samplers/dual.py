"""Dual Sampler objects. These are used to sample from paired datasets with
low and high resolution data. These paired datasets are contained in a
Sup3rDataset object."""

import logging
from typing import Optional

from sup3r.preprocessing.base import Sup3rDataset

from .base import Sampler
from .utilities import uniform_box_sampler, uniform_time_sampler

logger = logging.getLogger(__name__)


class DualSampler(Sampler):
    """Sampler for sampling from paired (or dual) datasets. Pairs consist of
    low and high resolution data, which are contained by a Sup3rDataset. This
    can also include extra observation data on the same grid as the
    high-resolution data which has NaNs at points where observation data
    doesn't exist. This will be used in an additional content loss term."""

    def __init__(
        self,
        data: Sup3rDataset,
        sample_shape: Optional[tuple] = None,
        batch_size: int = 16,
        s_enhance: int = 1,
        t_enhance: int = 1,
        feature_sets: Optional[dict] = None,
        proxy_obs_kwargs: Optional[dict] = None,
        mode: str = 'lazy',
    ):
        """
        Parameters
        ----------
        data : Sup3rDataset
            A :class:`~sup3r.preprocessing.base.Sup3rDataset` instance with
            low-res and high-res data members.
        sample_shape : tuple
            Size of arrays to sample from the high-res data. The sample shape
            for the low-res sampler will be determined from the enhancement
            factors.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
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
            respectively. For example, ``proxy_obs_kwargs={ 'onshore_obs_frac':
            { 'spatial': 0.1, 'temporal': 0.2}, 'offshore_obs_frac': {
            'spatial': 0.05, 'temporal': 0.1} }`` would specify that for the
            onshore region observations cover 10% of the spatial domain and 20%
            of the temporal domain, while for the offshore region observations
            cover 5% of the spatial domain and 10% of the temporal domain.
            Instead of a single float, these can also be lists to specify a
            lower and upper bound for the spatial and temporal fractions, in
            which case the actual fraction for each batch will be sampled
            uniformly between these bounds.
        mode : str
            Mode for sampling data. Options are 'lazy' or 'eager'. 'eager' mode
            pre-loads all data into memory as numpy arrays for faster access.
            'lazy' mode samples directly from the underlying data object, which
            could be backed by dask arrays or on-disk netCDF files.
        """
        msg = (
            f'{self.__class__.__name__} requires a Sup3rDataset object '
            'with `.low_res` and `.high_res` data members, in that order'
        )
        check = all(
            hasattr(data, dname) and getattr(data, dname) == data[i]
            for i, dname in enumerate(['low_res', 'high_res'])
        )
        assert check, msg

        self.data = data
        feature_sets = feature_sets or {}
        self._lr_features = feature_sets.get(
            'lr_features', self.data.low_res.features
        )
        self._hr_exo_features = feature_sets.get('hr_exo_features', [])
        self._hr_out_features = feature_sets.get(
            'hr_out_features', self.data.high_res.features
        )
        self.proxy_obs_kwargs = proxy_obs_kwargs or {}
        self.mode = mode
        self.sample_shape = sample_shape or (10, 10, 1)
        self.batch_size = batch_size

        self.lr_sample_shape = (
            self.hr_sample_shape[0] // s_enhance,
            self.hr_sample_shape[1] // s_enhance,
            self.hr_sample_shape[2] // t_enhance,
        )
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance

        self.preflight()
        self.check_shape_consistency()
        self.check_feature_consistency()
        post_init_args = {
            'lr_sample_shape': self.lr_sample_shape,
            'hr_sample_shape': self.hr_sample_shape,
            'lr_features': self.lr_features,
            'hr_features': self.hr_features,
        }
        self.post_init_log(post_init_args)

    @property
    def hr_source_features(self):
        """Features available natively at high-resolution."""
        out = [
            f for f in self.hr_out_features if f not in self.hr_exo_features
        ]
        out += self.hr_exo_features
        return out

    def check_shape_consistency(self):
        """Make sure container shapes are compatible with enhancement
        factors."""
        enhanced_shape = (
            self.data.low_res.shape[0] * self.s_enhance,
            self.data.low_res.shape[1] * self.s_enhance,
            self.data.low_res.shape[2] * self.t_enhance,
        )
        msg = (
            f'hr_data.shape {self.data.high_res.shape[:-1]} and enhanced '
            f'lr_data.shape {enhanced_shape} are not compatible with '
            'the given enhancement factors'
        )
        assert self.data.high_res.shape[:-1] == enhanced_shape, msg

    def get_sample_index(self, n_obs=None):
        """Get paired sample index, consisting of index for the low res sample
        and the index for the high res sample with the same spatiotemporal
        extent. Optionally includes an extra high res index if the sample data
        includes observation data."""
        n_obs = n_obs or self.batch_size
        spatial_slice = uniform_box_sampler(
            self.data.low_res.shape, self.lr_sample_shape[:2]
        )
        time_slice = uniform_time_sampler(
            self.data.low_res.shape, self.lr_sample_shape[2] * n_obs
        )
        lr_index = (*spatial_slice, time_slice, self.lr_features)
        hr_index = [
            slice(s.start * self.s_enhance, s.stop * self.s_enhance)
            for s in lr_index[:2]
        ]
        hr_index += [
            slice(s.start * self.t_enhance, s.stop * self.t_enhance)
            for s in lr_index[2:-1]
        ]
        hr_feats = (
            self.hr_source_features[: -len(self.obs_features)]
            if self.use_proxy_obs
            else self.hr_source_features
        )
        hr_index = (*hr_index, hr_feats)

        return (lr_index, hr_index)
