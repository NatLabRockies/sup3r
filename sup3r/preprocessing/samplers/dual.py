"""Dual Sampler objects. These are used to sample from paired datasets with
low and high resolution data. These paired datasets are contained in a
Sup3rDataset object."""

import logging
from typing import Optional

from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.utilities import Timer

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
            See :class:`~sup3r.preprocessing.Sampler` for full documentation.
        proxy_obs_kwargs : dict | None
            See :class:`~sup3r.preprocessing.Sampler` for full documentation.
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

        self.timer = Timer()
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

    def check_feature_consistency(self):
        """Make sure features are consistent with the data and with each
        other."""
        super().check_feature_consistency()
        msg = (
            f'lr_features {self.lr_features} must be in low res data features '
            f'{self.data.low_res.features}'
        )
        assert set(self.lr_features).issubset(
            set(self.data.low_res.features)
        ), msg
        msg = (
            f'hr_out_features {self.hr_out_features} must be in high res data '
            f'features {self.data.high_res.features}'
        )
        assert set(self.hr_out_features).issubset(
            set(self.data.high_res.features)
        ), msg

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
        hr_index = (*hr_index, self.hr_sample_features)

        return (lr_index, hr_index)
