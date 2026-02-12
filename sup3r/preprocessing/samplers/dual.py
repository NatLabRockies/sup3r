"""Dual Sampler objects. These are used to sample from paired datasets with
low and high resolution data. These paired datasets are contained in a
Sup3rDataset object."""

import logging
from typing import Optional

from sup3r.preprocessing.base import Sup3rDataset
from sup3r.preprocessing.utilities import lowered

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
        mode: str = 'lazy',
    ):
        """
        Parameters
        ----------
        data : Sup3rDataset
            A :class:`~sup3r.preprocessing.base.Sup3rDataset` instance with
            low-res and high-res data members, and optionally an obs member.
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
            split between `lr_only_features` and `hr_exo_features`.

            lr_only_features : list | tuple
                List of feature names or patt*erns that should only be
                included in the low-res training set and not the high-res
                observations.
            hr_exo_features : list | tuple
                List of feature names or patt*erns that should be included
                in the high-resolution observation but not expected to be
                output from the generative model. An example is high-res
                topography that is to be injected mid-network.
        mode : str
            Mode for sampling data. Options are 'lazy' or 'eager'. 'eager' mode
            pre-loads all data into memory as numpy arrays for faster access.
            'lazy' mode samples directly from the underlying data object, which
            could be backed by dask arrays or on-disk netCDF files.
        """
        msg = (
            f'{self.__class__.__name__} requires a Sup3rDataset object '
            'with `.low_res` and `.high_res` data members, and optionally an '
            '`.obs` member, in that order'
        )
        dnames = ['low_res', 'high_res', 'obs'][: len(data)]
        check = (
            hasattr(data, dname) and getattr(data, dname) == data[i]
            for i, dname in enumerate(dnames)
        )
        assert check, msg

        super().__init__(
            data=data,
            sample_shape=sample_shape,
            batch_size=batch_size,
            mode=mode,
        )
        self.lr_data, self.hr_data = self.data.low_res, self.data.high_res
        self.lr_sample_shape = (
            self.hr_sample_shape[0] // s_enhance,
            self.hr_sample_shape[1] // s_enhance,
            self.hr_sample_shape[2] // t_enhance,
        )
        feature_sets = feature_sets or {}
        self._lr_only_features = feature_sets.get('lr_only_features', [])
        self._hr_exo_features = feature_sets.get('hr_exo_features', [])
        self.features = self.get_features(feature_sets)
        self.lr_features = [
            f for f in self.features if f in self.lr_data.features
        ]
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        self.check_for_consistent_shapes()
        post_init_args = {
            'lr_sample_shape': self.lr_sample_shape,
            'hr_sample_shape': self.hr_sample_shape,
            'lr_features': self.lr_features,
            'hr_features': self.hr_features,
        }
        self.post_init_log(post_init_args)

    def get_features(self, feature_sets):
        """Return default set of features composed from data vars in low res
        and high res data objects or the value provided through the
        feature_sets dictionary."""
        features = []
        _ = [
            features.append(f)
            for f in [*self.lr_data.features, *self.hr_data.features]
            if f not in features and f not in lowered(self._hr_exo_features)
        ]
        features += lowered(self._hr_exo_features)
        return feature_sets.get('features', features)

    def check_for_consistent_shapes(self):
        """Make sure container shapes are compatible with enhancement
        factors."""
        enhanced_shape = (
            self.lr_data.shape[0] * self.s_enhance,
            self.lr_data.shape[1] * self.s_enhance,
            self.lr_data.shape[2] * self.t_enhance,
        )
        msg = (
            f'hr_data.shape {self.hr_data.shape[:-1]} and enhanced '
            f'lr_data.shape {enhanced_shape} are not compatible with '
            'the given enhancement factors'
        )
        assert self.hr_data.shape[:-1] == enhanced_shape, msg

    def get_sample_index(self, n_obs=None):
        """Get paired sample index, consisting of index for the low res sample
        and the index for the high res sample with the same spatiotemporal
        extent. Optionally includes an extra high res index if the sample data
        includes observation data."""
        n_obs = n_obs or self.batch_size
        spatial_slice = uniform_box_sampler(
            self.lr_data.shape, self.lr_sample_shape[:2]
        )
        time_slice = uniform_time_sampler(
            self.lr_data.shape, self.lr_sample_shape[2] * n_obs
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
        obs_index = (*hr_index, self.hr_out_features)
        hr_index = (*hr_index, self.hr_features)

        sample_index = (lr_index, hr_index, obs_index)
        return sample_index[: len(self.data)]
