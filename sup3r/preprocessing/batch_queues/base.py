"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging

from sup3r.preprocessing.utilities import numpy_if_tensor
from sup3r.utilities.utilities import spatial_coarsening, temporal_coarsening

from .abstract import AbstractBatchQueue
from .utilities import smooth_data

logger = logging.getLogger(__name__)


class SingleBatchQueue(AbstractBatchQueue):
    """Base BatchQueue class for single dataset containers

    Note
    ----
    Here we use `len(self.features)` for the last dimension of samples, since
    samples in :class:`SingleBatchQueue` queues are coarsened to produce
    low-res samples, and then the `lr_only_features` are removed with
    `hr_features_ind`. In contrast, for samples in :class:`DualBatchQueue`
    queues there are low / high res pairs and the high-res only stores the
    `hr_features`"""

    @property
    def queue_shape(self):
        """Shape of objects stored in the queue."""
        return [(self.batch_size, *self.hr_shape)]

    def transform(
        self,
        samples,
        smoothing=None,
        smoothing_ignore=None,
        temporal_coarsening_method='subsample',
    ):
        """Coarsen high res data to get corresponding low res batch.

        Parameters
        ----------
        samples : Union[np.ndarray, da.core.Array]
            High resolution batch of samples.
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        smoothing : float | None
            Standard deviation to use for gaussian filtering of the coarse
            data. This can be tuned by matching the kinetic energy of a low
            resolution simulation with the kinetic energy of a coarsened and
            smoothed high resolution simulation. If None no smoothing is
            performed.
        smoothing_ignore : list | None
            List of features to ignore for the smoothing filter. None will
            smooth all features if smoothing kwarg is not None
        temporal_coarsening_method : str
            Method to use for temporal coarsening. Can be subsample, average,
            min, max, or total

        Returns
        -------
        low_res : Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        high_res : Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        lr_samples = numpy_if_tensor(samples)[..., self.lr_features_ind]
        low_res = spatial_coarsening(lr_samples, self.s_enhance)
        low_res = (
            low_res
            if self.t_enhance == 1
            else temporal_coarsening(
                low_res, self.t_enhance, temporal_coarsening_method
            )
        )
        smoothing_ignore = (
            smoothing_ignore if smoothing_ignore is not None else []
        )
        low_res = smooth_data(
            low_res, self.lr_features, smoothing_ignore, smoothing
        )
        high_res = numpy_if_tensor(samples)[..., self.hr_features_ind]
        return low_res, high_res
