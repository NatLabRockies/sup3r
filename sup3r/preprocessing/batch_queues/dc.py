"""Data centric batch handler for dynamic batching strategy based on
performance during training"""

import logging

import numpy as np

from .base import SingleBatchQueue

logger = logging.getLogger(__name__)


class BatchQueueDC(SingleBatchQueue):
    """Sample from data based on spatial and temporal weights. These weights
    can be derived from validation training losses and updated during training
    or set a priori to construct a validation queue"""

    def __init__(self, samplers, n_space_bins=1, n_time_bins=1, **kwargs):
        """
        Parameters
        ----------
        samplers : list[Sampler]
            List of Sampler instances
        n_space_bins : int
            Number of spatial bins to use for weighted sampling. e.g. if this
            is 4 the spatial domain will be divided into 4 equal regions and
            losses will be calculated across these regions during traning in
            order to adaptively sample from lower performing regions.
        n_time_bins : int
            Number of time bins to use for weighted sampling. e.g. if this
            is 4 the temporal domain will be divided into 4 equal periods and
            losses will be calculated across these periods during traning in
            order to adaptively sample from lower performing time periods.
        kwargs : dict
            Keyword arguments for parent class.
        """
        self.n_space_bins = n_space_bins
        self.n_time_bins = n_time_bins
        self._spatial_weights = np.ones(n_space_bins) / n_space_bins
        self._temporal_weights = np.ones(n_time_bins) / n_time_bins
        super().__init__(samplers, **kwargs)

    _signature_objs = (__init__, SingleBatchQueue)

    def sample_batch(self):
        """Update weights and get batch of samples from sampled container."""
        sampler = self.get_random_container()
        sampler.update_weights(self.spatial_weights, self.temporal_weights)
        return next(sampler)

    @property
    def spatial_weights(self):
        """Get weights used to sample spatial bins."""
        return self._spatial_weights

    @property
    def temporal_weights(self):
        """Get weights used to sample temporal bins."""
        return self._temporal_weights

    def update_weights(self, spatial_weights, temporal_weights):
        """Set weights used to sample spatial and temporal bins. This is called
        by :class:`Sup3rGanDC` after an epoch to update weights based on model
        performance across validation samples."""
        self._spatial_weights = spatial_weights
        self._temporal_weights = temporal_weights


class ValBatchQueueDC(BatchQueueDC):
    """Queue to construct a single batch for each spatiotemporal validation
    bin. e.g. If we have 4 time bins and 1 space bin this will get `batch_size`
    samples for 4 batches, with `batch_size` samples from each bin. The model
    performance across these batches will determine the weights for how the
    training batch queue is sampled."""

    def __init__(self, samplers, n_space_bins=1, n_time_bins=1, **kwargs):
        """
        Parameters
        ----------
        samplers : list[Sampler]
            See :class:`~sup3r.preprocessing.BatchQueueDC`
            for full documentation.
        n_space_bins : int
            See :class:`~sup3r.preprocessing.BatchQueueDC`
            for full documentation.
        n_time_bins : int
            See :class:`~sup3r.preprocessing.BatchQueueDC`
            for full documentation.
        kwargs : dict
            Keyword arguments for parent class.
        """
        super().__init__(
            samplers,
            n_space_bins=n_space_bins,
            n_time_bins=n_time_bins,
            **kwargs,
        )
        self.n_batches = n_space_bins * n_time_bins

    _signature_objs = (__init__, BatchQueueDC)

    @property
    def spatial_weights(self):
        """Sample entirely from this spatial bin determined by the batch
        number."""
        self._spatial_weights = np.eye(
            1,
            self.n_space_bins,
            self._batch_count % self.n_space_bins,
            dtype=np.float32,
        )[0]
        return self._spatial_weights

    @property
    def temporal_weights(self):
        """Sample entirely from this temporal bin determined by the batch
        number."""
        self._temporal_weights = np.eye(
            1,
            self.n_time_bins,
            self._batch_count % self.n_time_bins,
            dtype=np.float32,
        )[0]
        return self._temporal_weights
