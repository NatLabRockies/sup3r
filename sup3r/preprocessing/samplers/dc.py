"""Data centric sampler. This samples container data according to weights
which are updated during training based on performance of the model."""

import logging
from typing import Optional, Union

import dask.array as da
import numpy as np

from sup3r.preprocessing.accessor import Sup3rX
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.preprocessing.samplers.base import Sampler
from sup3r.preprocessing.samplers.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)

logger = logging.getLogger(__name__)


class SamplerDC(Sampler):
    """DataCentric Sampler class used for sampling based on weights which can
    be updated during training."""

    def __init__(
        self,
        data: Union[Sup3rX, Sup3rDataset],
        sample_shape: Optional[tuple] = None,
        batch_size: int = 16,
        feature_sets: Optional[dict] = None,
        mode: str = 'lazy',
        spatial_weights: Optional[
            Union[np.ndarray, da.core.Array, list]
        ] = None,
        temporal_weights: Optional[
            Union[np.ndarray, da.core.Array, list]
        ] = None,
    ):
        """
        Parameters
        ----------
        data : Union[Sup3rX, Sup3rDataset],
            Object with data that will be sampled from. Usually the `.data`
            attribute of various :class:`Container` objects.  i.e.
            :class:`Loader`, :class:`Rasterizer`, :class:`Deriver`, as long as
            the spatial dimensions are not flattened.
        sample_shape : tuple
            Size of arrays to sample from the contained data.
        batch_size : int
            Number of samples to get to build a single batch. A sample of
            (sample_shape[0], sample_shape[1], batch_size * sample_shape[2])
            is first selected from underlying dataset and then reshaped into
            (batch_size, *sample_shape) to get a single batch. This is more
            efficient than getting N = batch_size samples and then stacking.
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
        mode : str
            Loading mode for sampling.
            See :class:`~sup3r.preprocessing.Sampler`
        spatial_weights : Union[np.ndarray, da.core.Array] | List | None
            Set of weights used to initialize the spatial sampling. e.g. If we
            want to start off sampling across 2 spatial bins evenly this should
            be [0.5, 0.5]. During training these weights will be updated based
            only performance across the bins associated with these weights.
        temporal_weights : Union[np.ndarray, da.core.Array] | List | None
            Set of weights used to initialize the temporal sampling. e.g. If we
            want to start off sampling only the first season of the year this
            should be [1, 0, 0, 0]. During training these weights will be
            updated based only performance across the bins associated with
            these weights.
        """
        self.spatial_weights = spatial_weights or [1]
        self.temporal_weights = temporal_weights or [1]
        super().__init__(
            data=data,
            sample_shape=sample_shape,
            batch_size=batch_size,
            feature_sets=feature_sets,
            mode=mode,
        )

    def update_weights(self, spatial_weights, temporal_weights):
        """Update spatial and temporal sampling weights."""
        self.spatial_weights = spatial_weights
        self.temporal_weights = temporal_weights

    def get_sample_index(self, n_obs=None):
        """Randomly gets weighted spatial sample and time sample indices

        Returns
        -------
        observation_index : tuple
            Tuple of sampled spatial grid, time slice, and features indices.
            Used to get single observation like self.data[observation_index]
        """
        n_obs = n_obs or self.batch_size
        if self.spatial_weights is not None:
            spatial_slice = weighted_box_sampler(
                self.shape, self.sample_shape[:2], weights=self.spatial_weights
            )
        else:
            spatial_slice = uniform_box_sampler(
                self.shape, self.sample_shape[:2]
            )
        if self.temporal_weights is not None:
            time_slice = weighted_time_sampler(
                self.shape,
                self.sample_shape[2] * n_obs,
                weights=self.temporal_weights,
            )
        else:
            time_slice = uniform_time_sampler(
                self.shape, self.sample_shape[2] * n_obs
            )
        return (*spatial_slice, time_slice, self.features)
