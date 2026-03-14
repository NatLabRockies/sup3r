"""Base collection classes. These are objects that contain sets / lists of
containers like batch handlers. Of course these also contain data so they're
containers too!

TODO: https://github.com/xarray-contrib/datatree could unify Sup3rDataset and
collections of data. Consider migrating once datatree has been fully
integrated into xarray (in progress as of 8/8/2024)
"""

from typing import TYPE_CHECKING, Union

import numpy as np

from sup3r.preprocessing.base import Container

if TYPE_CHECKING:
    from sup3r.preprocessing.samplers.base import Sampler
    from sup3r.preprocessing.samplers.dual import DualSampler


class Collection(Container):
    """Object consisting of a set of containers. These objects are distinct
    from :class:`~sup3r.preprocessing.base.Sup3rDataset` objects, which also
    contain multiple data members, because these members are completely
    independent of each other. They are collected together for the purpose of
    expanding a training dataset (e.g. BatchHandlers)."""

    def __init__(
        self,
        containers: Union[
            list['Container'],
            list['Sampler'],
            list['DualSampler'],
        ],
    ):
        super().__init__()
        self.data = tuple(c.data for c in containers)
        self.containers = containers
        self._features: list = []

    @property
    def features(self):
        """Get all features contained in data."""
        if not self._features:
            _ = [
                self._features.append(f)
                for f in np.concatenate([c.features for c in self.containers])
                if f not in self._features
            ]
        return self._features

    @property
    def container_weights(self):
        """Get weights used to sample from different containers based on
        relative sizes"""
        sizes = [c.size for c in self.containers]
        weights = sizes / np.sum(sizes)
        return weights.astype(np.float32)

    def __getattr__(self, attr):
        """Get attributes from self or the first container in the
        collection."""
        return self.check_shared_attr(attr)

    def check_shared_attr(self, attr):
        """Check if all containers have the same value for `attr`. If they do
        the collection effectively inherits those attributes."""
        msg = f'Not all collection containers have the same value for {attr}'
        out = getattr(self.containers[0], attr, None)
        if isinstance(out, (np.ndarray, list, tuple)):
            check = all(
                np.array_equal(getattr(c, attr, None), out)
                for c in self.containers
            )
        else:
            check = all(getattr(c, attr, None) == out for c in self.containers)
        assert check, msg
        return out
