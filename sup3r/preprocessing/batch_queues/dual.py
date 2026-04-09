"""Base objects which generate, build, and operate on batches. Also can
interface with models."""

import logging

from scipy.ndimage import gaussian_filter

from .abstract import AbstractBatchQueue

logger = logging.getLogger(__name__)


class DualBatchQueue(AbstractBatchQueue):
    """Base BatchQueue for use with
    :class:`~sup3r.preprocessing.samplers.DualSampler` objects."""

    def __init__(self, samplers, **kwargs):
        """
        See Also
        --------
        :class:`~sup3r.preprocessing.batch_queues.abstract.AbstractBatchQueue`
        """
        super().__init__(samplers, **kwargs)
        self.check_enhancement_factors()

    _signature_objs = (AbstractBatchQueue,)

    @property
    def queue_shape(self):
        """Shape of objects stored in the queue. Optionally includes shape of
        observation data which would be included in an extra content loss
        term"""
        return [
            (self.batch_size, *self.lr_shape),
            (self.batch_size, *self.hr_shape),
        ]

    def check_enhancement_factors(self):
        """Make sure each DualSampler has the same enhancment factors and they
        match those provided to the BatchQueue."""

        s_factors = [c.s_enhance for c in self.containers]
        msg = (
            f'Received s_enhance = {self.s_enhance} but not all '
            f'DualSamplers in the collection have the same value: {s_factors}.'
        )
        assert all(self.s_enhance == s for s in s_factors), msg
        t_factors = [c.t_enhance for c in self.containers]
        msg = (
            f'Received t_enhance = {self.t_enhance} but not all '
            f'DualSamplers in the collection have the same value.'
        )
        assert all(self.t_enhance == t for t in t_factors), msg

    def transform(self, samples, smoothing=None, smoothing_ignore=None):
        """Perform smoothing if requested.

        Note
        ----
        This does not include temporal or spatial coarsening like
        :class:`SingleBatchQueue`
        """
        low_res = samples[0]

        if smoothing is not None:
            feat_iter = [
                j
                for j in range(low_res.shape[-1])
                if self.features[j] not in smoothing_ignore
            ]
            for i in range(low_res.shape[0]):
                for j in feat_iter:
                    low_res[i, ..., j] = gaussian_filter(
                        low_res[i, ..., j], smoothing, mode='nearest'
                    )
        return low_res, *samples[1:]
