"""Abstract batch queue class used for conditional moment estimation."""

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from sup3r.models.conditional import Sup3rCondMom
from sup3r.preprocessing.base import DsetTuple
from sup3r.preprocessing.utilities import numpy_if_tensor

from .base import SingleBatchQueue
from .utilities import spatial_simple_enhancing, temporal_simple_enhancing

if TYPE_CHECKING:
    from sup3r.preprocessing.samplers import DualSampler, Sampler

logger = logging.getLogger(__name__)


class ConditionalBatchQueue(SingleBatchQueue):
    """BatchQueue class for conditional moment estimation."""

    def __init__(
        self,
        samplers: Union[list['Sampler'], list['DualSampler']],
        time_enhance_mode: str = 'constant',
        lower_models: Optional[dict[int, Sup3rCondMom]] = None,
        s_padding: int = 0,
        t_padding: int = 0,
        end_t_padding: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        samplers : list[Sampler] | list[DualSampler]
            List of samplers to use for queue.
        time_enhance_mode : str
            [constant, linear]
            Method to enhance temporally when constructing subfilter. At every
            temporal location, a low-res temporal data is subtracted from the
            high-res temporal data predicted.  constant will assume that the
            low-res temporal data is constant between landmarks.  linear will
            linearly interpolate between landmarks to generate the low-res data
            to remove from the high-res.
        lower_models : dict[int, Sup3rCondMom] | None
            Dictionary of models that predict lower moments. For example, if
            this queue is part of a handler to estimate the 3rd moment
            `lower_models` could include models that estimate the 1st and 2nd
            moments. These lower moments can be required in higher order moment
            calculations.
        s_padding : int | None
            Width of spatial padding to predict only middle part. If None, no
            padding is used
        t_padding : int | None
            Width of temporal padding to predict only middle part. If None, no
            padding is used
        end_t_padding : bool | False
            Zero pad the end of temporal space.  Ensures that loss is
            calculated only if snapshot is surrounded by temporal landmarks.
            False by default
        kwargs : dict
            Keyword arguments for parent class
        """
        self.low_res = None
        self.high_res = None
        self.output = None
        self.s_padding = s_padding
        self.t_padding = t_padding
        self.end_t_padding = end_t_padding
        self.time_enhance_mode = time_enhance_mode
        self.lower_models = lower_models
        super().__init__(samplers, **kwargs)

    _signature_objs = (__init__, SingleBatchQueue)

    def make_mask(self, high_res):
        """Make mask for output. This is used to ensure consistency when
        training conditional moments.

        Note
        ----
        Consider the case of learning E(HR|LR) where HR is the high_res and LR
        is the low_res. In theory, the conditional moment estimation works if
        the full LR is passed as input and predicts the full HR. In practice,
        only the LR data that overlaps and surrounds the HR data is useful, ie
        E(HR|LR) = E(HR|LR_nei) where LR_nei is the LR data that surrounds the
        HR data. Physically, this is equivalent to saying that data far away
        from a region of interest does not matter.  This allows learning the
        conditional moments on spatial and temporal chunks only if one
        restricts the high_res output as being overlapped and surrounded by the
        input low_res.  The role of the mask is to ensure that the input
        low_res always surrounds the output high_res.

        Parameters
        ----------
        high_res : Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        mask: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """
        mask = np.zeros(high_res.shape, dtype=high_res.dtype)
        s_min = self.s_padding
        t_min = self.t_padding
        s_max = None if self.s_padding == 0 else -self.s_padding
        t_max = None if self.t_padding == 0 else -self.t_padding
        if self.end_t_padding and self.t_enhance > 1:
            if t_max is None:
                t_max = 1 - self.t_enhance
            else:
                t_max = 1 - self.t_enhance - self.t_padding

        if len(high_res.shape) == 4:
            mask[:, s_min:s_max, s_min:s_max, :] = 1.0
        elif len(high_res.shape) == 5:
            mask[:, s_min:s_max, s_min:s_max, t_min:t_max, :] = 1.0

        return mask

    @abstractmethod
    def make_output(self, samples):
        """Make custom batch output. This depends on the moment type being
        estimated. e.g. This could be the 1st moment, which is just high_res
        or the 2nd moment, which is (high_res - 1st moment) ** 2

        Parameters
        ----------
        samples : tuple[Union[np.ndarray, da.core.Array], ...]
            Tuple of low_res, high_res. Each array is:
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        output: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
        """

    def post_proc(self, samples):
        """Returns normalized collection of samples / observations along with
        mask and target output for conditional moment estimation. Performs
        coarsening on high-res data if :class:`Collection` consists of
        :class:`Sampler` objects and not :class:`DualSampler` objects

        Returns
        -------
        DsetTuple
            Namedtuple-like object with `low_res`, `high_res`, `mask`, and
            `output` attributes
        """
        lr, hr = self.transform(samples, **self.transform_kwargs)
        mask = self.make_mask(high_res=hr)
        output = self.make_output(samples=(lr, hr))
        return DsetTuple(low_res=lr, high_res=hr, output=output, mask=mask)


class QueueMom1(ConditionalBatchQueue):
    """Batch handling class for conditional estimation of first moment"""

    def make_output(self, samples):
        """For the 1st moment the output is simply the high_res"""
        _, hr = samples
        return hr


class QueueMom1SF(ConditionalBatchQueue):
    """Batch handling class for conditional estimation of first moment
    of subfilter velocity"""

    def make_output(self, samples):
        """
        Returns
        -------
        SF: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR from HR
        lr, hr = samples
        enhanced_lr = spatial_simple_enhancing(lr, s_enhance=self.s_enhance)
        enhanced_lr = temporal_simple_enhancing(
            enhanced_lr,
            t_enhance=self.t_enhance,
            mode=self.time_enhance_mode,
        )
        enhanced_lr = enhanced_lr[..., self.hr_features_ind]

        return hr - enhanced_lr


class QueueMom2(ConditionalBatchQueue):
    """Batch handling class for conditional estimation of second moment"""

    def make_output(self, samples):
        """
        Returns
        -------
        (HR - <HR|LR>)**2: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            HR is high-res and LR is low-res
        """
        # Remove first moment from HR and square it
        lr, hr = samples
        exo_data = self.lower_models[1]._get_hr_exo_input(hr)
        out = numpy_if_tensor(self.lower_models[1]._tf_generate(lr, exo_data))
        out = self.lower_models[1]._combine_loss_input(hr, out)
        return (hr - out) ** 2


class QueueMom2Sep(QueueMom1):
    """Batch handling class for conditional estimation of second moment
    without subtraction of first moment"""

    def make_output(self, samples):
        """
        Returns
        -------
        HR**2: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            HR is high-res
        """
        return super().make_output(samples) ** 2


class QueueMom2SF(ConditionalBatchQueue):
    """Batch handling class for conditional estimation of second moment of
    subfilter velocity."""

    def make_output(self, samples):
        """
        Returns
        -------
        (SF - <SF|LR>)**2: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR and first moment from HR and square it
        lr, hr = samples
        exo_data = self.lower_models[1]._get_hr_exo_input(hr)
        out = numpy_if_tensor(self.lower_models[1]._tf_generate(lr, exo_data))
        out = self.lower_models[1]._combine_loss_input(hr, out)
        enhanced_lr = spatial_simple_enhancing(lr, s_enhance=self.s_enhance)
        enhanced_lr = temporal_simple_enhancing(
            enhanced_lr, t_enhance=self.t_enhance, mode=self.time_enhance_mode
        )
        enhanced_lr = enhanced_lr[..., self.hr_features_ind]
        return (hr - enhanced_lr - out) ** 2


class QueueMom2SepSF(QueueMom1SF):
    """Batch of low_res, high_res and output data when learning second moment
    of subfilter vel separate from first moment"""

    def make_output(self, samples):
        """
        Returns
        -------
        SF**2: Union[np.ndarray, da.core.Array]
            4D | 5D array
            (batch_size, spatial_1, spatial_2, features)
            (batch_size, spatial_1, spatial_2, temporal, features)
            SF is subfilter, HR is high-res and LR is low-res
            SF = HR - LR
        """
        # Remove LR from HR and square it
        return super().make_output(samples) ** 2
