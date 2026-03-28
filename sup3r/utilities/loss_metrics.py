"""Content loss metrics for Sup3r"""

from typing import ClassVar

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError


class Sup3rLoss(tf.keras.losses.Loss):
    """Base class for custom sup3r loss metrics. This is meant to be used as a
    base class for loss metrics that require specific input features."""

    def __init__(self, gen_features='all', true_features=None):
        """Initialize the loss with given input features

        Parameters
        ----------
        gen_features : list | str
            List of generator output features that the loss metric will be
            calculated on.  If 'all', the loss will be calculated on all
            generator features.  Otherwise, the loss will be calculated on the
            features specified in the list.  The order of features in the list
            will be checked to determine the order of features in the generator
            output tensor.
        true_features : list | str
            List of true features that the loss metric will be calculated on.
            If None, this will be the same as gen_features. The order of
            features in the list will be checked to determine the order of
            features in the ground truth tensor.
        """
        super().__init__()
        self.gen_features = gen_features
        self.true_features = (
            true_features if true_features is not None else gen_features
        )


def tf_derivative(x, axis=1):
    """Custom derivative function for compatibility with tensorflow.

    Note
    ----
    Matches np.gradient by using the central difference approximation.

    Parameters
    ----------
    x : tf.Tensor
        (n_observations, spatial_1, spatial_2, temporal)
    axis : int
        Axis to take derivative over
    """
    if axis == 1:
        return tf.concat(
            [
                x[:, 1:2] - x[:, 0:1],
                (x[:, 2:] - x[:, :-2]) / 2,
                x[:, -1:] - x[:, -2:-1],
            ],
            axis=axis,
        )
    if axis == 2:
        return tf.concat(
            [
                x[:, :, 1:2] - x[:, :, 0:1],
                (x[:, :, 2:] - x[:, :, :-2]) / 2,
                x[:, :, -1:] - x[:, :, -2:-1],
            ],
            axis=axis,
        )
    if axis == 3:
        return tf.concat(
            [
                x[:, :, :, 1:2] - x[:, :, :, 0:1],
                (x[:, :, :, 2:] - x[:, :, :, :-2]) / 2,
                x[:, :, :, -1:] - x[:, :, :, -2:-1],
            ],
            axis=axis,
        )

    msg = (
        f'tf_derivative received axis={axis}. This is meant to compute only '
        'temporal (axis=3) or spatial (axis=1/2) derivatives for tensors '
        'of shape (n_obs, spatial_1, spatial_2, temporal)'
    )
    raise ValueError(msg)


def gaussian_kernel(x1, x2, sigma=1.0):
    """Gaussian kernel for mmd content loss

    Parameters
    ----------
    x1 : tf.tensor
        synthetic generator output
        (n_obs, spatial_1, spatial_2, temporal, features)
    x2 : tf.tensor
        high resolution data
        (n_obs, spatial_1, spatial_2, temporal, features)
    sigma : float
        Standard deviation for gaussian kernel

    Returns
    -------
    tf.tensor
        kernel output tensor

    References
    ----------
    Following MMD implementation in https://github.com/lmjohns3/theanets
    """

    # The expand dims + subtraction compares every entry for the dimension
    # prior to the expanded dimension to every other entry. So expand_dims with
    # axis=1 will compare every observation along axis=0 to every other
    # observation along axis=0.
    result = tf.exp(
        -0.5
        * tf.reduce_sum((tf.expand_dims(x1, axis=1) - x2) ** 2, axis=-1)
        / sigma**2
    )
    return result


class ExpLoss(Sup3rLoss):
    """Loss class for squared exponential difference"""

    def __call__(self, x1, x2):
        """Exponential difference loss function

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        return tf.reduce_mean(1 - tf.exp(-((x1 - x2) ** 2)))


class MmdLoss(Sup3rLoss):
    """Loss class for max mean discrepancy loss"""

    def __call__(self, x1, x2, sigma=1.0):
        """Maximum mean discrepancy (MMD) based on Gaussian kernel function
        for keras models

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        sigma : float
            standard deviation for gaussian kernel

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        mmd = tf.reduce_mean(gaussian_kernel(x1, x1, sigma))
        mmd += tf.reduce_mean(gaussian_kernel(x2, x2, sigma))
        mmd -= tf.reduce_mean(2 * gaussian_kernel(x1, x2, sigma))
        return mmd


class SpatialDerivativeLoss(Sup3rLoss):
    """Loss class to encourage accurary of spatial derivatives."""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages accuracy of spatial derivatives

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        msg = (
            f'The {self.__class__.__name__} is meant to be used on spatial or '
            'spatiotemporal data only. Received tensor(s) that are not at '
            'least 4D'
        )
        assert len(x1.shape) >= 4 and len(x2.shape) >= 4, msg

        x1_div = tf_derivative(x1, axis=1) + tf_derivative(x1, axis=2)
        x2_div = tf_derivative(x2, axis=1) + tf_derivative(x2, axis=2)

        return self.LOSS_METRIC(x1_div, x2_div)


class TemporalDerivativeLoss(Sup3rLoss):
    """Loss class to encourage accurary of temporal derivative."""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages accuracy of temporal derivative

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        msg = (
            f'The {self.__class__.__name__} is meant to be used on '
            'spatiotemporal data only. Received tensor(s) that are not 5D'
        )
        assert len(x1.shape) == 5 and len(x2.shape) == 5, msg

        x1_div = tf_derivative(x1, axis=3)
        x2_div = tf_derivative(x2, axis=3)

        return self.LOSS_METRIC(x1_div, x2_div)


class CoarseMseLoss(Sup3rLoss):
    """Loss class for coarse mse on spatial average of 5D tensor"""

    MSE_LOSS = MeanSquaredError()

    def __call__(self, x1, x2):
        """Exponential difference loss function

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """

        x1_coarse = tf.reduce_mean(x1, axis=(1, 2))
        x2_coarse = tf.reduce_mean(x2, axis=(1, 2))
        return self.MSE_LOSS(x1_coarse, x2_coarse)


class SpatialExtremesLoss(Sup3rLoss):
    """Loss class that encourages accuracy of the min/max values in the
    spatial domain. This does not include an additional MAE term"""

    MAE_LOSS = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_min = tf.reduce_min(x1, axis=(1, 2))
        x2_min = tf.reduce_min(x2, axis=(1, 2))

        x1_max = tf.reduce_max(x1, axis=(1, 2))
        x2_max = tf.reduce_max(x2, axis=(1, 2))

        mae_min = self.MAE_LOSS(x1_min, x2_min)
        mae_max = self.MAE_LOSS(x1_max, x2_max)

        return (mae_min + mae_max) / 2


class TemporalExtremesLoss(Sup3rLoss):
    """Loss class that encourages accuracy of the min/max values in the
    timeseries. This does not include an additional mae term"""

    MAE_LOSS = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_min = tf.reduce_min(x1, axis=3)
        x2_min = tf.reduce_min(x2, axis=3)

        x1_max = tf.reduce_max(x1, axis=3)
        x2_max = tf.reduce_max(x2, axis=3)

        mae_min = self.MAE_LOSS(x1_min, x2_min)
        mae_max = self.MAE_LOSS(x1_max, x2_max)

        return (mae_min + mae_max) / 2


class SpatialFftLoss(Sup3rLoss):
    """Loss class that encourages accuracy of the spatial frequency spectrum"""

    MAE_LOSS = MeanAbsoluteError()

    @staticmethod
    def _freq_weights(x):
        """Get product of squared frequencies to weight frequency amplitudes"""
        k0 = np.array([k**2 for k in range(x.shape[1])])
        k1 = np.array([k**2 for k in range(x.shape[2])])
        freqs = np.multiply.outer(k0, k1)
        freqs = tf.convert_to_tensor(freqs[np.newaxis, ..., np.newaxis])
        return tf.cast(freqs, x.dtype)

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x_hat = tf.transpose(x, perm=[3, 0, 1, 2])
        x_hat = tf.signal.fft2d(tf.cast(x_hat, tf.complex64))
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 0])
        x_hat = tf.cast(tf.abs(x_hat), x.dtype)
        x_hat = tf.math.multiply(self._freq_weights(x), x_hat)
        return tf.math.log(1 + x_hat)

    def __call__(self, x1, x2):
        """Custom content loss that encourages frequency domain accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_hat = self._fft(x1)
        x2_hat = self._fft(x2)
        return self.MAE_LOSS(x1_hat, x2_hat)


class SpatiotemporalFftLoss(Sup3rLoss):
    """Loss class that encourages accuracy of the spatiotemporal frequency
    spectrum"""

    MAE_LOSS = MeanAbsoluteError()

    @staticmethod
    def _freq_weights(x):
        """Get product of squared frequencies to weight frequency amplitudes"""
        k0 = np.array([k**2 for k in range(x.shape[1])])
        k1 = np.array([k**2 for k in range(x.shape[2])])
        f = np.array([f**2 for f in range(x.shape[3])])
        freqs = np.multiply.outer(k0, k1)
        freqs = np.multiply.outer(freqs, f)
        freqs = tf.convert_to_tensor(freqs[np.newaxis, ..., np.newaxis])
        return tf.cast(freqs, x.dtype)

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x_hat = tf.transpose(x, perm=[4, 0, 1, 2, 3])
        x_hat = tf.signal.fft3d(tf.cast(x_hat, tf.complex64))
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 4, 0])
        x_hat = tf.cast(tf.abs(x_hat), x.dtype)
        x_hat = tf.math.multiply(self._freq_weights(x), x_hat)
        return tf.math.log(1 + x_hat)

    def __call__(self, x1, x2):
        """Custom content loss that encourages frequency domain accuracy

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x1_hat = self._fft(x1)
        x2_hat = self._fft(x2)
        return self.MAE_LOSS(x1_hat, x2_hat)


class LowResLoss(Sup3rLoss):
    """Content loss that is calculated by coarsening the synthetic and true
    high-resolution data pairs and then performing the pointwise content loss
    on the low-resolution fields"""

    EX_LOSS_METRICS: ClassVar = {
        'SpatialExtremesLoss': SpatialExtremesLoss,
        'TemporalExtremesLoss': TemporalExtremesLoss,
    }

    def __init__(
        self,
        s_enhance=1,
        t_enhance=1,
        t_method='average',
        tf_loss='MeanSquaredError',
        ex_loss=None,
    ):
        """Initialize the loss with given weight

        Parameters
        ----------
        s_enhance : int
            factor by which to coarsen spatial dimensions. 1 will keep the
            spatial axes as high-res
        t_enhance : int
            factor by which to coarsen temporal dimension. 1 will keep the
            temporal axes as high-res
        t_method : str
            Accepted options: [subsample, average]
            Subsample will take every t_enhance-th time step, average will
            average over t_enhance time steps
        tf_loss : str
            The tensorflow loss function to operate on the low-res fields. Must
            be the name of a loss class that can be retrieved from
            ``tf.keras.losses`` e.g., "MeanSquaredError" or "MeanAbsoluteError"
        ex_loss : None | str
            Optional additional loss metric evaluating the spatial or temporal
            extremes of the high-res data. Can be "SpatialExtremesLoss" or
            "TemporalExtremesLoss" (keys in ``EX_LOSS_METRICS``).
        """

        super().__init__()
        self._s_enhance = s_enhance
        self._t_enhance = t_enhance
        self._t_method = str(t_method).casefold()
        self._tf_loss = getattr(tf.keras.losses, tf_loss)()
        self._ex_loss = ex_loss
        if self._ex_loss is not None:
            self._ex_loss = self.EX_LOSS_METRICS[self._ex_loss]()

    def _s_coarsen_4d_tensor(self, tensor):
        """Perform spatial coarsening on a 4D tensor of shape
        (n_obs, spatial_1, spatial_2, features)"""
        shape = tensor.shape
        tensor = tf.reshape(
            tensor,
            (
                shape[0],
                shape[1] // self._s_enhance,
                self._s_enhance,
                shape[2] // self._s_enhance,
                self._s_enhance,
                shape[3],
            ),
        )
        tensor = tf.math.reduce_sum(tensor, axis=(2, 4)) / self._s_enhance**2
        return tensor

    def _s_coarsen_5d_tensor(self, tensor):
        """Perform spatial coarsening on a 5D tensor of shape
        (n_obs, spatial_1, spatial_2, time, features)"""
        shape = tensor.shape
        tensor = tf.reshape(
            tensor,
            (
                shape[0],
                shape[1] // self._s_enhance,
                self._s_enhance,
                shape[2] // self._s_enhance,
                self._s_enhance,
                shape[3],
                shape[4],
            ),
        )
        tensor = tf.math.reduce_sum(tensor, axis=(2, 4)) / self._s_enhance**2
        return tensor

    def _t_coarsen_sample(self, tensor):
        """Perform temporal subsampling on a 5D tensor of shape
        (n_obs, spatial_1, spatial_2, time, features)"""
        assert len(tensor.shape) == 5
        tensor = tensor[:, :, :, :: self._t_enhance, :]
        return tensor

    def _t_coarsen_avg(self, tensor):
        """Perform temporal coarsening on a 5D tensor of shape
        (n_obs, spatial_1, spatial_2, time, features)"""
        shape = tensor.shape
        assert len(shape) == 5
        tensor = tf.reshape(
            tensor,
            (shape[0], shape[1], shape[2], -1, self._t_enhance, shape[4]),
        )
        tensor = tf.math.reduce_sum(tensor, axis=4) / self._t_enhance
        return tensor

    def __call__(self, x1, x2):
        """Custom content loss calculated on re-coarsened low-res fields

        Parameters
        ----------
        x1 : tf.tensor
            Synthetic high-res generator output, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            True high resolution data, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """

        assert x1.shape == x2.shape
        s_only = len(x1.shape) == 4

        ex_loss = tf.constant(0, dtype=x1.dtype)
        if self._ex_loss is not None:
            ex_loss = self._ex_loss(x1, x2)

        if self._s_enhance > 1 and s_only:
            x1 = self._s_coarsen_4d_tensor(x1)
            x2 = self._s_coarsen_4d_tensor(x2)

        elif self._s_enhance > 1 and not s_only:
            x1 = self._s_coarsen_5d_tensor(x1)
            x2 = self._s_coarsen_5d_tensor(x2)

        if self._t_enhance > 1 and self._t_method == 'average':
            x1 = self._t_coarsen_avg(x1)
            x2 = self._t_coarsen_avg(x2)

        if self._t_enhance > 1 and self._t_method == 'subsample':
            x1 = self._t_coarsen_sample(x1)
            x2 = self._t_coarsen_sample(x2)

        return self._tf_loss(x1, x2) + ex_loss


class PerceptualLoss(Sup3rLoss):
    """Perceptual loss that is calculated as MSE between feature maps of
    ground truth and synthetic data"""

    def __init__(self, layer_names=None):
        """
        Parameters
        ----------
        layer_names : list | None
            List of layer names in VGG16 to use to extract feature maps from
            ground truth and synthetic data. Defaults to ['block1_conv2',
            'block2_conv2']
        """
        super().__init__()
        # VGG16 for perceptual loss
        vgg = VGG16(weights='imagenet', include_top=False)
        vgg.trainable = False
        self.layer_names = layer_names
        if self.layer_names is None:
            self.layer_names = ['block1_conv2', 'block2_conv2']
        vgg_outputs = [vgg.get_layer(name).output for name in self.layer_names]
        self.feature_extractor = tf.keras.Model(
            inputs=vgg.input, outputs=vgg_outputs
        )

    def _feature_loss(self, x1, x2):
        """Calculate loss for a single feature. e.g. A single pair of tensors
        each with only 3 channels"""
        x1 = preprocess_input(x1)
        x2 = preprocess_input(x2)
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        if len(self.layer_names) == 1:
            x1 = [x1]
            x2 = [x2]
        loss = 0
        for x1_f, x2_f in zip(x1, x2):
            loss += tf.reduce_mean(tf.square(x1_f - x2_f))
        return loss

    def __call__(self, x1, x2):
        """Perceptual loss calculated on true and synthetic feature maps

        Parameters
        ----------
        x1 : tf.tensor
            Synthetic high-res generator output, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            True high resolution data, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        if len(x1.shape) == 5:
            new_shape = (
                x1.shape[0] * x1.shape[3],
                x1.shape[1],
                x1.shape[2],
                x1.shape[-1],
            )
            x1 = tf.reshape(x1, new_shape)
            x2 = tf.reshape(x2, new_shape)

        losses = []
        for i in range(x1.shape[-1]):
            x1_f = x1[..., i]
            x2_f = x2[..., i]

            # VGG input needs 3 RGB channels
            x1_f = tf.stack([x1_f] * 3, axis=-1)
            x2_f = tf.stack([x2_f] * 3, axis=-1)

            losses.append(self._feature_loss(x1_f, x2_f))

        return tf.reduce_mean(losses)


class SlicedWassersteinLoss(Sup3rLoss):
    """Loss class for sliced wasserstein distance loss"""

    def __init__(self, n_projections=1024):
        """Parameters
        ----------
        n_projections : int
            number of random 1D projections to use

        Note:
        ----
        Experimentally, we get stability in the SW metric when n_projections
        is at least 30% of the number of projection dimensions, which for us
        is HWT. This might be computationally expensive for large
        spatial/temporal sizes so we default to 1024.
        """
        super().__init__()
        self._n_projections = n_projections

    def __call__(self, x1, x2):
        """Sliced Wasserstein distance based on random 1D projections

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        msg = (
            'The SlicedWassersteinLoss is meant to be used on spatial or '
            'spatiotemporal data only. Received tensor(s) that are not 4D '
            'or 5D'
        )
        assert len(x1.shape) in {4, 5}, msg
        if len(x1.shape) == 4:
            x1 = tf.expand_dims(x1, axis=3)
            x2 = tf.expand_dims(x2, axis=3)

        B, H, W, T, C = x1.shape

        # Flatten only spatial/time dims → (B, HWT, C)
        x1_flat = tf.reshape(x1, (B, H * W * T, C))
        x2_flat = tf.reshape(x2, (B, H * W * T, C))

        # Random projection directions over HWT only
        proj = tf.random.normal((self._n_projections, H * W * T))
        proj = tf.math.l2_normalize(proj, axis=-1)  # normalize

        # Project spatial dimensions → (num_proj, B, C)
        # matmul: (num_proj, HWT) @ (B, HWT, C) → (B, num_proj, C)
        x1_proj = proj @ x1_flat
        x2_proj = proj @ x2_flat

        # Sort each projection's distribution along the projection dimension
        x1_sorted = tf.sort(x1_proj, axis=1)
        x2_sorted = tf.sort(x2_proj, axis=1)

        return tf.reduce_mean((x1_sorted - x2_sorted) ** 2)


class MaterialDerivativeLoss(Sup3rLoss):
    """Loss class for the material derivative. This is the left hand side of
    the Navier-Stokes equation and is equal to internal + external forces
    divided by density in general. Under certain simplifying assumptions, this
    is equal to zero.

    References
    ----------
    https://en.wikipedia.org/wiki/Material_derivative
    """

    LOSS_METRIC = MeanAbsoluteError()

    def __init__(self, gen_features):
        super().__init__(gen_features=gen_features)
        self.u_inds = [
            i for i, f in enumerate(gen_features) if f.startswith('u_')
        ]
        self.v_inds = [
            i for i, f in enumerate(gen_features) if f.startswith('v_')
        ]
        self.u_heights = [
            f.split('_')[1] for f in gen_features if f.startswith('u_')
        ]
        self.v_heights = [
            f.split('_')[1] for f in gen_features if f.startswith('v_')
        ]
        assert len(self.u_inds) == len(self.v_inds), (
            'The number of u and v components must be equal for '
            f'MaterialDerivativeLoss. Found {len(self.u_inds)} u components '
            f'and {len(self.v_inds)} v components.'
        )
        msg = (
            'The u and v components must be at the same hub heights for '
            f'MaterialDerivativeLoss. Found u components at {self.u_heights} '
            f'and v components at {self.v_heights}.'
        )
        assert all(
            uh == vh for uh, vh in zip(self.u_heights, self.v_heights)
        ), msg

    def _compute_md(self, x, feature):
        """Compute material derivative for the feature given by the index fidx.

        Parameters
        ----------
        x : tf.tensor
            synthetic output or high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)
        feature : str
            Feature to compute material derivative for.
        """
        # df/dt
        height = feature.split('_')[1]
        fidx = self.gen_features.index(feature)
        uidx = self.u_inds[self.u_heights.index(height)]
        vidx = self.v_inds[self.v_heights.index(height)]
        x_div = tf_derivative(x[..., fidx], axis=3)
        # u * df/dx
        x_div += tf.math.multiply(
            x[..., uidx], tf_derivative(x[..., fidx], axis=1)
        )
        # v * df/dy
        x_div += tf.math.multiply(
            x[..., vidx], tf_derivative(x[..., fidx], axis=2)
        )

        return x_div

    def __call__(self, x1, x2):
        """Custom content loss that encourages accuracy of the material
        derivative.

        Parameters
        ----------
        x1 : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        x2 : tf.tensor
            high resolution data
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        msg = (
            f'The {self.__class__.__name__} is meant to be used on '
            'spatiotemporal data only. Received tensor(s) that are not 5D'
        )
        assert len(x1.shape) == 5 and len(x2.shape) == 5, msg

        x1_div = tf.stack(
            [self._compute_md(x1, feature) for feature in self.gen_features]
        )
        x2_div = tf.stack(
            [self._compute_md(x2, feature) for feature in self.gen_features]
        )

        return self.LOSS_METRIC(x1_div, x2_div)


class GeothermalConductiveHeatTransferLoss(Sup3rLoss):
    """Deviation from three-dimensional conductive heat transfer loss"""

    LOSS_METRIC = MeanSquaredError()

    def __init__(
        self,
        dx,
        dy,
        depths=range(0, 8000, 1000),
        temperature_prefix='t',
        heat_flux_prefix='q',
        thermal_conductivity_prefix='k',
    ):
        """Initialize the loss with the generated t, q, and k features

        Parameters
        ----------
        gen_features : list | str
            List of generator output features used to compute the
            three-dimensional conductive heat transfer loss. The
            expected features are temperature, heat-flow, and thermal
            conductivity channels named as ``t_<depth>m``,
            ``q_<depth>m``, and ``k_<depth>m``. Depths are discovered
            dynamically from this input and aligned by strict
            intersection.
        """
        self.t_inds, self.q_inds, self.k_inds = [], [], []
        self.dx, self.dy = dx, dy

        depths, self.dz = self._validate_depths(depths)
        gen_features = self._collect_gen_features(
            depths,
            temperature_prefix,
            heat_flux_prefix,
            thermal_conductivity_prefix,
        )
        super().__init__(gen_features=gen_features, true_features=None)

    @staticmethod
    def _validate_depths(depths):
        depths = list(depths)
        msg = (
            'GeothermalConductiveHeatTransferLoss requires at least two '
            'common depth across t_* and k_* features to compute '
            f'vertical derivative. Received depths: {depths}'
        )
        assert len(depths) > 1, msg

        depths = sorted(depths)
        msg = (
            'GeothermalConductiveHeatTransferLoss requires a depth of 0m to '
            'be present in the input features to compute the total heat '
            'generation (integral) from the surface to each depth. '
            f'Received depths: {depths}'
        )
        assert depths[0] == 0, msg

        dz_steps = np.diff(depths)
        msg = (
            'GeothermalConductiveHeatTransferLoss requires uniformly spaced '
            f'depth channels. Received depths: {depths}'
        )
        assert np.allclose(dz_steps, dz_steps[0]), msg
        return depths, float(dz_steps[0])

    def _collect_gen_features(self, depths, tp, hfp, tcp):
        """Collect the expected t, q, and k feature datasets + inds"""
        gen_features = []
        for depth in depths:
            self.t_inds.append(len(gen_features))
            gen_features.append(f'{tp}_{depth}m')

            self.k_inds.append(len(gen_features))
            gen_features.append(f'{tcp}_{depth}m')

        self.q_inds = [len(gen_features)]
        gen_features.append(f'{hfp}_0m')
        return gen_features

    def _get_feature_tensors(self, x):
        """Extract stacked temperature/heat-flow/conductivity tensors"""
        t = tf.stack([x[..., i] for i in self.t_inds], axis=-1)
        q = tf.stack([x[..., i] for i in self.q_inds], axis=-1)
        k = tf.stack([x[..., i] for i in self.k_inds], axis=-1)
        return t, q, k

    def _compute_heat_transfer_residual(self, x):
        """Compute heat transfer residual to be penalized towards zero"""
        t, q, k = self._get_feature_tensors(x)

        t = _reshape_depth_feature_for_vertical_derivative(t)  # C
        q = _reshape_depth_feature_for_vertical_derivative(q) / 1000.0  # W/m^2
        k = _reshape_depth_feature_for_vertical_derivative(k)  # W/m/K

        dx = tf.cast(self.dx, t.dtype)
        dy = tf.cast(self.dy, t.dtype)
        dz = tf.cast(self.dz, t.dtype)

        dtdx = tf_derivative(t, axis=2) / dx
        dtdy = tf_derivative(t, axis=1) / dy
        dtdz = tf_derivative(t, axis=3) / dz

        qc = k * (dtdx + dtdy + dtdz)

        g_dot = tf_derivative(k * dtdx, axis=2) / dx
        g_dot += tf_derivative(k * dtdy, axis=1) / dy
        g_dot += tf_derivative(k * dtdz, axis=3) / dz

        g_dot_mid = 0.5 * (g_dot[..., 1:] + g_dot[..., :-1])
        int_g = tf.concat(
            [
                tf.zeros_like(g_dot[..., :1]),
                tf.math.cumsum(g_dot_mid * dz, axis=3),
            ],
            axis=3,
        )
        return -qc + q + int_g

    def __call__(self, x_gen, __):
        """

        Parameters
        ----------
        x_gen : tf.tensor
            Synthetic generator output used to compute heat transfer
            residual. Shape must be either:
            (n_observations, spatial_1, spatial_2, features) or
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_true : tf.tensor
            Ground truth data (unused).

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        msg = (
            f'The {self.__class__.__name__} is meant to be used on spatial '
            'or spatiotemporal data only. Received tensor(s) that are not '
            '4D or 5D'
        )
        assert len(x_gen.shape) in {4, 5}, msg

        expr = self._compute_heat_transfer_residual(x_gen)
        return self.LOSS_METRIC(tf.zeros_like(expr), expr)


class GeothermalPositiveTemperatureGradientLoss(Sup3rLoss):
    """Positive geothermal gradient loss

    The expected feature is temperature, named as ``t_<depth>m``.
    Depths are discovered dynamically from ``input_features``.
    """

    LOSS_METRIC = MeanSquaredError()

    def __init__(self, depths=range(0, 8000, 1000), temperature_prefix='t'):
        """Initialize the loss with the generated temperature feature

        Parameters
        ----------
        gen_features : list | str
            List of generator output features used to compute the
            three-dimensional conductive heat transfer loss. The
            expected feature is temperature, named as ``t_<depth>m``.
            Depths are discovered dynamically from this input.
        """
        self.t_inds = []
        depths = self._validate_depths(depths)

        gen_features = []
        for depth in depths:
            self.t_inds.append(len(gen_features))
            gen_features.append(f'{temperature_prefix}_{depth}m')

        super().__init__(gen_features=gen_features, true_features=None)

    @staticmethod
    def _validate_depths(depths):
        depths = list(depths)

        msg = (
            'GeothermalPositiveTemperatureGradientLoss requires at least two '
            'common depth across t_* features to compute vertical derivative. '
            f'Received depths: {depths}'
        )
        assert len(depths) > 1, msg

        return depths

    def _compute_temperature_gradient(self, x):
        """Compute temp gradient be penalized for being negative"""
        t = tf.stack([x[..., i] for i in self.t_inds], axis=-1)
        t = _reshape_depth_feature_for_vertical_derivative(t)

        # Not a true derivative (missing divide by dz), but we only care
        # about the sign, so this is sufficient
        dt = tf_derivative(t, axis=3)
        return tf.math.maximum(-1 * dt, tf.constant([0.0], dt.dtype))

    def __call__(self, x_gen, __):
        """

        Parameters
        ----------
        x_gen : tf.tensor
            Synthetic generator output used to compute heat transfer
            residual. Shape must be either:
            (n_observations, spatial_1, spatial_2, features) or
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_true : tf.tensor
            Ground truth data (unused).

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        msg = (
            f'The {self.__class__.__name__} is meant to be used on spatial '
            'or spatiotemporal data only. Received tensor(s) that are not '
            '4D or 5D'
        )
        assert len(x_gen.shape) in {4, 5}, msg

        temp_grads = self._compute_temperature_gradient(x_gen)
        return self.LOSS_METRIC(tf.zeros_like(temp_grads), temp_grads)


class GeothermalMohoBCLoss(Sup3rLoss):
    """Heat flow across Moho layer boundary condition loss

    This loss helps satisfy the condition that the predicted heat flow
    across depths is greater than the minimum heat flow at the Moho
    layer.
    """

    LOSS_METRIC = MeanSquaredError()

    def __init__(
        self,
        heat_flow_features,
        moho_gradient_layer='',
        upper_mantle_thermal_conductivity=4.0,
    ):
        """Initialize the loss with the appropriate features

        Parameters
        ----------
        gen_features : list | str
            List of generator output features used to compute the
            three-dimensional conductive heat transfer loss. The
            expected feature is temperature, named as ``t_<depth>m``.
            Depths are discovered dynamically from this input.
        """
        self.lambda_um = upper_mantle_thermal_conductivity
        super().__init__(
            gen_features=list(heat_flow_features),
            true_features=[moho_gradient_layer],
        )

    def __call__(self, x_gen, x_moho):
        """

        Parameters
        ----------
        x_gen : tf.tensor
            Synthetic generator output of heat transfer values (mW/m^2).
            Shape must be either:
            (n_observations, spatial_1, spatial_2, features) or
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_true : tf.tensor
            Temperature gradient over Moho layer (K/km).

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """

        heat_flow_watts = x_gen * 1000
        temp_grad_K_per_m = x_moho / 1000

        residuals = tf.math.maximum(
            self.lambda_um * temp_grad_K_per_m - heat_flow_watts,
            tf.constant([0.0], x_gen.dtype),
        )
        return self.LOSS_METRIC(tf.zeros_like(residuals), residuals)


class GeothermalObsLoss(Sup3rLoss):
    """Geothermal loss for observed quantities"""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x1, x2):
        """Geothermal observed quantity loss"""
        check = x1.shape[-1] == len(self.gen_features)
        check &= x2.shape[-1] == len(self.true_features)
        msg = (
            f'Number of features in `x1`: {x1.shape[-1]} must match the '
            f'length of `gen_features`: {len(self.gen_features)}, `x2`: '
            f'{x2.shape[-1]} must match the length of `true_features`: '
            f'{len(self.true_features)}'
        )
        assert check, msg

        mask = tf.math.logical_not(tf.math.is_nan(x2))
        x1m = tf.boolean_mask(x1, mask)
        x2m = tf.boolean_mask(x2, mask)

        return (
            tf.constant(0, dtype=x1.dtype)
            if tf.math.reduce_all(tf.math.is_nan(x2m))
            else self.LOSS_METRIC(x1m, x2m)
        )


def _reshape_depth_feature_for_vertical_derivative(x):
    """Reshape a stacked depth tensor for use with tf_derivative.

    Parameters
    ----------
    x : tf.Tensor
        Either 4D tensor (n_obs, s1, s2, depth) or 5D tensor
        (n_obs, s1, s2, time, depth).

    Returns
    -------
    tf.Tensor
        4D Tensor where the last three dimensions are
        (s1, s2, depth). First dimension is either ``n_obs`` or
        ``n_obs * time``.
    """
    if len(x.shape) == 4:
        return x

    if len(x.shape) == 5:
        shape = tf.shape(x)
        return tf.reshape(
            x, (shape[0] * shape[3], shape[1], shape[2], shape[4])
        )

    msg = (
        'Geothermal losses expects 4D or 5D tensors before vertical '
        f'reshaping, received {len(x.shape)}D tensor.'
    )
    raise ValueError(msg)
