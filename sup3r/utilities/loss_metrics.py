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


def gaussian_kernel(x_true, x_gen, sigma=1.0):
    """Gaussian kernel for mmd content loss

    Parameters
    ----------
    x_true : tf.tensor
        high resolution ground truth data
        (n_obs, spatial_1, spatial_2, temporal, features)
    x_gen : tf.tensor
        synthetic generator output
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
        * tf.reduce_sum((tf.expand_dims(x_true, axis=1) - x_gen) ** 2, axis=-1)
        / sigma**2
    )
    return result


class ExpLoss(Sup3rLoss):
    """Loss class for squared exponential difference"""

    def __call__(self, x_true, x_gen):
        """Exponential difference loss function

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        return tf.reduce_mean(1 - tf.exp(-((x_true - x_gen) ** 2)))


class MmdLoss(Sup3rLoss):
    """Loss class for max mean discrepancy loss"""

    def __call__(self, x_true, x_gen, sigma=1.0):
        """Maximum mean discrepancy (MMD) based on Gaussian kernel function
        for keras models

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)
        sigma : float
            standard deviation for gaussian kernel

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        mmd = tf.reduce_mean(gaussian_kernel(x_true, x_true, sigma))
        mmd += tf.reduce_mean(gaussian_kernel(x_gen, x_gen, sigma))
        mmd -= tf.reduce_mean(2 * gaussian_kernel(x_true, x_gen, sigma))
        return mmd


class SpatialDerivativeLoss(Sup3rLoss):
    """Loss class to encourage accurary of spatial derivatives."""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages accuracy of spatial derivatives

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
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
        assert len(x_true.shape) >= 4 and len(x_gen.shape) >= 4, msg

        x_true_div = tf_derivative(x_true, axis=1) + tf_derivative(
            x_true, axis=2
        )
        x_gen_div = tf_derivative(x_gen, axis=1) + tf_derivative(x_gen, axis=2)

        return self.LOSS_METRIC(x_true_div, x_gen_div)


class TemporalDerivativeLoss(Sup3rLoss):
    """Loss class to encourage accurary of temporal derivative."""

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages accuracy of temporal derivative

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
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
        assert len(x_true.shape) == 5 and len(x_gen.shape) == 5, msg

        x_true_div = tf_derivative(x_true, axis=3)
        x_gen_div = tf_derivative(x_gen, axis=3)

        return self.LOSS_METRIC(x_true_div, x_gen_div)


class CoarseMseLoss(Sup3rLoss):
    """Loss class for coarse mse on spatial average of 5D tensor"""

    MSE_LOSS = MeanSquaredError()

    def __call__(self, x_true, x_gen):
        """Exponential difference loss function

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """

        x_true_coarse = tf.reduce_mean(x_true, axis=(1, 2))
        x_gen_coarse = tf.reduce_mean(x_gen, axis=(1, 2))
        return self.MSE_LOSS(x_true_coarse, x_gen_coarse)


class SpatialExtremesLoss(Sup3rLoss):
    """Loss class that encourages accuracy of the min/max values in the
    spatial domain. This does not include an additional MAE term"""

    MAE_LOSS = MeanAbsoluteError()

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x_true_min = tf.reduce_min(x_true, axis=(1, 2))
        x_gen_min = tf.reduce_min(x_gen, axis=(1, 2))

        x_true_max = tf.reduce_max(x_true, axis=(1, 2))
        x_gen_max = tf.reduce_max(x_gen, axis=(1, 2))

        mae_min = self.MAE_LOSS(x_true_min, x_gen_min)
        mae_max = self.MAE_LOSS(x_true_max, x_gen_max)

        return (mae_min + mae_max) / 2


class TemporalExtremesLoss(Sup3rLoss):
    """Loss class that encourages accuracy of the min/max values in the
    timeseries. This does not include an additional mae term"""

    MAE_LOSS = MeanAbsoluteError()

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages temporal min/max accuracy

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x_true_min = tf.reduce_min(x_true, axis=3)
        x_gen_min = tf.reduce_min(x_gen, axis=3)

        x_true_max = tf.reduce_max(x_true, axis=3)
        x_gen_max = tf.reduce_max(x_gen, axis=3)

        mae_min = self.MAE_LOSS(x_true_min, x_gen_min)
        mae_max = self.MAE_LOSS(x_true_max, x_gen_max)

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

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages frequency domain accuracy

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x_true_hat = self._fft(x_true)
        x_gen_hat = self._fft(x_gen)
        return self.MAE_LOSS(x_true_hat, x_gen_hat)


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

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages frequency domain accuracy

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor with loss value
        """
        x_true_hat = self._fft(x_true)
        x_gen_hat = self._fft(x_gen)
        return self.MAE_LOSS(x_true_hat, x_gen_hat)


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

    def __call__(self, x_true, x_gen):
        """Custom content loss calculated on re-coarsened low-res fields

        Parameters
        ----------
        x_true : tf.tensor
            True high resolution data, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            Synthetic high-res generator output, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """

        assert x_true.shape == x_gen.shape
        s_only = len(x_true.shape) == 4

        ex_loss = tf.constant(0, dtype=x_true.dtype)
        if self._ex_loss is not None:
            ex_loss = self._ex_loss(x_true, x_gen)

        if self._s_enhance > 1 and s_only:
            x_true = self._s_coarsen_4d_tensor(x_true)
            x_gen = self._s_coarsen_4d_tensor(x_gen)

        elif self._s_enhance > 1 and not s_only:
            x_true = self._s_coarsen_5d_tensor(x_true)
            x_gen = self._s_coarsen_5d_tensor(x_gen)

        if self._t_enhance > 1 and self._t_method == 'average':
            x_true = self._t_coarsen_avg(x_true)
            x_gen = self._t_coarsen_avg(x_gen)

        if self._t_enhance > 1 and self._t_method == 'subsample':
            x_true = self._t_coarsen_sample(x_true)
            x_gen = self._t_coarsen_sample(x_gen)

        return self._tf_loss(x_true, x_gen) + ex_loss


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

    def _feature_loss(self, x_true, x_gen):
        """Calculate loss for a single feature. e.g. A single pair of tensors
        each with only 3 channels"""
        x_true = preprocess_input(x_true)
        x_gen = preprocess_input(x_gen)
        x_true = self.feature_extractor(x_true)
        x_gen = self.feature_extractor(x_gen)
        if len(self.layer_names) == 1:
            x_true = [x_true]
            x_gen = [x_gen]
        loss = 0
        for x_true_f, x_gen_f in zip(x_true, x_gen):
            loss += tf.reduce_mean(tf.square(x_true_f - x_gen_f))
        return loss

    def __call__(self, x_true, x_gen):
        """Perceptual loss calculated on true and synthetic feature maps

        Parameters
        ----------
        x_true : tf.tensor
            True high resolution data, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            Synthetic high-res generator output, shape is either of these:
            (n_obs, spatial_1, spatial_2, features)
            (n_obs, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        if len(x_true.shape) == 5:
            new_shape = (
                x_true.shape[0] * x_true.shape[3],
                x_true.shape[1],
                x_true.shape[2],
                x_true.shape[-1],
            )
            x_true = tf.reshape(x_true, new_shape)
            x_gen = tf.reshape(x_gen, new_shape)

        losses = []
        for i in range(x_true.shape[-1]):
            x_true_f = x_true[..., i]
            x_gen_f = x_gen[..., i]

            # VGG input needs 3 RGB channels
            x_true_f = tf.stack([x_true_f] * 3, axis=-1)
            x_gen_f = tf.stack([x_gen_f] * 3, axis=-1)

            losses.append(self._feature_loss(x_true_f, x_gen_f))

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

    def __call__(self, x_true, x_gen):
        """Sliced Wasserstein distance based on random 1D projections

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        assert len(x_gen.shape) in {4, 5} and len(x_true.shape) in {4, 5}, (
            f'The {self.__class__.__name__} is meant to be used on spatial or '
            'spatiotemporal data only. Received tensor(s) that are not 4/5D'
        )
        if len(x_true.shape) == 4:
            x_true = tf.expand_dims(x_true, axis=3)
            x_gen = tf.expand_dims(x_gen, axis=3)

        B, H, W, T, C = x_true.shape

        # Flatten only spatial/time dims → (B, HWT, C)
        x_true_flat = tf.reshape(x_true, (B, H * W * T, C))
        x_gen_flat = tf.reshape(x_gen, (B, H * W * T, C))

        # Random projection directions over HWT only
        proj = tf.random.normal((self._n_projections, H * W * T))
        proj = tf.math.l2_normalize(proj, axis=-1)  # normalize

        # Project spatial dimensions → (num_proj, B, C)
        # matmul: (num_proj, HWT) @ (B, HWT, C) → (B, num_proj, C)
        x_true_proj = proj @ x_true_flat
        x_gen_proj = proj @ x_gen_flat

        # Sort each projection's distribution along the projection dimension
        x_true_sorted = tf.sort(x_true_proj, axis=1)
        x_gen_sorted = tf.sort(x_gen_proj, axis=1)

        return tf.reduce_mean((x_true_sorted - x_gen_sorted) ** 2)


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

    def __call__(self, x_true, x_gen):
        """Custom content loss that encourages accuracy of the material
        derivative.

        Parameters
        ----------
        x_true : tf.tensor
            high resolution ground truth data
            (n_observations, spatial_1, spatial_2, temporal, features)
        x_gen : tf.tensor
            synthetic generator output
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
        assert len(x_true.shape) == 5 and len(x_gen.shape) == 5, msg

        x_true_div = tf.stack([
            self._compute_md(x_true, feature) for feature in self.gen_features
        ])
        x_gen_div = tf.stack([
            self._compute_md(x_gen, feature) for feature in self.gen_features
        ])

        return self.LOSS_METRIC(x_true_div, x_gen_div)


class GeothermalPhysicsLoss(Sup3rLoss):
    """Physics based loss for Geothermal applications

    TODO: Fill in call with appropriate physics equations. This is currently
    just a dummy equation for testing.
    """

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x_true, x_gen):
        """Geothermal physics loss"""
        check = x_true.shape[-1] == len(self.true_features)
        check &= x_gen.shape[-1] == len(self.gen_features)
        msg = (
            f'Number of features in `x_true`: {x_true.shape[-1]} must match '
            f'the length of `true_features`: {len(self.true_features)}, '
            f'`x_gen`: {x_gen.shape[-1]} must match the length of '
            f'`gen_features`: {len(self.gen_features)}'
        )
        assert check, msg

        return self.LOSS_METRIC(x_true, x_gen)


class GeothermalPhysicsLossWithObs(Sup3rLoss):
    """Physics based loss for Geothermal applications

    TODO: Fill in call with appropriate physics equations. This is currently
    just a dummy equation for testing.
    """

    LOSS_METRIC = MeanAbsoluteError()

    def __call__(self, x_true, x_gen):
        """Geothermal physics loss"""
        check = x_true.shape[-1] == len(self.true_features)
        check &= x_gen.shape[-1] == len(self.gen_features)
        msg = (
            f'Number of features in `x_true`: {x_true.shape[-1]} must match '
            f'the length of `true_features`: {len(self.true_features)}, '
            f'`x_gen`: {x_gen.shape[-1]} must match the length of '
            f'`gen_features`: {len(self.gen_features)}'
        )
        assert check, msg

        mask = tf.math.logical_not(tf.math.is_nan(x_true))
        x_true_m = tf.boolean_mask(x_true, mask)
        x_gen_m = tf.boolean_mask(x_gen, mask)

        physics_loss = tf.constant(1e-3, dtype=x_true.dtype)
        obs_loss = (
            tf.constant(0, dtype=x_true.dtype)
            if tf.math.reduce_all(tf.math.is_nan(x_true_m))
            else self.LOSS_METRIC(x_true_m, x_gen_m)
        )
        return physics_loss + obs_loss
