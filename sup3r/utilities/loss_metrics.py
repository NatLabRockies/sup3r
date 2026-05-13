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


def _assert_rank_in(x, ranks, message):
    """TensorFlow rank assertion that is safe under tf.function tracing."""
    rank = x.shape.rank
    if rank is not None:
        if rank not in ranks:
            raise ValueError(message)
        return x

    assertion = tf.debugging.assert_equal(
        tf.reduce_any(
            tf.equal(tf.rank(x), tf.constant(ranks, dtype=tf.int32))
        ),
        True,
        message=message,
    )
    with tf.control_dependencies([assertion]):
        return tf.identity(x)


class ExpLoss(Sup3rLoss):
    """Loss class for squared exponential difference"""

    @staticmethod
    @tf.function
    def call(x_true, x_gen):
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

    @staticmethod
    @tf.function
    def call(x_true, x_gen, sigma=1.0):
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
        dtype = tf.as_dtype(tf.keras.backend.floatx())
        x_true = tf.cast(x_true, dtype)
        x_gen = tf.cast(x_gen, dtype)
        mmd = tf.reduce_mean(gaussian_kernel(x_true, x_true, sigma))
        mmd += tf.reduce_mean(gaussian_kernel(x_gen, x_gen, sigma))
        mmd -= tf.reduce_mean(2 * gaussian_kernel(x_true, x_gen, sigma))
        return mmd


class SpatialDerivativeLoss(Sup3rLoss):
    """Loss class to encourage accurary of spatial derivatives."""

    LOSS_METRIC = MeanAbsoluteError()

    @tf.function
    def call(self, x_true, x_gen):
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
        tf.debugging.assert_greater_equal(tf.rank(x_true), 4, message=msg)
        tf.debugging.assert_greater_equal(tf.rank(x_gen), 4, message=msg)

        x_true_div = tf_derivative(x_true, axis=1) + tf_derivative(
            x_true, axis=2
        )
        x_gen_div = tf_derivative(x_gen, axis=1) + tf_derivative(x_gen, axis=2)

        return self.LOSS_METRIC(x_true_div, x_gen_div)


class TemporalDerivativeLoss(Sup3rLoss):
    """Loss class to encourage accurary of temporal derivative."""

    LOSS_METRIC = MeanAbsoluteError()

    @tf.function
    def call(self, x_true, x_gen):
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
        x_true = _assert_rank_in(x_true, (5,), msg)
        x_gen = _assert_rank_in(x_gen, (5,), msg)

        x_true_div = tf_derivative(x_true, axis=3)
        x_gen_div = tf_derivative(x_gen, axis=3)

        return self.LOSS_METRIC(x_true_div, x_gen_div)


class CoarseMseLoss(Sup3rLoss):
    """Loss class for coarse mse on spatial average of 5D tensor"""

    MSE_LOSS = MeanSquaredError()

    @tf.function
    def call(self, x_true, x_gen):
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

    @tf.function
    def call(self, x_true, x_gen):
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

    @tf.function
    def call(self, x_true, x_gen):
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
        shape = tf.shape(x)
        k0 = tf.cast(tf.range(shape[1]), x.dtype)
        k1 = tf.cast(tf.range(shape[2]), x.dtype)
        freqs = tf.square(k0)[:, tf.newaxis] * tf.square(k1)[tf.newaxis, :]
        return freqs[tf.newaxis, ..., tf.newaxis]

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x_hat = tf.transpose(x, perm=[3, 0, 1, 2])
        x_hat = tf.signal.fft2d(tf.cast(x_hat, tf.complex64))
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 0])
        x_hat = tf.cast(tf.abs(x_hat), x.dtype)
        x_hat = tf.math.multiply(self._freq_weights(x), x_hat)
        return tf.math.log(1 + x_hat)

    @tf.function
    def call(self, x_true, x_gen):
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
        shape = tf.shape(x)
        k0 = tf.cast(tf.range(shape[1]), x.dtype)
        k1 = tf.cast(tf.range(shape[2]), x.dtype)
        freq_t = tf.cast(tf.range(shape[3]), x.dtype)
        freqs = (
            tf.square(k0)[:, tf.newaxis, tf.newaxis]
            * tf.square(k1)[tf.newaxis, :, tf.newaxis]
            * tf.square(freq_t)[tf.newaxis, tf.newaxis, :]
        )
        return freqs[tf.newaxis, ..., tf.newaxis]

    def _fft(self, x):
        """Apply needed transpositions and fft operation."""
        x_hat = tf.transpose(x, perm=[4, 0, 1, 2, 3])
        x_hat = tf.signal.fft3d(tf.cast(x_hat, tf.complex64))
        x_hat = tf.transpose(x_hat, perm=[1, 2, 3, 4, 0])
        x_hat = tf.cast(tf.abs(x_hat), x.dtype)
        x_hat = tf.math.multiply(self._freq_weights(x), x_hat)
        return tf.math.log(1 + x_hat)

    @tf.function
    def call(self, x_true, x_gen):
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
        **kwargs,
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

        super().__init__(**kwargs)
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
        shape = tf.shape(tensor)
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
        shape = tf.shape(tensor)
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
        shape = tf.shape(tensor)
        tensor = _assert_rank_in(
            tensor,
            (5,),
            'LowResLoss temporal coarsening expects 5D tensors',
        )
        tensor = tf.reshape(
            tensor,
            (shape[0], shape[1], shape[2], -1, self._t_enhance, shape[4]),
        )
        tensor = tf.math.reduce_sum(tensor, axis=4) / self._t_enhance
        return tensor

    @tf.function
    def call(self, x_true, x_gen):
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

        dtype = tf.as_dtype(tf.keras.backend.floatx())
        x_true = tf.cast(x_true, dtype)
        x_gen = tf.cast(x_gen, dtype)

        tf.debugging.assert_equal(
            tf.shape(x_true),
            tf.shape(x_gen),
            message=(
                'LowResLoss requires x_true and x_gen to have matching shapes'
            ),
        )
        s_only = x_true.shape.rank == 4

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

    @tf.function
    def call(self, x_true, x_gen):
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
        if x_true.shape.rank == 5:
            shape = tf.shape(x_true)
            new_shape = (shape[0] * shape[3], shape[1], shape[2], shape[4])
            x_true = tf.reshape(x_true, new_shape)
            x_gen = tf.reshape(x_gen, new_shape)

        losses = []
        for x_true_f, x_gen_f in zip(
            tf.unstack(x_true, axis=-1), tf.unstack(x_gen, axis=-1)
        ):
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

    @tf.function
    def call(self, x_true, x_gen):
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
        msg = (
            f'The {self.__class__.__name__} is meant to be used on spatial or '
            'spatiotemporal data only. Received tensor(s) that are not 4/5D'
        )
        x_true = _assert_rank_in(x_true, (4, 5), msg)
        x_gen = _assert_rank_in(x_gen, (4, 5), msg)

        if x_true.shape.rank == 4:
            x_true = tf.expand_dims(x_true, axis=3)
            x_gen = tf.expand_dims(x_gen, axis=3)

        shape = tf.shape(x_true)
        B, H, W, T, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        # Flatten only spatial/time dims → (B, HWT, C)
        x_true_flat = tf.reshape(x_true, (B, H * W * T, C))
        x_gen_flat = tf.reshape(x_gen, (B, H * W * T, C))

        # Random projection directions over HWT only
        proj = tf.random.normal(
            (self._n_projections, H * W * T), dtype=x_true.dtype
        )
        proj = tf.math.l2_normalize(proj, axis=-1)  # normalize

        # Project spatial dimensions → (num_proj, B, C)
        # matmul: (num_proj, HWT) @ (B, HWT, C) → (B, num_proj, C)
        x_true_proj = tf.einsum('ph,bhc->bpc', proj, x_true_flat)
        x_gen_proj = tf.einsum('ph,bhc->bpc', proj, x_gen_flat)

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

    @tf.function
    def call(self, x_true, x_gen):
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
        x_true = _assert_rank_in(x_true, (5,), msg)
        x_gen = _assert_rank_in(x_gen, (5,), msg)

        x_true_div = tf.stack([
            self._compute_md(x_true, feature) for feature in self.gen_features
        ])
        x_gen_div = tf.stack([
            self._compute_md(x_gen, feature) for feature in self.gen_features
        ])

        return self.LOSS_METRIC(x_true_div, x_gen_div)


class GeothermalConductiveHeatTransferLoss(Sup3rLoss):
    """Deviation from three-dimensional conductive heat transfer.

    This loss evaluates the conductive heat-transfer PDE residual described in
    [1] using predicted temperature, thermal conductivity, and surface heat
    flow. Temperature features are expected in C, thermal conductivity
    features in W/m-K, and heat-flow features in mW/m^2.

    The loss requires temperature and thermal conductivity channels at each
    requested depth and a single surface heat-flow channel at 0 m. Expected
    feature names are ``<temperature_prefix>_<depth>m`` (e.g. "t_1000m"),
    ``<thermal_conductivity_prefix>_<depth>m`` (e.g. "k_1000m"), and ``q_0m``.

    References
    ----------
    .. [1] Aljubran, M. J., and Horne, R. N., "Thermal Earth model for the
        conterminous United States using an interpolative physics-informed
        graph neural network," Geothermal Energy, vol. 12, no. 1, article 25,
        2024. doi:10.1186/s40517-024-00304-7.
    """

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
        """Initialize the conductive heat-transfer loss

        Parameters
        ----------
        dx : float
            Horizontal grid spacing along the x dimension in m.
        dy : float
            Horizontal grid spacing along the y dimension in m.
        depths : iterable of int, optional
            Depth levels in m used to assemble the temperature and thermal
            conductivity feature channels. Depths must include 0 m and be
            uniformly spaced so the vertical derivative can be evaluated.
        temperature_prefix : str, optional
            Prefix used for temperature channels in C. Expected feature names
            are ``<temperature_prefix>_<depth>m``.
        heat_flux_prefix : str, optional
            Prefix used for heat-flow channels in mW/m^2. This loss requires
            a surface heat-flow feature only, named
            ``<heat_flux_prefix>_0m``.
        thermal_conductivity_prefix : str, optional
            Prefix used for thermal-conductivity channels in W/m-K. Expected
            feature names are ``<thermal_conductivity_prefix>_<depth>m``.
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
        """Collect expected temperature, heat-flow, and conductivity feats"""
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
        """Extract temperature, surface heat-flow, and conductivity tensors"""
        t = tf.stack([x[..., i] for i in self.t_inds], axis=-1)
        q = tf.stack([x[..., i] for i in self.q_inds], axis=-1)
        k = tf.stack([x[..., i] for i in self.k_inds], axis=-1)
        return t, q, k

    def _compute_heat_transfer_residual(self, x):
        """Compute the conductive heat-transfer residual

        Temperature is interpreted in C, thermal conductivity in W/m-K, and
        heat flow in mW/m^2. Thermal conductivity is converted internally
        to mW/m-K before evaluating the residual.
        """
        t, q, k = self._get_feature_tensors(x)

        t = _reshape_depth_feature_for_vertical_derivative(t)  # C
        q = _reshape_depth_feature_for_vertical_derivative(q)  # mW/m^2
        k = _reshape_depth_feature_for_vertical_derivative(k) * 1000  # mW/m/K

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

    @tf.function
    def call(self, __, x_gen):
        """Evaluate the conductive heat-transfer loss

        Parameters
        ----------
        x_true : tf.tensor
            Ground truth data (unused).
        x_gen : tf.tensor
            Synthetic generator output used to compute the conductive
            heat-transfer residual. The feature axis must contain
            temperature channels in C, thermal conductivity channels in
            W/m-K, and a surface heat-flow channel in mW/m^2. Shape must be
            either:
            (n_observations, spatial_1, spatial_2, features) or
            (n_observations, spatial_1, spatial_2, temporal, features)

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
        x_gen = _assert_rank_in(x_gen, (4, 5), msg)

        expr = self._compute_heat_transfer_residual(x_gen)
        return self.LOSS_METRIC(tf.zeros_like(expr), expr)


class GeothermalPositiveTemperatureGradientLoss(Sup3rLoss):
    """Positive geothermal gradient loss

    This loss applies the positive-gradient regularization described in [1].
    It penalizes negative vertical temperature gradients so predicted
    temperature increases with depth. Temperature features are expected in C
    and named ``<temperature_prefix>_<depth>m`` (e.g. "t_2000m").

    References
    ----------
    .. [1] Aljubran, M. J., and Horne, R. N., "Thermal Earth model for the
        conterminous United States using an interpolative physics-informed
        graph neural network," Geothermal Energy, vol. 12, no. 1, article 25,
        2024. doi:10.1186/s40517-024-00304-7.
    """

    LOSS_METRIC = MeanSquaredError()

    def __init__(self, depths=range(0, 8000, 1000), temperature_prefix='t'):
        """Initialize the positive temperature-gradient loss

        Parameters
        ----------
        depths : iterable of int, optional
            Depth levels in m used to assemble temperature channels.
            At least two depths are required.
        temperature_prefix : str, optional
            Prefix used for temperature channels in C. Expected feature names
            are ``<temperature_prefix>_<depth>m``.
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
        """Compute temperature-gradient violations to penalize

        Temperature is interpreted in C. The loss only depends on the sign of
        the vertical temperature difference, so the finite difference is not
        normalized by depth spacing.
        """
        t = tf.stack([x[..., i] for i in self.t_inds], axis=-1)
        t = _reshape_depth_feature_for_vertical_derivative(t)

        # Not a true derivative (missing divide by dz), but we only care
        # about the sign, so this is sufficient
        dt = tf_derivative(t, axis=3)
        return tf.math.maximum(-1 * dt, tf.constant([0.0], dt.dtype))

    @tf.function
    def call(self, __, x_gen):
        """Evaluate the positive temperature-gradient loss

        Parameters
        ----------
        x_true : tf.tensor
            Ground truth data (unused).
        x_gen : tf.tensor
            Synthetic generator output used to compute vertical temperature
            gradients. The feature axis must contain temperature channels in
            C. Shape must be either:
            (n_observations, spatial_1, spatial_2, features) or
            (n_observations, spatial_1, spatial_2, temporal, features)

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
        x_gen = _assert_rank_in(x_gen, (4, 5), msg)

        temp_grads = self._compute_temperature_gradient(x_gen)
        return self.LOSS_METRIC(tf.zeros_like(temp_grads), temp_grads)


class GeothermalMohoBCLoss(Sup3rLoss):
    """Heat flow across Moho layer boundary condition loss

    This loss enforces the Moho boundary condition described in [1]. It helps
    satisfy the condition that the predicted heat flow is greater than or
    equal to the minimum heat flow implied at the Moho layer. Predicted
    heat-flow features are expected in mW/m^2 and the Moho
    temperature-gradient input is expected in C/km.

    References
    ----------
    .. [1] Aljubran, M. J., and Horne, R. N., "Thermal Earth model for the
        conterminous United States using an interpolative physics-informed
        graph neural network," Geothermal Energy, vol. 12, no. 1, article 25,
        2024. doi:10.1186/s40517-024-00304-7.
    """

    LOSS_METRIC = MeanSquaredError()

    def __init__(
        self,
        heat_flow_features,
        moho_gradient_layer='gg_mantle_60km',
        upper_mantle_thermal_conductivity=4.0,
    ):
        """Initialize the Moho boundary-condition loss.

        Parameters
        ----------
        heat_flow_features : iterable of str
            Names of predicted heat-flow features in mW/m^2.
        moho_gradient_layer : str, optional
            Name of the true-data Moho temperature-gradient layer in C/km.
        upper_mantle_thermal_conductivity : float, optional
            Upper-mantle thermal conductivity in W/m-K.
        """
        self.lambda_um = upper_mantle_thermal_conductivity
        super().__init__(
            gen_features=list(heat_flow_features),
            true_features=[moho_gradient_layer],
        )

    @tf.function
    def call(self, x_moho, x_gen):
        """Evaluate the Moho heat-flow boundary-condition loss

        Parameters
        ----------
        x_moho : tf.tensor
            Moho temperature gradient in C/km.
        x_gen : tf.tensor
            Synthetic generator output of surface heat-flow values in
            mW/m^2. Shape must be either:
            (n_observations, spatial_1, spatial_2, features) or
            (n_observations, spatial_1, spatial_2, temporal, features)

        Returns
        -------
        tf.tensor
            0D tensor loss value
        """
        moho_heat_flow = self.lambda_um * x_moho  # [W/m-K] * [K/km] = [mW/m^2]
        residuals = tf.math.maximum(
            # Moho [mW/m^2] - surface heat flow [mW/m^2]
            moho_heat_flow - x_gen,
            tf.constant([0.0], x_gen.dtype),
        )
        return self.LOSS_METRIC(tf.zeros_like(residuals), residuals)


class GeothermalObsLoss(Sup3rLoss):
    """Masked loss for geothermal observed quantities

    This loss performs the masked observation matching described in [1]. It
    compares predicted geothermal channels against observed targets while
    ignoring missing observations. Units are inherited from the paired
    features, such as temperature in C, thermal conductivity in W/m-K, and
    heat flow in mW/m^2.

    References
    ----------
    .. [1] Aljubran, M. J., and Horne, R. N., "Thermal Earth model for the
        conterminous United States using an interpolative physics-informed
        graph neural network," Geothermal Energy, vol. 12, no. 1, article 25,
        2024. doi:10.1186/s40517-024-00304-7.
    """

    LOSS_METRIC = MeanAbsoluteError()

    @tf.function
    def call(self, x_true, x_gen):
        """Evaluate the masked geothermal observation loss

        The feature dimensions of ``x_true`` and ``x_gen`` must align
        with the configured generated and true features. Observed values
        may contain NaNs, which are ignored when computing the loss.
        """
        msg = (
            f'Number of features in `x_true`: {x_true.shape[-1]} must match '
            f'the length of `true_features`: {len(self.true_features)}, '
            f'`x_gen`: {x_gen.shape[-1]} must match the length of '
            f'`gen_features`: {len(self.gen_features)}'
        )
        tf.debugging.assert_equal(
            tf.shape(x_true)[-1], len(self.true_features), message=msg
        )
        tf.debugging.assert_equal(
            tf.shape(x_gen)[-1], len(self.gen_features), message=msg
        )

        mask = tf.math.logical_not(tf.math.is_nan(x_true))
        x_true_m = tf.boolean_mask(x_true, mask)
        x_gen_m = tf.boolean_mask(x_gen, mask)
        obs_loss = tf.cond(
            tf.math.reduce_all(tf.math.is_nan(x_true)),
            lambda: tf.constant(0, dtype=x_true.dtype),
            lambda: self.LOSS_METRIC(x_true_m, x_gen_m),
        )
        return obs_loss


class ObsAssimilationLoss(Sup3rLoss):
    """Loss for training with both dense and sparse ground truth where the obs
    locations are explicitly considered. This is designed to encourage
    matching observations where they exist, matching the dense true fields
    where obs are missing, and blend smoothly between the two in a
    neighbourhood around obs locations through a gradient penalty.

    Assumes ``true_features`` contains ``gen_features`` followed by sparse
    observation versions of those same features in matching order. For
    example::

        gen_features  = ['u_10m', 'v_10m']
        true_features = ['u_10m', 'v_10m', 'u_10m_obs', 'v_10m_obs']

    Three weighted terms are combined:

    1. **Background MAE** – MAE between generator output and the dense true
       fields at locations where sparse obs are *missing* (NaN).
    2. **Observation MAE** – MAE between generator output and sparse obs at
       locations where sparse obs are *present* (not NaN).
    3. **Gradient penalty** – mean spatial gradient magnitude of the
       generator output inside a neighbourhood around each obs location,
       encouraging spatial smoothness near observations.
    """

    LOSS_METRIC = MeanAbsoluteError()

    def __init__(
        self,
        gen_features,
        true_features=None,
        background_weight=1.0,
        obs_weight=1.0,
        gradient_weight=1.0,
        gradient_radius=1,
    ):
        """
        Parameters
        ----------
        gen_features : list of str
            Generator output feature names (N features).
        true_features : list of str | None
            True-data feature names. Must contain 2N entries: the first N
            match ``gen_features`` (dense reference) and the last N are the
            corresponding sparse observation features (NaN where missing).
            Defaults to ``gen_features + [f + '_obs' for f in gen_features]``.
        background_weight : float
            Weight for the background (non-observed) MAE term.
        obs_weight : float
            Weight for the observation MAE term.
        gradient_weight : float
            Weight for the spatial gradient penalty near obs locations.
        gradient_radius : int
            Radius in grid cells of the neighbourhood around each obs
            location where the gradient penalty is applied. 0 restricts
            the penalty to the obs locations themselves.
        """
        gen_features = list(gen_features)
        if true_features is None:
            true_features = gen_features + [f + '_obs' for f in gen_features]
        true_features = list(true_features)

        n = len(gen_features)
        if len(true_features) != 2 * n:
            raise ValueError(
                'ObsBlendLoss requires len(true_features) == '
                f'2 * len(gen_features). Got {len(true_features)} true '
                f'features and {n} gen features.'
            )

        super().__init__(
            gen_features=gen_features, true_features=true_features
        )
        self._bg_weight = background_weight
        self._obs_weight = obs_weight
        self._gradient_weight = gradient_weight
        self._gradient_radius = gradient_radius

    def _dilate_obs_mask(self, mask):
        """Dilate a 2-D spatial obs mask by ``gradient_radius`` grid cells.

        Parameters
        ----------
        mask : tf.Tensor
            Float tensor of shape ``(B, H, W)`` with 1 where an observation
            is present and 0 elsewhere.

        Returns
        -------
        tf.Tensor
            Dilated float mask of shape ``(B, H, W)``.
        """
        r = self._gradient_radius
        if r == 0:
            return mask
        mask_4d = mask[..., tf.newaxis]  # (B, H, W, 1)
        mask_4d = tf.pad(mask_4d, [[0, 0], [r, r], [r, r], [0, 0]])
        mask_4d = tf.nn.max_pool2d(
            mask_4d, ksize=2 * r + 1, strides=1, padding='VALID'
        )
        return mask_4d[..., 0]  # (B, H, W)

    def _dilate_obs_mask_3d(self, mask):
        """Dilate a spatiotemporal obs mask by ``gradient_radius`` in 3-D.

        The dilation is applied jointly over the H, W, and T axes so that
        observations spread their neighbourhood through time as well as space.

        Parameters
        ----------
        mask : tf.Tensor
            Float tensor of shape ``(B, H, W, T)`` with 1 where an
            observation is present and 0 elsewhere.

        Returns
        -------
        tf.Tensor
            Dilated float mask of shape ``(B, H, W, T)``.
        """
        r = self._gradient_radius
        if r == 0:
            return mask
        # max_pool3d expects (B, D, H, W, C); treat T as depth D.
        mask_5d = tf.transpose(mask, [0, 3, 1, 2])[
            ..., tf.newaxis
        ]  # (B, T, H, W, 1)
        mask_5d = tf.pad(mask_5d, [[0, 0], [r, r], [r, r], [r, r], [0, 0]])
        mask_5d = tf.nn.max_pool3d(
            mask_5d,
            ksize=[2 * r + 1, 2 * r + 1, 2 * r + 1],
            strides=[1, 1, 1],
            padding='VALID',
        )
        return tf.transpose(mask_5d[..., 0], [0, 2, 3, 1])  # (B, H, W, T)

    def _gradient_penalty(self, x_gen, obs_present):
        """Spatial gradient penalty weighted by a dilated obs neighbourhood.

        For 5-D inputs ``(B, H, W, T, C)`` the neighbourhood dilation is
        performed in 3-D (H, W, T) so that observations extend their
        influence through time.

        Parameters
        ----------
        x_gen : tf.Tensor
            Generator output ``(B, H, W, C)`` or ``(B, H, W, T, C)``.
        obs_present : tf.Tensor
            Float mask of same shape as ``x_gen`` with 1 where a sparse
            observation is present for that feature.

        Returns
        -------
        tf.Tensor
            Scalar gradient penalty.
        """
        # Reduce over feature dim → (B, H, W) or (B, H, W, T)
        spatial_obs = tf.reduce_max(obs_present, axis=-1)
        if len(x_gen.shape) == 5:
            spatial_obs = self._dilate_obs_mask_3d(spatial_obs)  # (B, H, W, T)
        else:
            spatial_obs = self._dilate_obs_mask(spatial_obs)  # (B, H, W)

        grad_mag = tf.abs(tf_derivative(x_gen, axis=1)) + tf.abs(
            tf_derivative(x_gen, axis=2)
        )  # (B, H, W, [T,] C)
        # Broadcast spatial mask over the feature axis to match grad_mag
        mask = tf.cast(
            tf.broadcast_to(spatial_obs[..., tf.newaxis], tf.shape(grad_mag)),
            bool,
        )
        return tf.cond(
            tf.math.reduce_all(~mask),
            lambda: tf.constant(0.0, dtype=x_gen.dtype),
            lambda: self.LOSS_METRIC(
                tf.zeros_like(tf.boolean_mask(grad_mag, mask)),
                tf.boolean_mask(grad_mag, mask),
            ),
        )

    @tf.function
    def call(self, x_true, x_gen):
        """Evaluate the sparse observation loss.

        Parameters
        ----------
        x_true : tf.Tensor
            True data of shape ``(B, H, W, 2C)`` or ``(B, H, W, T, 2C)``.
            The first C channels are dense reference fields; the last C
            channels are sparse obs fields (NaN where no observation).
        x_gen : tf.Tensor
            Generator output of shape ``(B, H, W, C)`` or ``(B, H, W, T, C)``.

        Returns
        -------
        tf.Tensor
            Scalar (0-D) loss value.
        """
        dtype = tf.as_dtype(tf.keras.backend.floatx())
        x_true = tf.cast(x_true, dtype)
        x_gen = tf.cast(x_gen, dtype)

        n = len(self.gen_features)
        x_true_bg = x_true[..., :n]  # dense reference, shape (..., n)
        x_obs = x_true[
            ..., n:
        ]  # sparse obs, shape (..., n); NaN where missing

        obs_mask = ~tf.math.is_nan(x_obs)
        obs_present = tf.cast(obs_mask, dtype)

        # 1. Background MAE at locations where obs is missing
        bg_mask = ~obs_mask
        bg_loss = self.LOSS_METRIC(
            tf.boolean_mask(x_true_bg, bg_mask),
            tf.boolean_mask(x_gen, bg_mask),
        )

        # 2. Observation MAE at locations where obs are present
        obs_loss = tf.cond(
            tf.math.reduce_all(tf.math.is_nan(x_obs)),
            lambda: tf.constant(0.0, dtype=dtype),
            lambda: self.LOSS_METRIC(
                tf.boolean_mask(x_obs, obs_mask),
                tf.boolean_mask(x_gen, obs_mask),
            ),
        )

        # 3. Spatial gradient penalty around obs locations
        grad_loss = self._gradient_penalty(x_gen, obs_present)

        return (
            self._bg_weight * bg_loss
            + self._obs_weight * obs_loss
            + self._gradient_weight * grad_loss
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
