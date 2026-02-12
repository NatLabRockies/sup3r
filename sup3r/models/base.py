"""Sup3r model software"""

import copy
import logging
import os
import pprint
import time
from warnings import warn

import numpy as np
import pandas as pd
import tensorflow as tf

from sup3r.utilities import VERSION_RECORD

from .abstract import AbstractSingleModel
from .interface import AbstractInterface
from .utilities import get_optimizer_class

logger = logging.getLogger(__name__)


class Sup3rGan(AbstractSingleModel, AbstractInterface):
    """Basic sup3r GAN model."""

    def __init__(
        self,
        gen_layers,
        disc_layers,
        loss='MeanSquaredError',
        optimizer=None,
        learning_rate=1e-4,
        optimizer_disc=None,
        learning_rate_disc=None,
        history=None,
        meta=None,
        means=None,
        stdevs=None,
        default_device=None,
        name=None,
    ):
        """
        Parameters
        ----------
        gen_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            generative super resolving model. Can also be a str filepath to a
            .json config file containing the input layers argument or a .pkl
            for a saved pre-trained model.
        disc_layers : list | str
            Hidden layers input argument to phygnn.base.CustomNetwork for the
            discriminative model (spatial or spatiotemporal discriminator). Can
            also be a str filepath to a .json config file containing the input
            layers argument or a .pkl for a saved pre-trained model.
        loss : str | dict
            Loss function class name from sup3r.utilities.loss_metrics
            (prioritized) or tensorflow.keras.losses. Defaults to
            tf.keras.losses.MeanSquaredError. As a dictionary this can
            include multiple loss function classes, each with
            dictionaries of kwargs for that function. Can also include a
            key ``term_weights``, which provides a list of weights for
            each loss function. e.g. ``{'SpatialExtremesLoss': {},
            'MeanAbsoluteError': {}, 'term_weights': [0.8, 0.2]}``
        optimizer : tf.keras.optimizers.Optimizer | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        optimizer_disc : tf.keras.optimizers.Optimizer | dict | None
            Same as optimizer input, but if specified this makes a different
            optimizer just for the discriminator network (spatial or
            spatiotemporal disc).
        learning_rate_disc : float, optional
            Same as learning_rate input, but if specified this makes a
            different learning_rate just for the discriminator network (spatial
            or spatiotemporal disc).
        history : pd.DataFrame | str | None
            Model training history with "epoch" index, str pointing to a saved
            history csv file with "epoch" as first column, or None for clean
            history
        meta : dict | None
            Model meta data that describes how the model was created.
        means : dict | None
            Set of mean values for data normalization keyed by feature name.
            Can be used to maintain a consistent normalization scheme between
            transfer learning domains.
        stdevs : dict | None
            Set of stdev values for data normalization keyed by feature name.
            Can be used to maintain a consistent normalization scheme between
            transfer learning domains.
        default_device : str | None
            Option for default device placement of model weights. If None and a
            single GPU exists, that GPU will be the default device. If None and
            multiple GPUs exist, the first GPU will be the default device
            (this was tested as most efficient given the custom multi-gpu
             strategy developed in self.run_gradient_descent()). Examples:
            "/gpu:0" or "/cpu:0"
        name : str | None
            Optional name for the GAN.
        """
        super().__init__()

        self.default_device = default_device
        if self.default_device is None and len(self.gpu_list) >= 1:
            self.default_device = '/gpu:0'

        self.name = name if name is not None else self.__class__.__name__
        self._meta = meta if meta is not None else {}

        self.loss_name = loss
        self.loss_fun = self.get_loss_fun(loss)

        self._history = history
        if isinstance(self._history, str):
            self._history = pd.read_csv(self._history, index_col=0)

        self._init_records()

        optimizer_disc = optimizer_disc or copy.deepcopy(optimizer)
        learning_rate_disc = learning_rate_disc or learning_rate
        self._optimizer = self.init_optimizer(optimizer, learning_rate)
        self._optimizer_disc = self.init_optimizer(
            optimizer_disc, learning_rate_disc
        )

        self._gen = self.load_network(gen_layers, 'generator')
        self._disc = self.load_network(disc_layers, 'discriminator')

        self._means = means
        self._stdevs = stdevs

    def save(self, out_dir):
        """Save the GAN with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save GAN model files. This directory will be created
            if it does not already exist.
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        self.generator.save(fp_gen)

        fp_disc = os.path.join(out_dir, 'model_disc.pkl')
        self.discriminator.save(fp_disc)

        fp_history = None
        if isinstance(self.history, pd.DataFrame):
            fp_history = os.path.join(out_dir, 'history.csv')
            self.history.to_csv(fp_history)

        self.save_params(out_dir)

        logger.info('Saved GAN to disk in directory: {}'.format(out_dir))

    @classmethod
    def _load(cls, model_dir, verbose=True):
        """Get gen, disc, and params for given model_dir.

        Parameters
        ----------
        model_dir : str
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        fp_gen : str
            Path to generator model
        fp_disc : str
            Path to discriminator model
        params : dict
            Dictionary of model params to be used in model initialization
        """
        if verbose:
            logger.info(
                'Loading GAN from disk in directory: {}'.format(model_dir)
            )
            msg = 'Active python environment versions: \n{}'.format(
                pprint.pformat(VERSION_RECORD, indent=4)
            )
            logger.info(msg)

        fp_gen = os.path.join(model_dir, 'model_gen.pkl')
        fp_disc = os.path.join(model_dir, 'model_disc.pkl')
        params = cls.load_saved_params(model_dir, verbose=verbose)

        return fp_gen, fp_disc, params

    @classmethod
    def load(cls, model_dir, verbose=True):
        """Load the GAN with its sub-networks from a previously saved-to output
        directory.

        Parameters
        ----------
        model_dir : str
            Directory to load GAN model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to out_dir
        """
        fp_gen, fp_disc, params = cls._load(model_dir, verbose=verbose)
        return cls(fp_gen, fp_disc, **params)

    @property
    def discriminator(self):
        """Get the discriminator model.

        Returns
        -------
        phygnn.base.CustomNetwork
        """
        return self._disc

    @property
    def discriminator_weights(self):
        """Get a list of layer weights and bias terms for the discriminator
        model.

        Returns
        -------
        list
        """
        return self.discriminator.weights

    def discriminate(self, hi_res, norm_in=False):
        """Run the discriminator model on a hi resolution input field.

        Parameters
        ----------
        hi_res : np.ndarray
            Real or fake high res data in a 4D or 5D tensor:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        norm_in : bool
            Flag to normalize low_res input data if the self._means,
            self._stdevs attributes are available. The disc should always
            received normalized data with mean=0 stdev=1.

        Returns
        -------
        out : np.ndarray
            Discriminator output logits
        """

        if isinstance(hi_res, tf.Tensor):
            hi_res = hi_res.numpy()

        if norm_in and self._means is not None:
            mean_arr = [self._means[fn] for fn in self.hr_out_features]
            std_arr = [self._stdevs[fn] for fn in self.hr_out_features]
            mean_arr = np.array(mean_arr, dtype=np.float32)
            std_arr = np.array(std_arr, dtype=np.float32)
            hi_res = hi_res if isinstance(hi_res, tf.Tensor) else hi_res.copy()
            hi_res = (hi_res - mean_arr) / std_arr

        out = self.discriminator.layers[0](hi_res)
        layer_num = 1
        try:
            for i, layer in enumerate(self.discriminator.layers[1:]):
                out = layer(out)
                layer_num = i + 1
        except Exception as e:
            msg = 'Could not run layer #{} "{}" on tensor of shape {}'.format(
                layer_num, layer, out.shape
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        return out.numpy()

    @tf.function
    def _tf_discriminate(self, hi_res):
        """Run the discriminator model on a hi resolution input field.

        Parameters
        ----------
        hi_res : np.ndarray
            Real or fake high res data in a 4D or 5D tensor:
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
            This input should always be normalized with mean=0 and stdev=1

        Returns
        -------
        out : np.ndarray
            Discriminator output logits
        """
        out = self.discriminator.layers[0](hi_res)
        layer_num = 1
        try:
            for i, layer in enumerate(self.discriminator.layers[1:]):
                layer_num = i + 1
                out = layer(out)
        except Exception as e:
            msg = 'Could not run layer #{} "{}" on tensor of shape {}'.format(
                layer_num, layer, out.shape
            )
            logger.error(msg)
            raise RuntimeError(msg) from e

        return out

    @property
    def optimizer_disc(self):
        """Get the tensorflow optimizer to perform gradient descent
        calculations for the discriminator network.

        Returns
        -------
        tf.keras.optimizers.Optimizer
        """
        return self._optimizer_disc

    def update_optimizer(self, option='generator', **kwargs):
        """Update optimizer by changing current configuration

        Parameters
        ----------
        option : str
            Which optimizer to update. Can be "generator", "discriminator", or
            "all"
        kwargs : dict
            kwargs to use for optimizer configuration update
        """

        if 'gen' in option.lower() or 'all' in option.lower():
            conf = self.get_optimizer_config(self.optimizer)
            conf.update(**kwargs)
            optimizer_class = get_optimizer_class(conf)
            self._optimizer = optimizer_class.from_config(conf)

        if 'disc' in option.lower() or 'all' in option.lower():
            conf = self.get_optimizer_config(self.optimizer_disc)
            conf.update(**kwargs)
            optimizer_class = get_optimizer_class(conf)
            self._optimizer_disc = optimizer_class.from_config(conf)

    @property
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

        if 'class' not in self._meta:
            self._meta['class'] = self.__class__.__name__

        return self._meta

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """

        means = self._means
        stdevs = self._stdevs
        if means is not None and stdevs is not None:
            means = {k: float(v) for k, v in means.items()}
            stdevs = {k: float(v) for k, v in stdevs.items()}

        return {
            'name': self.name,
            'loss': self.loss_name,
            'version_record': self.version_record,
            'optimizer': self.get_optimizer_config(self.optimizer),
            'optimizer_disc': self.get_optimizer_config(self.optimizer_disc),
            'means': means,
            'stdevs': stdevs,
            'meta': self.meta,
            'default_device': self.default_device,
        }

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator and discriminator networks
        """
        return self.generator_weights + self.discriminator_weights

    def init_weights(self, lr_shape, hr_shape, device=None):
        """Initialize the generator and discriminator weights with device
        placement.

        Parameters
        ----------
        lr_shape : tuple
            Shape of one batch of low res input data for sup3r resolution. Note
            that the batch size (axis=0) must be included, but the actual batch
            size doesnt really matter.
        hr_shape : tuple
            Shape of one batch of high res input data for sup3r resolution.
            Note that the batch size (axis=0) must be included, but the actual
            batch size doesnt really matter.
        device : str | None
            Option to place model weights on a device. If None,
            self.default_device will be used.
        """

        if not self.generator_weights:
            if device is None:
                device = self.default_device

            logger.info(
                'Initializing model weights on device "{}"'.format(device)
            )
            low_res = tf.cast(np.ones(lr_shape), dtype=tf.float32)
            hi_res = tf.cast(np.ones(hr_shape), dtype=tf.float32)

            hr_exo_shape = hr_shape[:-1] + (1,)
            hr_exo = tf.cast(np.ones(hr_exo_shape), dtype=tf.float32)

            with tf.device(device):
                hr_exo_data = {}
                for feature in self.hr_exo_features + self.obs_features:
                    hr_exo_data[feature] = hr_exo
                out = self._tf_generate(low_res, hr_exo_data)
                msg = (
                    f'Number of model outputs {out.shape[-1]} does not '
                    'match the number of computed hr_out_features '
                    f'{len(self.hr_out_features)}'
                )
                assert out.shape[-1] == len(self.hr_out_features), msg
                _ = self._tf_discriminate(hi_res)

    @staticmethod
    def get_weight_update_fraction(
        history, comparison_key, update_bounds=(0.5, 0.95), update_frac=0.0
    ):
        """Get the factor by which to multiply previous adversarial loss
        weight

        Parameters
        ----------
        history : dict
            Dictionary with information on how often discriminators
            were trained during previous epoch.
        comparison_key : str
            history key to use for update check
        update_bounds : tuple
            Tuple specifying allowed range for history[comparison_key]. If
            history[comparison_key] < update_bounds[0] then the weight will be
            increased by (1 + update_frac). If history[comparison_key] >
            update_bounds[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        update_frac : float
            Fraction by which to increase/decrease adversarial loss weight

        Returns
        -------
        float
            Factor by which to multiply old weight to get updated weight
        """

        val = history[comparison_key]
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[-1]

        if val < update_bounds[0]:
            return 1 + update_frac
        if val > update_bounds[1]:
            return 1 / (1 + update_frac)
        return 1

    @tf.function
    def calc_loss_gen_content(self, hi_res_true, hi_res_gen):
        """Calculate the content loss term for the generator model.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.

        Returns
        -------
        loss_gen_s : tf.Tensor
            0D tensor generator model loss for the content loss comparing the
            hi res ground truth to the hi res synthetically generated output.
        """
        slc = (
            slice(0, None)
            if len(self.hr_exo_features) == 0
            else slice(0, -len(self.hr_exo_features))
        )
        # gen is first since loss can included regularizers which just
        # apply to generator output
        return self.loss_fun(hi_res_gen[..., slc], hi_res_true[..., slc])

    @staticmethod
    @tf.function
    def calc_loss_disc(disc_out_true, disc_out_gen):
        """Calculate the loss term for the discriminator model (either the
        spatial or temporal discriminator. This uses the relativistic
        discriminator loss described in [Wang2018]_.

        Note: Instead of training the discriminator to label data as either
        real or fake this trains the disc to label data as more or less
        realistic. To use this for adversarial loss we simply set
        ``disc_out_true`` to ``disc_out_gen`` and vice versa, which then
        encourages the generator to produce output which is "more realistic"
        than the true high-res data.

        References
        ----------
        .. [Wang2018] Wang, Xintao, et al. "Esrgan: Enhanced super-resolution
            generative adversarial networks." Proceedings of the European
            conference on computer vision (ECCV) workshops. 2018.

        Parameters
        ----------
        disc_out_true : tf.Tensor
            Raw discriminator outputs from the discriminator model predicting
            only on ground truth data hi_res_true (not on hi_res_gen).
        disc_out_gen : tf.Tensor
            Raw discriminator outputs from the discriminator model predicting
            only on synthetic data hi_res_gen (not on hi_res_true).

        Returns
        -------
        loss_disc : tf.Tensor
            0D tensor discriminator model loss for either the spatial or
            temporal component of the super resolution generated output.
        """
        true_logits = disc_out_true - tf.reduce_mean(disc_out_gen)
        fake_logits = disc_out_gen - tf.reduce_mean(disc_out_true)
        logits = tf.concat([true_logits, fake_logits], axis=0)
        labels = tf.concat(
            [tf.ones_like(disc_out_true), tf.zeros_like(disc_out_gen)], axis=0
        )
        loss_disc = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
        return tf.reduce_mean(loss_disc)

    def update_adversarial_weights(
        self,
        history,
        adaptive_update_fraction,
        adaptive_update_bounds,
        weight_gen_advers,
        train_disc,
    ):
        """Update spatial / temporal adversarial loss weights based on training
        fraction history.

        Parameters
        ----------
        history : dicts
            Dictionary with information on how often discriminators
            were trained during current and previous epochs.
        adaptive_update_fraction : float
            Amount by which to increase or decrease adversarial loss weights
            for adaptive updates
        adaptive_update_bounds : tuple
            Tuple specifying allowed range for history[comparison_key]. If
            history[comparison_key] < update_bounds[0] then the weight will be
            increased by (1 + update_frac). If history[comparison_key] >
            update_bounds[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_disc : bool
            Whether the discriminator was set to be trained during the
            previous epoch

        Returns
        -------
        weight_gen_advers : float
            Updated weight factor for the adversarial loss component of the
            generator vs. the discriminator.
        """

        if adaptive_update_fraction > 0:
            update_frac = 1
            if train_disc:
                update_frac = self.get_weight_update_fraction(
                    history,
                    'disc_train_frac',
                    update_frac=adaptive_update_fraction,
                    update_bounds=adaptive_update_bounds,
                )
                weight_gen_advers *= update_frac

            if update_frac != 1:
                logger.debug(
                    f'New discriminator weight: {weight_gen_advers:.4e}'
                )

        return weight_gen_advers

    @staticmethod
    def check_batch_handler_attrs(batch_handler):
        """Not all batch handlers have the following attributes. So we perform
        some sanitation before sending to `set_model_params`"""
        return {
            k: getattr(batch_handler, k, None)
            for k in [
                'smoothing',
                'lr_features',
                'hr_exo_features',
                'hr_out_features',
                'smoothed_features',
            ]
            if hasattr(batch_handler, k)
        }

    def train(
        self,
        batch_handler,
        input_resolution,
        n_epoch,
        weight_gen_advers=0.001,
        train_gen=True,
        train_disc=True,
        disc_loss_bounds=(0.45, 0.6),
        checkpoint_int=None,
        out_dir='./gan_{epoch}',
        early_stop_on=None,
        early_stop_threshold=0.005,
        early_stop_n_epoch=5,
        adaptive_update_bounds=(0.9, 0.99),
        adaptive_update_fraction=0.0,
        multi_gpu=False,
        log_tb=False,
        export_tb=False,
    ):
        """Train the GAN model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through
        input_resolution : dict
            Dictionary specifying spatiotemporal input resolution. e.g.
            {'temporal': '60min', 'spatial': '30km'}
        n_epoch : int
            Number of epochs to train on
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminator will not train unless train_disc=True and
            train_gen=False.
        checkpoint_int : int | None
            Epoch interval at which to save checkpoint models.
        out_dir : str
            Directory to save checkpoint GAN models. Should have {epoch} in
            the directory name. This directory will be created if it does not
            already exist.
        early_stop_on : str | None
            If not None, this should be a column in the training history to
            evaluate for early stopping (e.g. validation_loss_gen,
            validation_loss_disc). If this value in this history decreases by
            an absolute fractional relative difference of less than 0.01 for
            more than 5 epochs in a row, the training will stop early.
        early_stop_threshold : float
            The absolute relative fractional difference in validation loss
            between subsequent epochs below which an early termination is
            warranted. E.g. if val losses were 0.1 and 0.0998 the relative
            diff would be calculated as 0.0002 / 0.1 = 0.002 which would be
            less than the default thresold of 0.01 and would satisfy the
            condition for early termination.
        early_stop_n_epoch : int
            The number of consecutive epochs that satisfy the threshold that
            warrants an early stop.
        adaptive_update_bounds : tuple
            Tuple specifying allowed range for loss_details[comparison_key]. If
            history[comparison_key] < threshold_range[0] then the weight will
            be increased by (1 + update_frac). If history[comparison_key] >
            threshold_range[1] then the weight will be decreased by 1 / (1 +
            update_frac).
        adaptive_update_fraction : float
            Amount by which to increase or decrease adversarial weights for
            adaptive updates
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and resulting gradients from each GPU will be
            summed and then applied once per batch at the nominal learning
            rate that the model and optimizer were initialized with.
            If true and multiple gpus are found, ``default_device`` device
            should be set to /gpu:0
        log_tb : bool
            Whether to write log file for use with tensorboard. Log data can
            be viewed with ``tensorboard --logdir <logdir>`` where ``<logdir>``
            is the parent directory of ``out_dir``, and pointing the browser to
            the printed address.
        export_tb : bool
            Whether to export profiling information to tensorboard. This can
            then be viewed in the tensorboard dashboard under the profile tab

        TODO: (1) args here are getting excessive. Might be time for some
        refactoring.
        (2) cal_val_loss should be done in a separate thread from train_epoch
        so they can be done concurrently. This would be especially important
        for batch handlers which require val data, like dc handlers.
        (3) Would like an automatic way to exit the batch handler thread
        instead of manually calling .stop() here.
        """
        if log_tb:
            self._init_tensorboard_writer(out_dir)

        self.set_norm_stats(batch_handler.means, batch_handler.stds)
        params = self.check_batch_handler_attrs(batch_handler)
        self.set_model_params(
            input_resolution=input_resolution,
            s_enhance=batch_handler.s_enhance,
            t_enhance=batch_handler.t_enhance,
            **params,
        )

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(columns=['elapsed_time'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        t0 = time.time()
        logger.info(
            'Training model with adversarial weight: {} '
            'for {} epochs starting at epoch {}'.format(
                weight_gen_advers, n_epoch, epochs[0]
            )
        )

        for epoch in epochs:
            t_epoch = time.time()
            loss_details = self._train_epoch(
                batch_handler,
                weight_gen_advers,
                train_gen,
                train_disc,
                disc_loss_bounds,
                multi_gpu=multi_gpu,
                export_tb=export_tb,
            )
            loss_details.update(
                self.calc_val_loss(batch_handler, weight_gen_advers)
            )

            msg = f'Epoch {epoch} of {epochs[-1]} '
            msg += 'gen/disc train loss: {:.2e}/{:.2e} '.format(
                loss_details['train_loss_gen'], loss_details['train_loss_disc']
            )

            check1 = 'val_loss_gen' in loss_details
            check2 = 'val_loss_disc' in loss_details
            if check1 and check2:
                msg += 'gen/disc val loss: {:.2e}/{:.2e} '.format(
                    loss_details['val_loss_gen'], loss_details['val_loss_disc']
                )

            logger.info(msg)

            extras = {
                'weight_gen_advers': weight_gen_advers,
                'disc_loss_bound_0': disc_loss_bounds[0],
                'disc_loss_bound_1': disc_loss_bounds[1],
            }

            opt_g = self.get_optimizer_state(self.optimizer)
            opt_d = self.get_optimizer_state(self.optimizer_disc)
            opt_g = {f'OptmGen/{key}': val for key, val in opt_g.items()}
            opt_d = {f'OptmDisc/{key}': val for key, val in opt_d.items()}
            extras.update(opt_g)
            extras.update(opt_d)

            weight_gen_advers = self.update_adversarial_weights(
                loss_details,
                adaptive_update_fraction,
                adaptive_update_bounds,
                weight_gen_advers,
                train_disc,
            )

            stop = self.finish_epoch(
                epoch,
                epochs,
                t0,
                loss_details,
                checkpoint_int,
                out_dir,
                early_stop_on,
                early_stop_threshold,
                early_stop_n_epoch,
                extras=extras,
            )
            logger.info(
                'Finished training epoch in {:.4f} seconds'.format(
                    time.time() - t_epoch
                )
            )
            if stop:
                break
        logger.info(
            'Finished training {} epochs in {:.4f} seconds'.format(
                n_epoch,
                time.time() - t0,
            )
        )

        batch_handler.stop()

    def calc_loss(
        self,
        hi_res_true,
        hi_res_gen,
        weight_gen_advers=0.001,
        train_gen=True,
        train_disc=False,
        compute_disc=False,
    ):
        """Calculate the GAN loss function using generated and true high
        resolution data.

        Parameters
        ----------
        hi_res_true : tf.Tensor
            Ground truth high resolution spatiotemporal data.
        hi_res_gen : tf.Tensor
            Superresolved high resolution spatiotemporal data generated by the
            generative model.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            True if generator is being trained, then loss=loss_gen
        train_disc : bool
            True if disc is being trained, then loss=loss_disc
        compute_disc : bool
            True if discriminator loss should be computed, even if not being
            trained. Outside of generator pre-training this needs to be
            tracked to determine if the discriminator is "too good" or "not
            good enough"

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the network being trained
            (either generator or one of the discriminators)
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        hi_res_gen = self._combine_loss_input(hi_res_true, hi_res_gen)

        if hi_res_gen.shape != hi_res_true.shape:
            msg = (
                'The tensor shapes of the synthetic output {} and '
                'true high res {} did not have matching shape! '
                'Check the spatiotemporal enhancement multipliers in your '
                'your model config and data handlers.'.format(
                    hi_res_gen.shape, hi_res_true.shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        disc_out_true = self._tf_discriminate(hi_res_true)
        disc_out_gen = self._tf_discriminate(hi_res_gen)

        loss_details = {}
        loss = None

        if compute_disc or train_disc:
            loss_details['loss_disc'] = self.calc_loss_disc(
                disc_out_true=disc_out_true, disc_out_gen=disc_out_gen
            )

        if train_gen:
            loss_gen_content, loss_gen_content_details = (
                self.calc_loss_gen_content(hi_res_true, hi_res_gen)
            )
            loss_gen_advers = self.calc_loss_disc(
                disc_out_true=disc_out_gen, disc_out_gen=disc_out_true
            )
            loss = loss_gen_content + weight_gen_advers * loss_gen_advers
            loss_details['loss_gen'] = loss
            loss_details['loss_gen_content'] = loss_gen_content
            loss_details['loss_gen_advers'] = loss_gen_advers
            loss_details.update(loss_gen_content_details)

        elif train_disc:
            loss = loss_details['loss_disc']

        return loss, loss_details

    def calc_val_loss(self, batch_handler, weight_gen_advers):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.

        Returns
        -------
        loss_details : dict
            Running mean for validation loss details
        """
        logger.debug('Starting end-of-epoch validation loss calculation...')
        for batch in batch_handler.val_data:
            _, v_loss_details, _, _ = self._get_hr_exo_and_loss(
                batch.low_res,
                batch.high_res,
                weight_gen_advers=weight_gen_advers,
            )
            self._val_record = self.update_loss_details(
                self._val_record,
                v_loss_details,
                len(batch_handler.val_data),
                prefix='val_',
            )
        return self._val_record.mean(axis=0)

    def _train_batch(
        self,
        batch,
        train_gen,
        only_gen,
        gen_too_good,
        train_disc,
        only_disc,
        disc_too_good,
        weight_gen_advers,
        multi_gpu=False,
    ):
        """Run gradient descent and get loss details for a given batch for the
        current epoch.

        Parameters
        ----------
        batch : sup3r.preprocessing.base.DsetTuple
            Object with ``.low_res`` and ``.high_res`` arrays
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        only_gen : bool
            Flag whether to only train the generator for this set of epochs
        gen_too_good : bool
            Flag whether to skip training the generator and only train the
            discriminator, due to superior performance, for this batch.
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        only_disc : bool
            Flag whether to only train the discriminator for this set of epochs
        gen_too_good : bool
            Flag whether to skip training the discriminator and only train the
            generator, due to superior performance, for this batch.
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and resulting gradients from each GPU will be
            summed and then applied once per batch at the nominal learning
            rate that the model and optimizer were initialized with.
            If true and multiple gpus are found, ``default_device`` device
            should be set to /gpu:0

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components for the given batch
        """

        trained_gen = False
        trained_disc = False
        loss_details = {}
        if only_gen or (train_gen and not gen_too_good):
            trained_gen = True
            b_loss_details = self.timer(self.run_gradient_descent)(
                batch.low_res,
                batch.high_res,
                self.generator_weights,
                weight_gen_advers=weight_gen_advers,
                optimizer=self.optimizer,
                train_gen=True,
                train_disc=False,
                compute_disc=train_disc,
                multi_gpu=multi_gpu,
            )
            loss_details.update(b_loss_details)

        if only_disc or (train_disc and not disc_too_good):
            trained_disc = True
            b_loss_details = self.timer(self.run_gradient_descent)(
                batch.low_res,
                batch.high_res,
                self.discriminator_weights,
                weight_gen_advers=weight_gen_advers,
                optimizer=self.optimizer_disc,
                train_gen=False,
                train_disc=True,
                multi_gpu=multi_gpu,
            )
            loss_details.update(b_loss_details)

        loss_details = {k: float(v) for k, v in loss_details.items()}
        loss_details['gen_train_frac'] = float(trained_gen)
        loss_details['disc_train_frac'] = float(trained_disc)
        return loss_details

    def _post_batch(self, ib, b_loss_details, n_batches, previous_means):
        """Update loss details after the current batch and write to log.

        Parameters
        ----------
        ib : int
            Index of the current batch
        b_loss_details : dict
            Dictionary of loss details for the current batch
        n_batches : int
            Number of batches in an epoch
        previous_means : dict
            Dictionary of loss means over the last epoch

        Returns
        -------
        loss_means : dict
            Dictionary of running loss means
        """
        # set default values for when either disc / gen is not trained for the
        # last batch
        for key, val in previous_means.items():
            if key.startswith('train_'):
                b_loss_details.setdefault(key.replace('train_', ''), val)

        self._train_record = self.update_loss_details(
            self._train_record,
            b_loss_details,
            n_batches,
            prefix='train_',
        )

        if self._tb_writer is not None:
            self.dict_to_tensorboard(b_loss_details)
            self.dict_to_tensorboard(self.timer.log)

        trained_gen = bool(self._train_record['gen_train_frac'].values[-1])
        trained_disc = bool(self._train_record['disc_train_frac'].values[-1])
        disc_loss = self._train_record['train_loss_disc'].values.mean()
        gen_loss = self._train_record['train_loss_gen'].values.mean()

        logger.info(
            'Batch {} out of {} has (gen / disc) loss of: ({:.2e} / {:.2e}). '
            'Running mean (gen / disc): ({:.2e} / {:.2e}). Trained '
            '(gen / disc): ({} / {})'.format(
                ib + 1,
                n_batches,
                b_loss_details['loss_gen'],
                b_loss_details['loss_disc'],
                gen_loss,
                disc_loss,
                trained_gen,
                trained_disc,
            )
        )
        if all([not trained_gen, not trained_disc]):
            msg = (
                'For some reason none of the GAN networks trained during '
                'batch {} out of {}!'.format(ib, n_batches)
            )
            logger.warning(msg)
            warn(msg)
        return self._train_record.mean(axis=0).to_dict()

    def _train_epoch(
        self,
        batch_handler,
        weight_gen_advers,
        train_gen,
        train_disc,
        disc_loss_bounds,
        multi_gpu=False,
        export_tb=False,
    ):
        """Train the GAN for one epoch.

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.
        train_gen : bool
            Flag whether to train the generator for this set of epochs
        train_disc : bool
            Flag whether to train the discriminator for this set of epochs
        disc_loss_bounds : tuple
            Lower and upper bounds for the discriminator loss outside of which
            the discriminators will not train unless train_disc=True or
            and train_gen=False.
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and resulting gradients from each GPU will be
            summed and then applied once per batch at the nominal learning
            rate that the model and optimizer were initialized with.
            If true and multiple gpus are found, ``default_device`` device
            should be set to /gpu:0
        export_tb : bool
            Whether to export profiling information to tensorboard. This can
            then be viewed in the tensorboard dashboard under the profile tab

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        lr_shape, hr_shape = batch_handler.shapes
        self.init_weights(lr_shape, hr_shape)

        self.init_weights(
            (1, *batch_handler.lr_shape), (1, *batch_handler.hr_shape)
        )

        disc_th_low = np.min(disc_loss_bounds)
        disc_th_high = np.max(disc_loss_bounds)
        loss_means = self._train_record.mean().to_dict()
        loss_means.setdefault('train_loss_disc', 0)
        loss_means.setdefault('train_loss_gen', 0)

        only_gen = train_gen and not train_disc
        only_disc = train_disc and not train_gen

        if export_tb:
            tf.summary.trace_on(graph=True, profiler=True)

        prev_time = time.time()
        for ib, batch in enumerate(batch_handler):
            start = time.time()

            b_loss_details = {}
            loss_disc = loss_means['train_loss_disc']
            disc_too_good = loss_disc <= disc_th_low
            disc_too_bad = (loss_disc > disc_th_high) and train_disc
            gen_too_good = disc_too_bad

            b_loss_details = self._train_batch(
                batch,
                train_gen,
                only_gen,
                gen_too_good,
                train_disc,
                only_disc,
                disc_too_good,
                weight_gen_advers,
                multi_gpu,
            )

            loss_means = self._post_batch(
                ib, b_loss_details, len(batch_handler), loss_means
            )

            total_step_time = time.time() - prev_time
            batch_step_time = time.time() - start
            batch_load_time = total_step_time - batch_step_time

            logger.info(
                f'Finished batch step {ib + 1} / {len(batch_handler)} in '
                f'{total_step_time:.4f} seconds. Batch load time: '
                f'{batch_load_time:.4f} seconds. Batch train time: '
                f'{batch_step_time:.4f} seconds.'
            )

            prev_time = time.time()

        self.total_batches += len(batch_handler)
        loss_details = self._train_record.mean().to_dict()
        loss_details['total_batches'] = int(self.total_batches)
        self.profile_to_tensorboard('training_epoch', export_tb=export_tb)
        return loss_details
