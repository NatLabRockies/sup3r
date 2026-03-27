"""Sup3r conditional moment model software"""

import logging
import os
import pprint
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers

from sup3r.utilities import VERSION_RECORD

from .abstract import AbstractSingleModel
from .interface import AbstractInterface

logger = logging.getLogger(__name__)


class Sup3rCondMom(AbstractSingleModel, AbstractInterface):
    """Basic Sup3r conditional moments model."""

    def __init__(
        self,
        gen_layers,
        optimizer=None,
        learning_rate=1e-4,
        num_par=None,
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
            JSON/JSON5/YAML/TOML config file containing the input layers
            argument or a .pkl for a saved pre-trained model.
        optimizer : tf.keras.optimizers.Optimizer | dict | None | str
            Instantiated tf.keras.optimizers object or a dict optimizer config
            from tf.keras.optimizers.get_config(). None defaults to Adam.
        learning_rate : float, optional
            Optimizer learning rate. Not used if optimizer input arg is a
            pre-initialized object or if optimizer input arg is a config dict.
        num_par : int | None
            Number of trainable parameters in the model
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
            multiple GPUs exist, the CPU will be the default device (this was
            tested as most efficient given the custom multi-gpu strategy
            developed in self.run_gradient_descent())
        name : str | None
            Optional name for the model.
        """
        super().__init__()

        self.default_device = default_device
        if self.default_device is None and len(self.gpu_list) == 1:
            self.default_device = '/gpu:0'
        elif self.default_device is None and len(self.gpu_list) > 1:
            self.default_device = '/cpu:0'

        self.name = name if name is not None else self.__class__.__name__
        self._meta = meta if meta is not None else {}
        self._num_par = num_par if num_par is not None else 0
        self.loss_name = 'MeanSquaredError'

        self._history = history
        if isinstance(self._history, str):
            self._history = pd.read_csv(self._history, index_col=0)

        self._init_records()

        self._optimizer = self.init_optimizer(optimizer, learning_rate)

        self._gen = self.load_network(gen_layers, 'generator')

        self._means = means
        self._stdevs = stdevs

    def save(self, out_dir):
        """Save the model with its sub-networks to a directory.

        Parameters
        ----------
        out_dir : str
            Directory to save model files. This directory will be created
            if it does not already exist.
        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        fp_gen = os.path.join(out_dir, 'model_gen.pkl')
        self.generator.save(fp_gen)

        fp_history = None
        if isinstance(self.history, pd.DataFrame):
            fp_history = os.path.join(out_dir, 'history.csv')
            self.history.to_csv(fp_history)

        self.save_params(out_dir)

        logger.info('Saved model to disk in directory: {}'.format(out_dir))

    @classmethod
    def load(cls, model_dir, verbose=True):
        """Load the model with its sub-networks from a previously saved-to
        output directory.

        Parameters
        ----------
        model_dir : str
            Directory to load model files from.
        verbose : bool
            Flag to log information about the loaded model.

        Returns
        -------
        out : BaseModel
            Returns a pretrained gan model that was previously saved to out_dir
        """
        if verbose:
            logger.info(
                'Loading model from disk in directory: {}'.format(model_dir)
            )
            msg = 'Active python environment versions: \n{}'.format(
                pprint.pformat(VERSION_RECORD, indent=4)
            )
            logger.info(msg)

        fp_gen = os.path.join(model_dir, 'model_gen.pkl')
        params = cls.load_saved_params(model_dir, verbose=verbose)
        return cls(fp_gen, **params)

    def update_optimizer(self, **kwargs):
        """Update optimizer by changing current configuration

        Parameters
        ----------
        kwargs : dict
            kwargs to use for optimizer configuration update
        """

        conf = self.get_optimizer_config(self.optimizer)
        conf.update(**kwargs)
        OptimizerClass = getattr(optimizers, conf['name'])
        self._optimizer = OptimizerClass.from_config(conf)

    @property
    def meta(self):
        """Get meta data dictionary that defines how the model was created"""

        if 'class' not in self._meta:
            self._meta['class'] = self.__class__.__name__

        return self._meta

    @property
    def model_params(self):
        """Model parameters, used to save model to disc

        Returns
        -------
        model_params: dict
        """

        config_optm_g = self.get_optimizer_config(self.optimizer)

        num_par = int(
            np.sum([np.prod(v.get_shape().as_list()) for v in self.weights])
        )

        means = self._means
        stdevs = self._stdevs
        if means is not None and stdevs is not None:
            means = {k: float(v) for k, v in means.items()}
            stdevs = {k: float(v) for k, v in stdevs.items()}

        model_params = {
            'name': self.name,
            'num_par': num_par,
            'version_record': self.version_record,
            'optimizer': config_optm_g,
            'means': means,
            'stdevs': stdevs,
            'meta': self.meta,
        }

        return model_params

    @property
    def weights(self):
        """Get a list of all the layer weights and bias terms for the
        generator network
        """
        return self.generator_weights

    @tf.function
    def calc_loss_cond_mom(self, output_true, output_gen, mask):
        """Calculate the loss of the moment predictor

        Parameters
        ----------
        output_true : tf.Tensor
            True realization output
        output_gen : tf.Tensor
            Predicted realization output
        mask : tf.Tensor
            Mask to apply

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the
            moment predictor
        loss_details : dict
            Namespace of the breakdown of loss components
        """

        return self.loss_fun(output_gen * mask, output_true * mask)

    def calc_loss(self, output_true, output_gen, mask):
        """Calculate the total moment predictor loss

        Parameters
        ----------
        output_true : tf.Tensor
            True realization output
        output_gen : tf.Tensor
            Predicted realization output
        mask : tf.Tensor
            Mask to apply

        Returns
        -------
        loss : tf.Tensor
            0D tensor representing the loss value for the
            moment predictor
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        output_gen = self._combine_loss_input(output_true, output_gen)

        if output_gen.shape != output_true.shape:
            msg = (
                'The tensor shapes of the synthetic output {} and '
                'true output {} did not have matching shape! '
                'Check the spatiotemporal enhancement multipliers in your '
                'your model config and data handlers.'.format(
                    output_gen.shape, output_true.shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        loss, loss_details = self.calc_loss_cond_mom(
            output_true, output_gen, mask
        )

        loss_details.update({'loss_gen': loss})

        return loss, loss_details

    def calc_val_loss(self, batch_handler):
        """Calculate the validation loss at the current state of model training

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through

        Returns
        -------
        loss_details : dict
            Running mean of validation loss details
        """
        logger.debug('Starting end-of-epoch validation loss calculation...')
        for val_batch in batch_handler.val_data:
            val_exo_data = self.get_hr_exo_input(val_batch.high_res)
            output_gen = self._tf_generate(val_batch.low_res, val_exo_data)
            _, v_loss_details = self.calc_loss(
                val_batch.output, output_gen, val_batch.mask
            )

            self._val_record = self.update_loss_details(
                self._val_record,
                v_loss_details,
                len(batch_handler.val_data),
                prefix='val_',
            )

        return self._val_record.mean(axis=0)

    def _train_epoch(self, batch_handler, multi_gpu=False):
        """Train the model for one epoch.

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and the resulting gradient from each GPU will
            constitute a single gradient descent step with the nominal learning
            rate that the model was initialized with.

        Returns
        -------
        loss_details : dict
            Namespace of the breakdown of loss components
        """
        for ib, batch in enumerate(batch_handler):
            b_loss_details = {}
            b_loss_details = self.run_gradient_descent(
                batch.low_res,
                batch.output,
                self.generator_weights,
                optimizer=self.optimizer,
                multi_gpu=multi_gpu,
                mask=batch.mask,
            )

            self._train_record = self.update_loss_details(
                self._train_record,
                b_loss_details,
                len(batch_handler),
                prefix='train_',
            )
            loss_details = self._train_record.mean().to_dict()

            logger.debug(
                'Batch {} out of {} has epoch-average gen loss of: '
                '{:.2e}. '.format(
                    ib, len(batch_handler), loss_details['train_loss_gen']
                )
            )

        return loss_details

    def train(
        self,
        batch_handler,
        input_resolution,
        n_epoch,
        checkpoint_int=None,
        out_dir='./condMom_{epoch}',
        early_stop_on=None,
        early_stop_threshold=0.005,
        early_stop_n_epoch=5,
        multi_gpu=False,
        tensorboard_log=True,
    ):
        """Train the model on real low res data and real high res data

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandler
            BatchHandler object to iterate through
        input_resolution : dict
            Dictionary specifying spatiotemporal input resolution. e.g.
            {'temporal': '60min', 'spatial': '30km'}
        n_epoch : int
            Number of epochs to train on
        checkpoint_int : int | None
            Epoch interval at which to save checkpoint models.
        out_dir : str
            Directory to save checkpoint models. Should have {epoch} in
            the directory name. This directory will be created if it does not
            already exist.
        early_stop_on : str | None
            If not None, this should be a column in the training history to
            evaluate for early stopping (e.g. validation_loss_gen).
            If this value in this history decreases by
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
        multi_gpu : bool
            Flag to break up the batch for parallel gradient descent
            calculations on multiple gpus. If True and multiple GPUs are
            present, each batch from the batch_handler will be divided up
            between the GPUs and the resulting gradient from each GPU will
            constitute a single gradient descent step with the nominal learning
            rate that the model was initialized with.
        tensorboard_log : bool
            Whether to write log file for use with tensorboard. Log data can
            be viewed with ``tensorboard --logdir <logdir>`` where ``<logdir>``
            is the parent directory of ``out_dir``, and pointing the browser to
            the printed address.
        """
        if tensorboard_log:
            self._init_tensorboard_writer(out_dir)

        self.set_norm_stats(batch_handler.means, batch_handler.stds)
        lower_models = getattr(batch_handler, 'lower_models', {})
        for model in [self, *lower_models.values()]:
            model.set_model_params(
                input_resolution=input_resolution, batch_handler=batch_handler
            )

        epochs = list(range(n_epoch))

        if self._history is None:
            self._history = pd.DataFrame(columns=['elapsed_time'])
            self._history.index.name = 'epoch'
        else:
            epochs += self._history.index.values[-1] + 1

        t0 = time.time()
        logger.info(
            'Training model for {} epochs starting at epoch {}'.format(
                n_epoch, epochs[0]
            )
        )

        for epoch in epochs:
            loss_details = self._train_epoch(
                batch_handler, multi_gpu=multi_gpu
            )
            loss_details.update(self.calc_val_loss(batch_handler))

            msg = 'Epoch {} of {} gen train loss: {:.2e} '.format(
                epoch, epochs[-1], loss_details['train_loss_gen']
            )

            if all(loss in loss_details for loss in ['val_loss_gen']):
                msg += 'gen val loss: {:.2e} '.format(
                    loss_details['val_loss_gen']
                )

            logger.info(msg)

            lr_g = self.get_optimizer_config(self.optimizer)['learning_rate']

            extras = {'learning_rate_gen': lr_g}

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

            if stop:
                break

        batch_handler.stop()
