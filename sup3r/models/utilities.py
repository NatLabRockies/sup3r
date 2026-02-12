"""Utilities shared across the `sup3r.models` module"""

import logging
import os
import sys
import threading

import numpy as np
import tensorflow as tf
from phygnn.layers.custom_layers import (
    Sup3rAdder,
    Sup3rConcat,
    Sup3rConcatObs,
    Sup3rObsModel,
)
from scipy.interpolate import RegularGridInterpolator
from tensorflow.keras import optimizers

from sup3r.utilities.utilities import Timer

logger = logging.getLogger(__name__)

SUP3R_OBS_LAYERS = Sup3rObsModel, Sup3rConcatObs

SUP3R_EXO_LAYERS = Sup3rAdder, Sup3rConcat

SUP3R_LAYERS = (*SUP3R_EXO_LAYERS, *SUP3R_OBS_LAYERS)


class TrainingSession:
    """Wrapper to gracefully exit batch handler thread during training, upon a
    keyboard interruption."""

    def __init__(self, batch_handler, model, **kwargs):
        """
        Parameters
        ----------
        batch_handler: BatchHandler
            Batch iterator
        model: Sup3rGan
            Gan model to run in new thread
        **kwargs : dict
            Model keyword args
        """
        self.batch_handler = batch_handler
        self.model = model
        self.kwargs = kwargs

    def run(self):
        """Wrap model.train()."""
        model_thread = threading.Thread(
            target=self.model.train,
            args=(self.batch_handler,),
            kwargs=self.kwargs,
        )
        try:
            logger.info(
                'Starting training session. Training for %s epochs',
                self.kwargs['n_epoch'],
            )
            model_thread.start()
        except KeyboardInterrupt:
            logger.info('Ending training session.')
            self.batch_handler.stop()
            model_thread.join()
            sys.exit()
        except Exception as e:
            logger.info('Ending training session. %s', e)
            self.batch_handler.stop()
            model_thread.join()
            sys.exit()

        model_thread.join()
        logger.info('Finished training')


class TensorboardMixIn:
    """MixIn class for tensorboard logging and profiling."""

    def __init__(self):
        self._tb_writer = None
        self._tb_log_dir = None
        self._total_batches = None
        self._history = None
        self.timer = Timer()

    @property
    def total_batches(self):
        """Record of total number of batches for logging."""
        if self._total_batches is None:
            if self._history is not None and 'total_batches' in self._history:
                self._total_batches = self._history['total_batches'].values[-1]
            else:
                self._total_batches = 0
        return self._total_batches

    @total_batches.setter
    def total_batches(self, value):
        """Set total number of batches."""
        self._total_batches = value

    def dict_to_tensorboard(self, entry):
        """Write data to tensorboard log file. This is usually a loss_details
        dictionary.

        Parameters
        ----------
        entry: dict
            Dictionary of values to write to tensorboard log file
        """
        with self._tb_writer.as_default():
            for name, value in entry.items():
                if isinstance(value, str):
                    tf.summary.text(name, value, self.total_batches)
                else:
                    tf.summary.scalar(name, value, self.total_batches)

    def profile_to_tensorboard(self, name, export_tb=True):
        """Write profile data to tensorboard log file.

        Parameters
        ----------
        name : str
            Tag name to use for profile info
        export_tb : bool
            Flag to enable/disable tensorboard profiling
        """
        if self._tb_writer is not None and export_tb:
            with self._tb_writer.as_default():
                tf.summary.trace_export(
                    name=name,
                    step=self.total_batches,
                    profiler_outdir=self._tb_log_dir,
                )

    def _init_tensorboard_writer(self, out_dir):
        """Initialize the ``tf.summary.SummaryWriter`` to use for writing
        tensorboard compatible log files.

        Parameters
        ----------
        out_dir : str
            Standard out_dir where model epochs are saved. e.g. './gan_{epoch}'
        """
        tb_log_pardir = os.path.abspath(os.path.join(out_dir, os.pardir))
        self._tb_log_dir = os.path.join(tb_log_pardir, 'logs')
        os.makedirs(self._tb_log_dir, exist_ok=True)
        self._tb_writer = tf.summary.create_file_writer(self._tb_log_dir)


def get_optimizer_class(conf):
    """Get optimizer class from keras"""
    if hasattr(optimizers, conf['name']):
        optimizer_class = getattr(optimizers, conf['name'])
    else:
        msg = '%s not found in keras optimizers.'
        logger.error(msg, conf['name'])
        raise ValueError(msg)
    return optimizer_class


def st_interp(low, s_enhance, t_enhance, t_centered=False):
    """Spatiotemporal bilinear interpolation for low resolution field on a
    regular grid. Used to provide baseline for comparison with gan output

    Parameters
    ----------
    low : ndarray
        Low resolution field to interpolate.
        (spatial_1, spatial_2, temporal)
    s_enhance : int
        Factor by which to enhance the spatial domain
    t_enhance : int
        Factor by which to enhance the temporal domain
    t_centered : bool
        Flag to switch time axis from time-beginning (Default, e.g.
        interpolate 00:00 01:00 to 00:00 00:30 01:00 01:30) to
        time-centered (e.g. interp 01:00 02:00 to 00:45 01:15 01:45 02:15)

    Returns
    -------
    ndarray
        Spatiotemporally interpolated low resolution output
    """
    assert len(low.shape) == 3, 'Input to st_interp must be 3D array'
    msg = 'Input to st_interp cannot include axes with length 1'
    assert not any(s <= 1 for s in low.shape), msg

    lr_y, lr_x, lr_t = low.shape
    hr_y, hr_x, hr_t = lr_y * s_enhance, lr_x * s_enhance, lr_t * t_enhance

    # assume outer bounds of mesh (0, 10) w/ points on inside of that range
    y = np.arange(0, 10, 10 / lr_y) + 5 / lr_y
    x = np.arange(0, 10, 10 / lr_x) + 5 / lr_x

    # remesh (0, 10) with high res spacing
    new_y = np.arange(0, 10, 10 / hr_y) + 5 / hr_y
    new_x = np.arange(0, 10, 10 / hr_x) + 5 / hr_x

    t = np.arange(0, 10, 10 / lr_t)
    new_t = np.arange(0, 10, 10 / hr_t)
    if t_centered:
        t += 5 / lr_t
        new_t += 5 / hr_t

    # set RegularGridInterpolator to do extrapolation
    interp = RegularGridInterpolator(
        (y, x, t), low, bounds_error=False, fill_value=None
    )

    # perform interp
    X, Y, T = np.meshgrid(new_x, new_y, new_t)
    return interp((Y, X, T))
