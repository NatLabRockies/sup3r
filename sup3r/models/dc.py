"""Sup3r data-centric model software"""

import logging

import numpy as np

from .base import Sup3rGan

np.set_printoptions(precision=3)

logger = logging.getLogger(__name__)


class Sup3rGanDC(Sup3rGan):
    """Data-centric model using loss across time bins to select training
    observations"""

    def calc_val_loss_gen(self, batch_handler, weight_gen_advers):
        """Calculate the validation total loss across the validation
        samples. e.g. If the sample domain has 100 steps and the
        validation set has 10 bins then this will get a list of losses across
        step 0 to 10, 10 to 20, etc.  Use this to determine performance
        within bins and to update how observations are selected from these
        bins.

        Parameters
        ----------
        batch_handler : sup3r.preprocessing.BatchHandlerDC
            BatchHandler object to iterate through
        weight_gen_advers : float
            Weight factor for the adversarial loss component of the generator
            vs. the discriminator.

        Returns
        -------
        total_losses : np.ndarray
            Array of total losses for all sample bins, with shape
            (n_space_bins, n_time_bins)
        content_losses : np.ndarray
            Array of content losses for all sample bins, with shape
            (n_space_bins, n_time_bins)
        """
        total_losses = np.zeros(
            (batch_handler.n_space_bins, batch_handler.n_time_bins),
            dtype=np.float32,
        )
        content_losses = np.zeros(
            (batch_handler.n_space_bins, batch_handler.n_time_bins),
            dtype=np.float32,
        )
        for i, batch in enumerate(batch_handler.val_data):
            logger.info(f'Calculating validation loss for batch {i} / '
                        f'{len(batch_handler.val_data)}...')
            hi_res_exo = self.get_hr_exo_input(batch.high_res)
            hi_res_gen = self._tf_generate(batch.low_res, hi_res_exo)
            loss, loss_details = self.calc_loss(
                batch.high_res,
                hi_res_gen,
                weight_gen_advers=weight_gen_advers,
            )
            row = i // batch_handler.n_time_bins
            col = i % batch_handler.n_time_bins
            total_losses[row, col] = loss
            content_losses[row, col] = loss_details['loss_gen_content']
        return total_losses, content_losses

    def calc_val_loss(self, batch_handler, weight_gen_advers):
        """Overloading the base calc_val_loss method. Method updates the
        temporal weights for the batch handler based on the losses across the
        time bins

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
            Updated loss_details with mean validation loss calculated using
            the validation samples across the time bins
        """
        logger.debug('Starting end-of-epoch validation loss calculation...')
        loss_details = {}
        total_losses, content_losses = self.calc_val_loss_gen(
            batch_handler, weight_gen_advers
        )

        t_weights = total_losses.mean(axis=0)
        t_weights /= t_weights.sum()

        s_weights = total_losses.mean(axis=1)
        s_weights /= s_weights.sum()

        logger.debug(
            f'Previous spatial weights: {batch_handler.spatial_weights}'
        )
        logger.debug(
            f'Previous temporal weights: {batch_handler.temporal_weights}'
        )
        batch_handler.update_weights(
            spatial_weights=s_weights, temporal_weights=t_weights
        )
        logger.debug(
            'New spatiotemporal weights (space, time):\n'
            f'{total_losses / total_losses.sum()}'
        )
        logger.debug(f'New spatial weights: {s_weights}')
        logger.debug(f'New temporal weights: {t_weights}')

        loss_details['mean_val_loss_gen'] = round(np.mean(total_losses), 3)
        loss_details['mean_val_loss_gen_content'] = round(
            np.mean(content_losses), 3
        )
        return loss_details
