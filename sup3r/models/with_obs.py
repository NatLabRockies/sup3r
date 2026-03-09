"""Sup3r model with training on observation data."""

import logging

import numpy as np
import tensorflow as tf

from sup3r.utilities.utilities import RANDOM_GENERATOR

from .base import Sup3rGan

logger = logging.getLogger(__name__)


# TODO: Refactor so the observations can be either "proxies" sampled from
# gridded high res or "real" observations drawn from the .obs attribute of
# the batches in the batch handler.
#   - either construct mask like currently done or calculate it from the
#     .obs data by checking for NaN values. The former is done when using
#     sampled proxies for training and the latter is done when using real
#     observations for training.
#   - might be able to remove obs loss weighting and delegate this to loss
#     functions
#   - add flag in init for proxies vs real.
#   - or maybe the obs data can just be supplied through the high_res attr
#     and can allow for some NaNs. Then the mask can be calculated on the high
#     res data by checking for NaNs and this would work for both proxies and
#     real obs. This would be simpler and more flexible but would require
#     changes to the batch handler to allow for NaNs in the high res data
#   - might be some quirks with tracking output features


class Sup3rGanWithObs(Sup3rGan):
    """Sup3r GAN model which includes mid network observation fusion. This
    model is useful for when production runs will be over a domain for which
    observation data is available.

    Note
    ----
    During training this model uses sparse sampling of ground truth data to
    simulate observation data. This is done by creating masks of ground truth
    data and then selecting unmasked data. The features used for sampling are
    defined in the model configuration, in observation specific model layers
    such as ``Sup3rObsModel`` or ``Sup3rConcatObs``. The observation features
    in the model configuration should be named with an `_obs` suffix, and
    during training these are matched with the corresponding true high res
    features without the `_obs` suffix. e.g. If the goal is to condition a
    model on sparse temperature_10m data then the observation feature should be
    named temperature_10m_obs, and during training the gridded high res
    temperature_10m data will be masked to create synthetic observation data
    for this feature. The observation masks are only used during training.
    During inference "real" observation data, with the full name defined in the
    model configuration, is passed in as exogenous data with NaN values for
    where the observations are not available. These NaN values are then handled
    by observation specific model layers.
    """

    def __init__(
        self,
        *args,
        use_proxy_obs=True,
        onshore_obs_frac=None,
        offshore_obs_frac=None,
        loss_obs_weight=0.0,
        loss_obs=None,
        **kwargs,
    ):
        """
        Initialize the Sup3rGanWithObs model.

        Parameters
        ----------
        args : list
            Positional args for ``Sup3rGan`` parent class.
        use_proxy_obs : bool
            Whether to use proxy observations sampled from the gridded high res
            data during training. If False, the model will expect real
            observation data in the .high_res attribute of the batches in the
            batch handler and will calculate the observation mask based on
            where there are NaN values in the high res data. If True, the model
            will create synthetic observation data by masking the gridded high
            res data during training and will calculate the observation mask
            based on the specified onshore and offshore observation fractions.
        onshore_obs_frac : dict[List] | dict[float]
            Fraction of the batch that should be treated as onshore
            observations. Should include ``spatial`` key and optionally
            ``time`` key if this is a spatiotemporal model. The values should
            correspond roughly to the fraction of the production domain for
            which onshore observations are available (spatial) and the fraction
            of the full time period that these cover. The values can be either
            a list (for a lower and upper bound, respectively) or a single
            float. For each batch a spatial frac will be selected by either
            sampling uniformly between this lower and upper bound or just
            using a single float.
        offshore_obs_frac : dict
            Same as ``onshore_obs_frac`` but for offshore observations.
            Offshore observations are frequently sparser than onshore
            observations.
        loss_obs : str | dict
            Loss function to use for the additional observation loss term. This
            defaults to the content loss function specified with ``loss``.
        loss_obs_weight : float
            Value used to weight observation locations in extra content loss
            term. e.g. The new content loss will include ``obs_loss_weight *
            MAE(hi_res_gen[~obs_mask], hi_res_true[~obs_mask])``
        kwargs : dict
            Keyword arguments for the ``Sup3rGan`` parent class.
        """
        super().__init__(*args, **kwargs)
        self.use_proxy_obs = use_proxy_obs
        self.onshore_obs_frac = (
            {} if onshore_obs_frac is None else onshore_obs_frac
        )
        self.offshore_obs_frac = (
            {} if offshore_obs_frac is None else offshore_obs_frac
        )
        loss_obs = self.loss_name if loss_obs is None else loss_obs
        self.loss_obs_name = loss_obs
        self.loss_obs_fun = self.get_loss_fun(loss_obs)
        self.loss_obs_weight = loss_obs_weight

    @tf.function
    def _get_loss_obs_comparison(self, hi_res_true, hi_res_gen, obs_mask):
        """Get loss for observation locations and for non observation
        locations."""
        hr_true = hi_res_true[..., : len(self.hr_out_features)]
        loss_obs, _ = self.loss_obs_fun(
            hi_res_gen[~obs_mask], hr_true[~obs_mask]
        )
        loss_non_obs, _ = self.loss_obs_fun(
            hi_res_gen[obs_mask], hr_true[obs_mask]
        )
        return loss_obs, loss_non_obs

    @property
    def obs_training_inds(self):
        """Get the observation feature indices in the true high res
        data. True observation features are named with an '_obs' suffix.
        When training with proxy observations these indices select
        the corresponding gridded features (no '_obs' suffix). Otherwise,
        these indices select observation features with an '_obs' suffix."""

        if self.use_proxy_obs:
            hr_feats = [f.replace('_obs', '') for f in self.hr_features]
            obs_inds = [
                hr_feats.index(f.replace('_obs', ''))
                for f in self.obs_features
            ]
        else:
            obs_inds = [self.hr_features.index(f) for f in self.obs_features]
        return obs_inds

    def _get_single_obs_mask(self, hi_res, spatial_frac, time_frac=1.0):
        """Get observation mask for a given spatial and temporal obs
        fraction for a single batch entry.

        Parameters
        ----------
        hi_res : np.ndarray
            True high resolution data for a single batch entry.
        spatial_frac : float
            Fraction of the spatial domain that should be treated as
            observations. This is a value between 0 and 1.
        time_frac : float, optional
            Fraction of the temporal domain that should be treated as
            observations. This is a value between 0 and 1. Default is 1.0

        Returns
        -------
        np.ndarray
            Mask which is True for locations that are not observed and False
            for locations that are observed.
            (spatial_1, spatial_2, n_features)
            (spatial_1, spatial_2, n_temporal, n_features)
        """
        mask_shape = [*hi_res.shape[:3], 1, len(self.hr_out_features)]
        mask_shape[3] = hi_res.shape[3] if self.is_5d else 1
        s_mask = RANDOM_GENERATOR.uniform(size=mask_shape[1:3]) <= spatial_frac
        s_mask = s_mask[..., None, None]
        t_mask = RANDOM_GENERATOR.uniform(size=mask_shape[-2]) <= time_frac
        t_mask = t_mask[None, None, ..., None]
        mask = ~(s_mask & t_mask)
        mask = np.repeat(mask, mask_shape[-1], axis=-1)
        return mask if self.is_5d else np.squeeze(mask, axis=-2)

    def _get_obs_mask(self, hi_res, spatial_frac, time_frac=1.0):
        """Get observation mask for a given spatial and temporal obs
        fraction for an entire batch. This is divided between spatial and
        temporal fractions because often the spatial fraction is significantly
        lower than the temporal fraction in practice, e.g. for a given spatial
        location there might be observations for most of the time period but
        only a small fraction of the spatial domain is observed.

        Parameters
        ----------
        hi_res : np.ndarray
            True high resolution data for the entire batch.
        spatial_frac : float | list
            Fraction of the spatial domain that should be treated as
            observations. This is a value between 0 and 1 or a list with
            lower and upper bounds for the spatial fraction.
        time_frac : float | list, optional
            Fraction of the temporal domain that should be treated as
            observations. This is a value between 0 and 1 or a list with
            lower and upper bounds for the temporal fraction. Default is 1.0

        Returns
        -------
        np.ndarray
            Mask which is True for locations that are not observed and False
            for locations that are observed.
            (n_obs, spatial_1, spatial_2, n_features)
            (n_obs, spatial_1, spatial_2, n_temporal, n_features)
        """
        s_range = (
            spatial_frac
            if isinstance(spatial_frac, (list, tuple))
            else [spatial_frac, spatial_frac]
        )
        t_range = (
            time_frac
            if isinstance(time_frac, (list, tuple))
            else [time_frac, time_frac]
        )
        s_fracs = RANDOM_GENERATOR.uniform(*s_range, size=hi_res.shape[0])
        t_fracs = RANDOM_GENERATOR.uniform(*t_range, size=hi_res.shape[0])
        s_fracs = np.clip(s_fracs, 0, 1)
        t_fracs = np.clip(t_fracs, 0, 1)
        mask = tf.stack(
            [
                self._get_single_obs_mask(hi_res, s, t)
                for s, t in zip(s_fracs, t_fracs)
            ],
            axis=0,
        )
        return mask

    def _get_full_obs_mask(self, hi_res):
        """Define observation mask for the current batch. This differs from
        ``_get_obs_mask`` by defining a composite mask based on separate
        onshore and offshore masks. This is because there is often more
        observation data available onshore than offshore."""
        on_sf = self.onshore_obs_frac.get('spatial', 0.0)
        on_tf = self.onshore_obs_frac.get('time', 1.0)
        obs_mask = self._get_obs_mask(hi_res, on_sf, on_tf)
        if 'topography' in self.hr_features and self.offshore_obs_frac:
            topo_idx = self.hr_features.index('topography')
            topo = hi_res[..., topo_idx]
            off_sf = self.offshore_obs_frac.get('spatial', 0.0)
            off_tf = self.offshore_obs_frac.get('time', 1.0)
            offshore_mask = self._get_obs_mask(hi_res, off_sf, off_tf)
            obs_mask = tf.where(topo[..., None] > 0, obs_mask, offshore_mask)
        return obs_mask

    @property
    def model_params(self):
        """
        Model parameters, used to save model to disc

        Returns
        -------
        dict
        """
        params = super().model_params
        params['onshore_obs_frac'] = self.onshore_obs_frac
        params['offshore_obs_frac'] = self.offshore_obs_frac
        params['loss_obs_weight'] = self.loss_obs_weight
        params['loss_obs'] = self.loss_obs_name
        return params

    @tf.function
    def get_hr_exo_input(self, hi_res_true):
        """Mask high res data to act as sparse observation data. Add this to
        the standard high res exo input"""
        exo_data = super().get_hr_exo_input(hi_res_true)
        if len(self.obs_features) == 0:
            return exo_data

        if self.use_proxy_obs:
            obs, obs_mask = self._get_proxy_obs(hi_res_true)
        else:
            obs, obs_mask = self._get_real_obs(hi_res_true)

        obs = tf.expand_dims(obs, axis=-2)
        exo_obs = dict(zip(self.obs_features, tf.unstack(obs, axis=-1)))
        exo_data.update(exo_obs)
        exo_data['mask'] = obs_mask
        return exo_data

    def _get_real_obs(self, hi_res_true):
        """Get real observation data and the corresponding mask from the
        .high_res attribute of the batches in the batch handler. This is used
        when not training with proxy observations."""
        obs = tf.gather(hi_res_true, self.obs_training_inds, axis=-1)
        obs_mask = tf.math.is_nan(obs)
        return obs, obs_mask

    def _get_proxy_obs(self, hi_res_true):
        """Get proxy observation data by masking the true high res data."""
        obs_mask = self._get_full_obs_mask(hi_res_true)
        nan_const = tf.constant(float('nan'), dtype=hi_res_true.dtype)
        obs = tf.gather(hi_res_true, self.obs_training_inds, axis=-1)
        obs = tf.where(obs_mask[..., : obs.shape[-1]], nan_const, obs)
        return obs, obs_mask

    def _get_hr_exo_and_loss(
        self,
        low_res,
        hi_res_true,
        **calc_loss_kwargs,
    ):
        """Get high-resolution exogenous data, generate synthetic output, and
        compute loss. Includes artificially masking hi res data to act as
        sparse observation data."""
        out = super()._get_hr_exo_and_loss(
            low_res, hi_res_true, **calc_loss_kwargs
        )
        loss, loss_details, hi_res_gen, hi_res_exo = out

        if calc_loss_kwargs.get('train_gen', True):
            loss_obs, loss_non_obs = self._get_loss_obs_comparison(
                hi_res_true,
                hi_res_gen,
                hi_res_exo['mask'],
            )
            n_obs = tf.reduce_sum(tf.cast(~hi_res_exo['mask'], tf.float32))
            n_total = tf.cast(tf.size(hi_res_exo['mask']), tf.float32)
            obs_frac = n_obs / n_total
            loss_update = {
                'loss_obs': loss_obs,
                'loss_non_obs': loss_non_obs,
                'obs_frac': obs_frac,
            }
            if self.loss_obs_weight and obs_frac > 0:
                loss_obs *= self.loss_obs_weight
                loss += loss_obs
                loss_details['loss_gen'] += loss_obs
                loss_details['loss_gen_content'] += loss_obs
            loss_details.update(loss_update)
        return loss, loss_details, hi_res_gen, hi_res_exo

    def _post_batch(self, ib, b_loss_details, n_batches, previous_means):
        """Update loss details after the current batch and write to log."""
        if 'obs_frac' in b_loss_details:
            logger.debug(
                'Batch {} out of {} has obs_frac: {:.4e}'.format(
                    ib + 1, n_batches, b_loss_details['obs_frac']
                )
            )
        return super()._post_batch(
            ib, b_loss_details, n_batches, previous_means
        )
