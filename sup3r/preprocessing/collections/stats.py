"""Collection object with methods to compute and save stats."""

import logging
import os
import pprint
from warnings import warn

import numpy as np
import xarray as xr
from gaps.config import load_config

from sup3r.preprocessing.utilities import log_args, lowered
from sup3r.utilities.utilities import safe_serialize

from .base import Collection

logger = logging.getLogger(__name__)


class StatsCollection(Collection):
    """Extended collection object with methods for computing means and stds and
    saving these to files.

    Note
    ----
    We write stats as float64 because float32 is not json serializable
    """

    @log_args
    def __init__(self, containers, means=None, stds=None):
        """
        Parameters
        ----------
        containers: list[Rasterizer]
            List of containers to compute stats for.
        means : str | dict | None
            Usually a file path for saving results, or None for just
            calculating stats and not saving. Can also be a dict, which will
            just get returned as the "result".
        stds : str | dict | None
            Usually a file path for saving results, or None for just
            calculating stats and not saving. Can also be a dict, which will
            just get returned as the "result".
        """
        super().__init__(containers=containers)
        self.means = self.get_means(means)
        self.stds = self.get_stds(stds)
        self.save_stats(stds=stds, means=means)
        self.normalize(containers)

    def _get_stat(self, stat_type, needed_features='all'):
        """Get either mean or std for all features and all containers."""
        all_feats = lowered(
            self.features if needed_features == 'all' else needed_features
        )
        hr_feats = lowered(
            set(self.containers[0].high_res.features).intersection(all_feats)
        )
        lr_feats = set(all_feats) - set(hr_feats)

        cstats = [
            getattr(c.high_res[hr_feats], stat_type)(skipna=True)
            for c in self.containers
        ]
        lr_check = any(lr_feats)
        container_check = all(hasattr(c, 'low_res') for c in self.containers)
        if lr_check and not container_check:
            msg = (
                f'Found low-res features {lr_feats} but not all containers '
                'have low-res data. '
            )
            logger.error(msg)
            raise ValueError(msg)
        elif lr_check and container_check:
            cstats_lr = [
                getattr(c.low_res[lr_feats], stat_type)(skipna=True)
                for c in self.containers
            ]
            cstats = [
                xr.merge([c._ds, c_lr._ds])
                for c, c_lr in zip(cstats, cstats_lr)
            ]
        return cstats

    def _init_stats_dict(self, stats):
        """Initialize dictionary for stds or means from given input. Check if
        any existing stats are provided."""
        if isinstance(stats, str) and os.path.exists(stats):
            stats = load_config(stats)
        elif stats is None or isinstance(stats, str):
            stats = {}
        else:
            msg = (
                f'Received incompatible type {type(stats)}. Need a file '
                'path or dictionary'
            )
            assert isinstance(stats, dict), msg
        if (
            isinstance(stats, dict)
            and stats != {}
            and any(f not in stats for f in self.features)
        ):
            msg = (
                f'Not all features ({self.features}) are found in the '
                f'given stats dictionary {stats}. This is obviously from a '
                'prior run so you better be sure these stats carry over.'
            )
            logger.warning(msg)
            warn(msg)
        return stats

    def get_means(self, means):
        """Dictionary of means for each feature, computed across all data
        handlers."""
        means = self._init_stats_dict(means)
        needed_features = set(self.features) - set(means)
        if any(needed_features):
            logger.info(f'Getting means for {needed_features}.')
            cmeans = [
                cm * w
                for cm, w in zip(
                    self._get_stat('mean', needed_features),
                    self.container_weights,
                )
            ]
            for f in needed_features:
                logger.info(f'Computing mean for {f}.')
                means[f] = np.float32(np.nansum([cm[f] for cm in cmeans]))
        return means

    def get_stds(self, stds):
        """Dictionary of standard deviations for each feature, computed across
        all data handlers."""
        stds = self._init_stats_dict(stds)
        needed_features = set(self.features) - set(stds)
        if any(needed_features):
            logger.info(f'Getting stds for {needed_features}.')
            cstds = [
                w * cm**2
                for cm, w in zip(self._get_stat('std'), self.container_weights)
            ]
            for f in needed_features:
                logger.info(f'Computing std for {f}.')
                stds[f] = np.float32(
                    np.sqrt(np.nansum([cs[f] for cs in cstds]))
                )
        return stds

    @staticmethod
    def _added_stats(fp, stat_dict):
        """Check if stats were added to the given file or not."""
        return any(f not in load_config(fp) for f in stat_dict)

    def save_stats(self, stds, means):
        """Save stats to json files."""
        if isinstance(stds, str) and (
            not os.path.exists(stds) or self._added_stats(stds, self.stds)
        ):
            with open(stds, 'w') as f:
                f.write(safe_serialize(self.stds))
                logger.info(
                    f'Saved standard deviations {self.stds} to {stds}.'
                )
        if isinstance(means, str) and (
            not os.path.exists(means) or self._added_stats(means, self.means)
        ):
            with open(means, 'w') as f:
                f.write(safe_serialize(self.means))
                logger.info(f'Saved means {self.means} to {means}.')

    def normalize(self, containers):
        """Normalize container data with computed stats."""
        logger.debug(
            'Normalizing containers with:\n'
            f'means: {pprint.pformat(self.means, indent=2)}\n'
            f'stds: {pprint.pformat(self.stds, indent=2)}'
        )
        for i, c in enumerate(containers):
            logger.info(f'Normalizing container {i + 1}')
            c.normalize(means=self.means, stds=self.stds)
