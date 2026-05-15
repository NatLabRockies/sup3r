"""Interpolator class with methods for pressure and height interpolation"""

import logging
from typing import Union
from warnings import warn

import dask.array as da
import numpy as np

logger = logging.getLogger(__name__)


class Interpolator:
    """Class for handling pressure and height interpolation"""

    @classmethod
    def get_level_masks(cls, lev_array, level):
        """Get the masks used to select closest surrounding levels in the
        lev_array to the requested interpolation level. If there are levels
        above and below the requested level these are prioritized.

        Parameters
        ----------
        lev_array : Union[np.ndarray, da.core.Array]
            Height or pressure values for the corresponding entries in
            var_array, in the same shape as var_array. If this is height and
            the requested levels are hub heights above surface, lev_array
            should be the geopotential height corresponding to every var_array
            index relative to the surface elevation (subtract the elevation at
            the surface from the geopotential height)
        level : float
            level to interpolate to (e.g. final desired hub height
            above surface elevation)

        Returns
        -------
        mask1 : Union[np.ndarray, da.core.Array]
            Array of bools selecting the entries with the closest levels to the
            one requested.
            (lat, lon, time, level)
        mask2 : Union[np.ndarray, da.core.Array]
            Array of bools selecting the entries with the second closest levels
            to the one requested.
            (lat, lon, time, level)
        """
        lev_indices = da.broadcast_to(
            da.arange(lev_array.shape[-1]), lev_array.shape
        )

        above_mask = lev_array >= level
        below_mask = lev_array < level
        below = da.ma.masked_array(lev_array, above_mask)
        above = da.ma.masked_array(lev_array, below_mask)

        argmin1 = da.argmin(np.abs(below - level), axis=-1, keepdims=True)
        mask1 = lev_indices == argmin1
        argmin2 = da.argmin(np.abs(above - level), axis=-1, keepdims=True)
        mask2 = lev_indices == argmin2

        # Get alternative levels in case there is no level below or above
        below_exists = da.any(below_mask, axis=-1, keepdims=True)
        argmin3 = da.argmin(np.abs(lev_array - level), axis=-1, keepdims=True)
        mask1 = da.where(below_exists, mask1, lev_indices == argmin3)

        above_exists = da.any(above_mask, axis=-1, keepdims=True)
        alts = da.ma.masked_array(lev_array, mask1)
        argmin3 = da.argmin(np.abs(alts - level), axis=-1, keepdims=True)
        mask2 = da.where(above_exists, mask2, lev_indices == argmin3)

        return mask1, mask2

    @classmethod
    def _lin_interp(cls, lev_samps, var_samps, level):
        """Linearly interpolate between levels."""
        diff = da.map_blocks(lambda x, y: x - y, lev_samps[1], lev_samps[0])
        alpha = da.where(
            np.abs(diff) < 1e-3,  # to avoid excessively large alpha values
            0,
            da.map_blocks(lambda x, y: x / y, (level - lev_samps[0]), diff),
        )
        indices = 'ijk'[: lev_samps[0].ndim]
        out = da.blockwise(
            lambda x, y, a: x * (1 - a) + y * a,
            indices,
            var_samps[0],
            indices,
            var_samps[1],
            indices,
            alpha,
            indices,
        )
        return out

    @classmethod
    def _log_interp(cls, lev_samps, var_samps, level):
        """Interpolate between levels with log profile.

        Note
        ----
        Here we fit the function a * log(h - h0 + 1) + v0 to the two given
        levels and variable values. So a is calculated with `v1 = a * log(h1 -
        h0 + 1) + v0` where v1, v0 are var_samps[0], var_samps[1] and h1, h0
        are lev_samps[1], lev_samps[0]
        """
        mask = lev_samps[0] < lev_samps[1]
        h0 = da.where(mask, lev_samps[0], lev_samps[1])
        h1 = da.where(mask, lev_samps[1], lev_samps[0])
        v0 = da.where(mask, var_samps[0], var_samps[1])
        v1 = da.where(mask, var_samps[1], var_samps[0])
        coeff = da.where(h1 == h0, 0, (v1 - v0) / np.log(h1 - h0 + 1))
        coeff = da.where(level < h0, -coeff, coeff)
        return coeff * np.log(da.abs(level - h0) + 1) + v0

    @classmethod
    def interp_to_level(
        cls,
        lev_array: Union[np.ndarray, da.core.Array],
        var_array: Union[np.ndarray, da.core.Array],
        level,
        interp_kwargs=None,
    ):
        """Interpolate var_array to the given level.

        Parameters
        ----------
        lev_array : xr.DataArray
            Height or pressure values for the corresponding entries in
            var_array, in the same shape as var_array. If this is height and
            the requested levels are hub heights above surface, lev_array
            should be the geopotential height corresponding to every var_array
            index relative to the surface elevation (subtract the elevation at
            the surface from the geopotential height)
        var_array : xr.DataArray
            Array of variable data, for example u-wind in a 4D array of shape
            (lat, lon, time, level)
        level : float
            level or levels to interpolate to (e.g. final desired hub height
            above surface elevation)
        interp_kwargs: dict | None
            Dictionary of kwargs for level interpolation. Can include "method"
            and "run_level_check" keys

        Returns
        -------
        out : Union[np.ndarray, da.core.Array]
            Interpolated var_array (lat, lon, time)
        """
        interp_kwargs = interp_kwargs or {}
        interp_method = interp_kwargs.get('method', 'linear')
        run_level_check = interp_kwargs.get('run_level_check', False)

        if run_level_check:
            cls._check_lev_array(lev_array, levels=[level])
        levs = da.ma.masked_array(lev_array, da.isnan(lev_array))
        mask1, mask2 = cls.get_level_masks(levs, level)
        lev1 = da.where(mask1, lev_array, np.nan)
        lev2 = da.where(mask2, lev_array, np.nan)
        var1 = da.where(mask1, var_array, np.nan)
        var2 = da.where(mask2, var_array, np.nan)
        lev1 = np.nanmean(lev1, axis=-1)
        lev2 = np.nanmean(lev2, axis=-1)
        var1 = np.nanmean(var1, axis=-1)
        var2 = np.nanmean(var2, axis=-1)

        if interp_method == 'log':
            out = cls._log_interp(
                lev_samps=[lev1, lev2], var_samps=[var1, var2], level=level
            )
        else:
            out = cls._lin_interp(
                lev_samps=[lev1, lev2], var_samps=[var1, var2], level=level
            )
        return out

    @classmethod
    def _check_lev_array(cls, lev_array, levels):
        """Check if the requested levels are consistent with the given
        lev_array and if there are any nans in the lev_array."""

        if np.isnan(lev_array).all():
            msg = 'All pressure level height data is NaN!'
            logger.error(msg)
            raise RuntimeError(msg)

        nans = np.isnan(lev_array)
        logger.debug('Level array shape: %s', lev_array.shape)

        lowest_height = np.min(lev_array, axis=-1)
        highest_height = np.max(lev_array, axis=-1)
        bad_min = min(levels) < lowest_height
        bad_max = max(levels) > highest_height

        if nans.any():
            nans = np.asarray(nans)
            msg = (
                'Approximately {:.2f}% of the vertical level '
                'array is NaN. Data will be interpolated or extrapolated '
                'past these NaN values.'.format(100 * nans.sum() / nans.size)
            )
            logger.warning(msg)
            warn(msg)

        # This and the next if statement can return warnings in the case of
        # pressure inversions, in which case the "lowest" or "highest" pressure
        # does not correspond to the lowest or highest height. Interpolation
        # can be performed without issue in this case.
        if bad_min.any():
            bad_min = np.asarray(bad_min)
            lev_array = np.asarray(lev_array)
            msg = (
                'Approximately {:.2f}% of the lowest vertical levels '
                '(maximum value of {:.3f}, minimum value of {:.3f}) '
                'were greater than the minimum requested level: {}'.format(
                    100 * bad_min.sum() / bad_min.size,
                    np.nanmax(lev_array[..., 0]),
                    np.nanmin(lev_array[..., 0]),
                    min(levels),
                )
            )
            logger.warning(msg)
            warn(msg)

        if bad_max.any():
            bad_max = np.asarray(bad_max)
            lev_array = np.asarray(lev_array)
            msg = (
                'Approximately {:.2f}% of the highest vertical levels '
                '(minimum value of {:.3f}, maximum value of {:.3f}) '
                'were lower than the maximum requested level: {}'.format(
                    100 * bad_max.sum() / bad_max.size,
                    np.nanmin(lev_array[..., -1]),
                    np.nanmax(lev_array[..., -1]),
                    max(levels),
                )
            )
            logger.warning(msg)
            warn(msg)
