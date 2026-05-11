"""Utilities to calculate the bias correction factors for biased data that is
going to be fed into the sup3r downscaling models. This is typically used to
bias correct GCM data vs. some historical record like the WTK or NSRDB."""

import copy
import json
import logging
import os

import h5py
import numpy as np
from scipy import stats

from .abstract import AbstractBiasCorrection
from .base import DataRetrievalBase
from .mixins import FillAndSmoothMixin

logger = logging.getLogger(__name__)


class LinearCorrection(
    AbstractBiasCorrection, FillAndSmoothMixin, DataRetrievalBase
):
    """Calculate linear correction *scalar +adder factors to bias correct data

    This calculation operates on single bias sites for the full time series of
    available data (no season bias correction)
    """

    NT = 1
    """size of the time dimension, 1 is no time-based bias correction"""

    def _init_out(self):
        """Initialize output arrays"""
        keys = [
            f'{self.bias_feature}_scalar',
            f'{self.bias_feature}_adder',
            f'bias_{self.bias_feature}_mean',
            f'bias_{self.bias_feature}_std',
            f'base_{self.base_dset}_mean',
            f'base_{self.base_dset}_std',
        ]
        self.out = {
            k: np.full(
                (*self.bias_gid_raster.shape, self.NT), np.nan, np.float32
            )
            for k in keys
        }

    @staticmethod
    def get_linear_correction(bias_data, base_data, bias_feature, base_dset):
        """Get the linear correction factors based on 1D bias and base datasets

        Parameters
        ----------
        bias_data : np.ndarray
            1D array of biased data observations.
        base_data : np.ndarray
            1D array of base data observations.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be u_100m or v_100m which will retrieve
            windspeed and winddirection and derive the U/V component.

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder
        """

        bias_std = np.nanstd(bias_data)
        if bias_std == 0:
            bias_std = np.nanstd(base_data)

        scalar = np.nanstd(base_data) / bias_std
        adder = np.nanmean(base_data) - np.nanmean(bias_data) * scalar

        out = {
            f'bias_{bias_feature}_mean': np.nanmean(bias_data),
            f'bias_{bias_feature}_std': bias_std,
            f'base_{base_dset}_mean': np.nanmean(base_data),
            f'base_{base_dset}_std': np.nanstd(base_data),
            f'{bias_feature}_scalar': scalar,
            f'{bias_feature}_adder': adder,
        }

        return out

    # pylint: disable=W0613
    @classmethod
    def _run_single(
        cls,
        bias_data,
        base_fps,
        bias_feature,
        base_dset,
        base_gid,
        base_handler,
        daily_reduction,
        bias_ti,  # noqa: ARG003
        decimals,
        base_dh_inst=None,
        match_zero_rate=False,
    ):
        """Find the nominal scalar + adder combination to bias correct data
        at a single site"""

        base_data, _ = cls.get_base_data(
            base_fps,
            base_dset,
            base_gid,
            base_handler,
            daily_reduction=daily_reduction,
            decimals=decimals,
            base_dh_inst=base_dh_inst,
        )

        if match_zero_rate:
            bias_data = cls._match_zero_rate(bias_data, base_data)

        out = cls.get_linear_correction(
            bias_data, base_data, bias_feature, base_dset
        )
        return out

    def write_outputs(self, fp_out, out):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        """

        if fp_out is not None:
            if not os.path.exists(os.path.dirname(fp_out)):
                os.makedirs(os.path.dirname(fp_out), exist_ok=True)

            with h5py.File(fp_out, 'w') as f:
                # pylint: disable=E1136
                lat = self.bias_dh.lat_lon[..., 0]
                lon = self.bias_dh.lat_lon[..., 1]
                f.create_dataset('latitude', data=lat)
                f.create_dataset('longitude', data=lon)
                for dset, data in out.items():
                    f.create_dataset(dset, data=data)

                for k, v in self.meta.items():
                    f.attrs[k] = json.dumps(v)

                logger.info('Wrote scalar adder factors to file: %s', fp_out)

    def _get_run_kwargs(self, **kwargs_extras):
        """Get dictionary of kwarg dictionaries to use for calls to
        ``_run_single``. Each key-value pair is a bias_gid with the associated
        ``_run_single`` arguments for that gid"""
        task_kwargs = {}
        for bias_gid in self.bias_meta.index:
            _, base_gid = self.get_base_gid(bias_gid)

            if not base_gid.any():
                self.bad_bias_gids.append(bias_gid)
            else:
                bias_data = self.get_bias_data(bias_gid)
                task_kwargs[bias_gid] = {
                    'bias_data': bias_data,
                    'base_fps': self.base_fps,
                    'bias_feature': self.bias_feature,
                    'base_dset': self.base_dset,
                    'base_gid': base_gid,
                    'base_handler': self.base_handler,
                    'bias_ti': self.bias_ti,
                    'decimals': self.decimals,
                    'match_zero_rate': self.match_zero_rate,
                    **kwargs_extras,
                }
        return task_kwargs

    def run(
        self,
        fp_out=None,
        max_workers=None,
        daily_reduction='avg',
        fill_extend=True,
        smooth_extend=0,
        smooth_interior=0,
    ):
        """Run linear correction factor calculations for every site in the bias
        dataset

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        max_workers : int
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)
        fill_extend : bool
            Flag to fill data past distance_upper_bound using spatial nearest
            neighbor. If False, the extended domain will be left as NaN.
        smooth_extend : float
            Option to smooth the scalar/adder data outside of the spatial
            domain set by the distance_upper_bound input. This alleviates the
            weird seams far from the domain of interest. This value is the
            standard deviation for the gaussian_filter kernel
        smooth_interior : float
            Option to smooth the scalar/adder data within the valid spatial
            domain.  This can reduce the affect of extreme values within
            aggregations over large number of pixels.

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder. Each value is of shape
            (lat, lon, time).
        """
        logger.debug('Starting linear correction calculation...')

        logger.info(
            'Initialized scalar / adder with shape: %s',
            self.bias_gid_raster.shape,
        )
        self.out = self._run(
            out=self.out,
            max_workers=max_workers,
            daily_reduction=daily_reduction,
            fill_extend=fill_extend,
            smooth_extend=smooth_extend,
            smooth_interior=smooth_interior,
        )
        self.write_outputs(fp_out, self.out)

        return copy.deepcopy(self.out)


class ScalarCorrection(LinearCorrection):
    """Calculate annual linear correction *scalar factors to bias correct data.
    This typically used when base data is just monthly or annual means and
    standard deviations cannot be computed. This is case for vortex data, for
    example. Thus, just scalar factors are computed as mean(base_data) /
    mean(bias_data). Adder factors are still written but are exactly zero.

    This calculation operates on single bias sites on a monthly basis
    """

    @staticmethod
    def get_linear_correction(bias_data, base_data, bias_feature, base_dset):
        """Get the linear correction factors based on 1D bias and base datasets

        Parameters
        ----------
        bias_data : np.ndarray
            1D array of biased data observations.
        base_data : np.ndarray
            1D array of base data observations.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be u_100m or v_100m which will retrieve
            windspeed and winddirection and derive the U/V component.

        Returns
        -------
        out : dict
            Dictionary of values defining the mean/std of the bias + base
            data and the scalar + adder factors to correct the biased data
            like: bias_data * scalar + adder
        """

        bias_std = np.nanstd(bias_data)
        if bias_std == 0:
            bias_std = np.nanstd(base_data)

        base_mean = np.nanmean(base_data)
        bias_mean = np.nanmean(bias_data)
        scalar = base_mean / bias_mean
        adder = np.zeros(scalar.shape)

        out = {
            f'bias_{bias_feature}_mean': bias_mean,
            f'base_{base_dset}_mean': base_mean,
            f'{bias_feature}_scalar': scalar,
            f'{bias_feature}_adder': adder,
        }

        return out


class MonthlyLinearCorrection(LinearCorrection):
    """Calculate linear correction *scalar +adder factors to bias correct data

    This calculation operates on single bias sites on a monthly basis
    """

    NT = 12
    """size of the time dimension, 12 is monthly bias correction"""

    @classmethod
    def _run_single(
        cls,
        bias_data,
        base_fps,
        bias_feature,
        base_dset,
        base_gid,
        base_handler,
        daily_reduction,
        bias_ti,
        decimals,
        base_dh_inst=None,
        match_zero_rate=False,
    ):
        """Find the nominal scalar + adder combination to bias correct data
        at a single site"""

        base_data, base_ti = cls.get_base_data(
            base_fps,
            base_dset,
            base_gid,
            base_handler,
            daily_reduction=daily_reduction,
            decimals=decimals,
            base_dh_inst=base_dh_inst,
        )

        if match_zero_rate:
            bias_data = cls._match_zero_rate(bias_data, base_data)

        base_arr = np.full(cls.NT, np.nan, dtype=np.float32)
        out = {}

        for month in range(1, 13):
            bias_mask = bias_ti.month == month
            base_mask = base_ti.month == month

            if any(bias_mask) and any(base_mask):
                mout = cls.get_linear_correction(
                    bias_data[bias_mask],
                    base_data[base_mask],
                    bias_feature,
                    base_dset,
                )
                for k, v in mout.items():
                    if k not in out:
                        out[k] = base_arr.copy()
                    out[k][month - 1] = v

        return out


class MonthlyScalarCorrection(MonthlyLinearCorrection, ScalarCorrection):
    """Calculate linear correction *scalar factors for each month"""

    NT = 12


class SkillAssessment(MonthlyLinearCorrection):
    """Calculate historical skill of one dataset compared to another."""

    PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
    """Data percentiles to report."""

    def _init_out(self):
        """Initialize output arrays"""

        monthly_keys = [
            f'{self.bias_feature}_scalar',
            f'{self.bias_feature}_adder',
            f'bias_{self.bias_feature}_mean_monthly',
            f'bias_{self.bias_feature}_std_monthly',
            f'base_{self.base_dset}_mean_monthly',
            f'base_{self.base_dset}_std_monthly',
        ]

        annual_keys = [
            f'{self.bias_feature}_ks_stat',
            f'{self.bias_feature}_ks_p',
            f'{self.bias_feature}_bias',
            f'bias_{self.bias_feature}_mean',
            f'bias_{self.bias_feature}_std',
            f'bias_{self.bias_feature}_skew',
            f'bias_{self.bias_feature}_kurtosis',
            f'bias_{self.bias_feature}_zero_rate',
            f'base_{self.base_dset}_mean',
            f'base_{self.base_dset}_std',
            f'base_{self.base_dset}_skew',
            f'base_{self.base_dset}_kurtosis',
            f'base_{self.base_dset}_zero_rate',
        ]

        self.out = {
            k: np.full(
                (*self.bias_gid_raster.shape, self.NT), np.nan, np.float32
            )
            for k in monthly_keys
        }

        arr = np.full((*self.bias_gid_raster.shape, 1), np.nan, np.float32)
        for k in annual_keys:
            self.out[k] = arr.copy()

        for p in self.PERCENTILES:
            base_k = f'base_{self.base_dset}_percentile_{p}'
            bias_k = f'bias_{self.bias_feature}_percentile_{p}'
            self.out[base_k] = arr.copy()
            self.out[bias_k] = arr.copy()

    @classmethod
    def _run_skill_eval(
        cls,
        bias_data,
        base_data,
        bias_feature,
        base_dset,
        match_zero_rate=False,
    ):
        """Run skill assessment metrics on 1D datasets at a single site.

        Note we run the KS test on the mean=0 distributions as per:
        S. Brands et al. 2013 https://doi.org/10.1007/s00382-013-1742-8
        """

        if match_zero_rate:
            bias_data = cls._match_zero_rate(bias_data, base_data)

        out = {}
        bias_mean = np.nanmean(bias_data)
        base_mean = np.nanmean(base_data)
        out[f'{bias_feature}_bias'] = bias_mean - base_mean

        out[f'bias_{bias_feature}_mean'] = bias_mean
        out[f'bias_{bias_feature}_std'] = np.nanstd(bias_data)
        out[f'bias_{bias_feature}_skew'] = stats.skew(bias_data)
        out[f'bias_{bias_feature}_kurtosis'] = stats.kurtosis(bias_data)
        out[f'bias_{bias_feature}_zero_rate'] = np.nanmean(bias_data == 0)

        out[f'base_{base_dset}_mean'] = base_mean
        out[f'base_{base_dset}_std'] = np.nanstd(base_data)
        out[f'base_{base_dset}_skew'] = stats.skew(base_data)
        out[f'base_{base_dset}_kurtosis'] = stats.kurtosis(base_data)
        out[f'base_{base_dset}_zero_rate'] = np.nanmean(base_data == 0)

        if match_zero_rate:
            ks_out = stats.ks_2samp(base_data, bias_data)
        else:
            ks_out = stats.ks_2samp(
                base_data - base_mean, bias_data - bias_mean
            )

        out[f'{bias_feature}_ks_stat'] = ks_out.statistic
        out[f'{bias_feature}_ks_p'] = ks_out.pvalue

        for p in cls.PERCENTILES:
            base_k = f'base_{base_dset}_percentile_{p}'
            bias_k = f'bias_{bias_feature}_percentile_{p}'
            out[base_k] = np.percentile(base_data, p)
            out[bias_k] = np.percentile(bias_data, p)

        return out

    @classmethod
    def _run_single(
        cls,
        bias_data,
        base_fps,
        bias_feature,
        base_dset,
        base_gid,
        base_handler,
        daily_reduction,
        bias_ti,
        decimals,
        base_dh_inst=None,
        match_zero_rate=False,
    ):
        """Do a skill assessment at a single site"""

        base_data, base_ti = cls.get_base_data(
            base_fps,
            base_dset,
            base_gid,
            base_handler,
            daily_reduction=daily_reduction,
            decimals=decimals,
            base_dh_inst=base_dh_inst,
        )

        arr = np.full(cls.NT, np.nan, dtype=np.float32)
        out = {
            f'bias_{bias_feature}_mean_monthly': arr.copy(),
            f'bias_{bias_feature}_std_monthly': arr.copy(),
            f'base_{base_dset}_mean_monthly': arr.copy(),
            f'base_{base_dset}_std_monthly': arr.copy(),
        }

        out.update(
            cls._run_skill_eval(
                bias_data,
                base_data,
                bias_feature,
                base_dset,
                match_zero_rate=match_zero_rate,
            )
        )

        for month in range(1, 13):
            bias_mask = bias_ti.month == month
            base_mask = base_ti.month == month

            if any(bias_mask) and any(base_mask):
                mout = cls.get_linear_correction(
                    bias_data[bias_mask],
                    base_data[base_mask],
                    bias_feature,
                    base_dset,
                )
                for k, v in mout.items():
                    if not k.endswith(('_scalar', '_adder')):
                        k += '_monthly'
                        out[k][month - 1] = v

        return out
