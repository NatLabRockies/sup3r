"""QDM related methods to correct and bias and trend

Procedures to apply Quantile Delta Method correction and derived methods such
as PresRat.
"""

import copy
import json
import logging
import os
from typing import Optional

import h5py
import numpy as np
from rex.utilities.bc_utils import (
    QuantileDeltaMapping,
)

from .mixins import ZeroRateMixin
from .qdm import QuantileDeltaMappingCorrection

logger = logging.getLogger(__name__)


class PresRat(ZeroRateMixin, QuantileDeltaMappingCorrection):
    """PresRat bias correction method (precipitation)

    The PresRat correction [Pierce2015]_ is defined as the combination of
    three steps:
    * Use the model-predicted change ratio (with the CDFs);
    * The treatment of zero-precipitation days (with the fraction of dry days);
    * The final correction factor (K) to preserve the mean (ratio between both
    estimated means);

    To keep consistency with the full sup3r pipeline, PresRat was implemented
    as follows:

    1) Define zero rate from observations (oh)

    Using the historical observations, estimate the zero rate precipitation
    for each gridpoint. It is expected a long time series here, such as
    decadal or longer. A threshold larger than zero is an option here.

    The result is a 2D (space) `zero_rate` (non-dimensional).

    2) Find the threshold for each gridpoint (mh)

    Using the zero rate from the previous step, identify the magnitude
    threshold for each gridpoint that satisfies that dry days rate.

    Note that Pierce (2015) impose `tau` >= 0.01 mm/day for precipitation.

    The result is a 2D (space) threshold `tau` with the same dimensions
    of the data been corrected. For instance, it could be mm/day for
    precipitation.

    3) Define `Z_fg` using `tau` (mf)

    The `tau` that was defined with the *modeled historical*, is now
    used as a threshold on *modeled future* before any correction to define
    the equivalent zero rate in the future.

    The result is a 2D (space) rate (non-dimensional)

    4) Estimate `tau_fut` using `Z_fg`

    Since sup3r process data in smaller chunks, it wouldn't be possible to
    apply the rate `Z_fg` directly. To address that, all *modeled future*
    data is corrected with QDM, and applying `Z_fg` it is defined the
    `tau_fut`.

    References
    ----------
    .. [Pierce2015] Pierce, D. W., Cayan, D. R., Maurer, E. P., Abatzoglou, J.
       T., & Hegewisch, K. C. (2015). Improved bias correction techniques for
       hydrological simulations of climate change. Journal of Hydrometeorology,
       16(6), 2421-2442.
    """

    def _init_out(self):
        super()._init_out()

        shape = (*self.bias_gid_raster.shape, 1)
        self.out[f'{self.base_dset}_zero_rate'] = np.full(
            shape, np.nan, np.float32
        )
        self.out[f'{self.bias_feature}_tau_fut'] = np.full(
            shape, np.nan, np.float32
        )
        shape = (*self.bias_gid_raster.shape, self.n_time_steps)
        self.out[f'{self.bias_feature}_k_factor'] = np.full(
            shape, np.nan, np.float32
        )

    @classmethod
    def calc_tau_fut(
        cls,
        base_data,
        bias_data,
        bias_fut_data,
        corrected_fut_data,
        zero_rate_threshold=1.157e-7,
    ):
        """Calculate a precipitation threshold (tau) that preserves the
        model-predicted changes in fraction of dry days at a single spatial
        location.

        Parameters
        ----------
        base_data : np.ndarray
            A 1D array of (usually) daily precipitation observations from a
            historical dataset like Daymet
        bias_data : np.ndarray
            A 1D array of (usually) daily precipitation historical climate
            simulation outputs from a GCM dataset from CMIP6
        bias_future_data : np.ndarray
            A 1D array of (usually) daily precipitation future (e.g., ssp245)
            climate simulation outputs from a GCM dataset from CMIP6
        corrected_fut_data : np.ndarray
            Bias corrected bias_future_data usually with relative QDM
        zero_rate_threshold : float, default=1.157e-7
            Threshold value used to determine the zero rate in the observed
            historical dataset. For instance, 0.01 means that anything less
            than that will be considered negligible, hence equal to zero. Dai
            2006 defined this as 1mm/day. Pierce 2015 used 0.01mm/day. We
            recommend 0.01mm/day (1.157e-7 kg/m2/s).

        Returns
        -------
        tau_fut : float
            Precipitation threshold that will preserve the model predicted
            changes in fraction of dry days. Precipitation less than this value
            in the modeled future data can be set to zero.
        obs_zero_rate : float
            Rate of dry days in the observed historical data.
        """

        # Step 1: Define zero rate from observations
        assert base_data.ndim == 1
        obs_zero_rate = cls.zero_precipitation_rate(
            base_data, zero_rate_threshold
        )

        # Step 2: Find tau for each grid point
        # Removed NaN handling, thus reinforce finite-only data.
        assert np.isfinite(bias_data).all(), 'Unexpected invalid values'
        assert bias_data.ndim == 1, 'Assumed bias_data to be 1D'
        n_threshold = round(obs_zero_rate * bias_data.size)
        n_threshold = min(n_threshold, bias_data.size - 1)
        tau = np.sort(bias_data)[n_threshold]
        # Pierce (2015) imposes 0.01 mm/day
        # tau = max(tau, 0.01)

        # Step 3: Find Z_gf as the zero rate in mf
        assert np.isfinite(bias_fut_data).all(), 'Unexpected invalid values'
        z_fg = (bias_fut_data < tau).astype('i').sum() / bias_fut_data.size

        # Step 4: Estimate tau_fut with corrected mf
        tau_fut = np.sort(corrected_fut_data)[
            round(z_fg * corrected_fut_data.size)
        ]

        return tau_fut, obs_zero_rate

    @classmethod
    def calc_k_factor(
        cls,
        base_data,
        bias_data,
        bias_fut_data,
        corrected_fut_data,
        base_ti,
        bias_ti,
        bias_fut_ti,
        window_center,
        window_size,
        n_time_steps,
        zero_rate_threshold,
    ):
        """Calculate the K factor at a single spatial location that will
        preserve the original model-predicted mean change in precipitation

        Parameters
        ----------
        base_data : np.ndarray
            A 1D array of (usually) daily precipitation observations from a
            historical dataset like Daymet
        bias_data : np.ndarray
            A 1D array of (usually) daily precipitation historical climate
            simulation outputs from a GCM dataset from CMIP6
        bias_future_data : np.ndarray
            A 1D array of (usually) daily precipitation future (e.g., ssp245)
            climate simulation outputs from a GCM dataset from CMIP6
        corrected_fut_data : np.ndarray
            Bias corrected bias_future_data usually with relative QDM
        base_ti : pd.DatetimeIndex
            Datetime index associated with bias_data and of the same length
        bias_fut_ti : pd.DatetimeIndex
            Datetime index associated with bias_fut_data and of the same length
        window_center : np.ndarray
            Sequence of days of the year equally spaced and shifted by half
            window size, thus `ntimes`=12 results in approximately [15, 45,
            ...]. It includes the fraction of a day, thus 15.5 is equivalent
            to January 15th, 12:00h. Shape is (N,)
        window_size : int
            Total time window period in days to be considered for each time QDM
            is calculated. For instance, `window_size=30` with
            `n_time_steps=12` would result in approximately monthly estimates.
        n_time_steps : int
            Number of times to calculate QDM parameters equally distributed
            along a year. For instance, `n_time_steps=1` results in a single
            set of parameters while `n_time_steps=12` is approximately every
            month.
        zero_rate_threshold : float, default=1.157e-7
            Threshold value used to determine the zero rate in the observed
            historical dataset. For instance, 0.01 means that anything less
            than that will be considered negligible, hence equal to zero. Dai
            2006 defined this as 1mm/day. Pierce 2015 used 0.01mm/day. We
            recommend 0.01mm/day (1.157e-7 kg/m2/s).

        Returns
        -------
        k : np.ndarray
            K factor from the Pierce 2015 paper with shape (n_time_steps,)
            for a single spatial location.
        """

        k = np.full(n_time_steps, np.nan, np.float32)
        for nt, t in enumerate(window_center):
            base_idt = cls.window_mask(base_ti.day_of_year, t, window_size)
            bias_idt = cls.window_mask(bias_ti.day_of_year, t, window_size)
            bias_fut_idt = cls.window_mask(
                bias_fut_ti.day_of_year, t, window_size
            )

            oh = base_data[base_idt].mean()
            mh = bias_data[bias_idt].mean()
            mf = bias_fut_data[bias_fut_idt].mean()
            mf_unbiased = corrected_fut_data[bias_fut_idt].mean()

            oh = np.maximum(oh, zero_rate_threshold)
            mh = np.maximum(mh, zero_rate_threshold)
            mf = np.maximum(mf, zero_rate_threshold)
            mf_unbiased = np.maximum(mf_unbiased, zero_rate_threshold)

            x = mf / mh
            x_hat = mf_unbiased / oh
            k[nt] = x / x_hat
        return k

    # pylint: disable=W0613
    @classmethod
    def _run_single(
        cls,
        bias_data,
        bias_fut_data,
        base_fps,
        bias_feature,
        base_dset,
        base_gid,
        base_handler,
        daily_reduction,
        *,
        bias_ti,
        bias_fut_ti,
        decimals,
        dist,
        relative,
        sampling,
        n_samples,
        log_base,
        n_time_steps,
        window_size,
        zero_rate_threshold,
        base_dh_inst=None,
    ):
        """Estimate probability distributions at a single site

        TODO! This should be refactored. There is too much redundancy in
        the code. Let's make it work first, and optimize later.
        """
        base_data, base_ti = cls.get_base_data(
            base_fps,
            base_dset,
            base_gid,
            base_handler,
            daily_reduction=daily_reduction,
            decimals=decimals,
            base_dh_inst=base_dh_inst,
        )

        window_size = window_size or 365 / n_time_steps
        window_center = cls._window_center(n_time_steps)

        template = np.full((n_time_steps, n_samples), np.nan, np.float32)
        out = {}
        corrected_fut_data = np.full_like(bias_fut_data, np.nan)
        for nt, t in enumerate(window_center):
            # Define indices for which data goes in the current time window
            base_idx = cls.window_mask(base_ti.day_of_year, t, window_size)
            bias_idx = cls.window_mask(bias_ti.day_of_year, t, window_size)
            bias_fut_idx = cls.window_mask(
                bias_fut_ti.day_of_year, t, window_size
            )

            if any(base_idx) and any(bias_idx) and any(bias_fut_idx):
                tmp = cls.get_qdm_params(
                    bias_data[bias_idx],
                    bias_fut_data[bias_fut_idx],
                    base_data[base_idx],
                    bias_feature,
                    base_dset,
                    sampling,
                    n_samples,
                    log_base,
                )
                for k, v in tmp.items():
                    if k not in out:
                        out[k] = template.copy()
                    out[k][(nt), :] = v

            QDM = QuantileDeltaMapping(
                np.asarray(out[f'base_{base_dset}_params'][nt]),
                np.asarray(out[f'bias_{bias_feature}_params'][nt]),
                np.asarray(out[f'bias_fut_{bias_feature}_params'][nt]),
                dist=dist,
                relative=relative,
                sampling=sampling,
                log_base=log_base,
                delta_denom_min=zero_rate_threshold,
            )
            subset = bias_fut_data[bias_fut_idx]
            corrected_fut_data[bias_fut_idx] = QDM(subset).squeeze()

        tau_fut, obs_zero_rate = cls.calc_tau_fut(
            base_data,
            bias_data,
            bias_fut_data,
            corrected_fut_data,
            zero_rate_threshold,
        )
        k = cls.calc_k_factor(
            base_data,
            bias_data,
            bias_fut_data,
            corrected_fut_data,
            base_ti,
            bias_ti,
            bias_fut_ti,
            window_center,
            window_size,
            n_time_steps,
            zero_rate_threshold,
        )

        out[f'{bias_feature}_k_factor'] = k
        out[f'{base_dset}_zero_rate'] = obs_zero_rate
        out[f'{bias_feature}_tau_fut'] = tau_fut

        return out

    def run(
        self,
        fp_out=None,
        max_workers=None,
        daily_reduction='avg',
        fill_extend=True,
        smooth_extend=0,
        smooth_interior=0,
        zero_rate_threshold=1.157e-7,
    ):
        """Estimate the required information for PresRat correction

        Parameters
        ----------
        fp_out : str | None
            Optional .h5 output file to write scalar and adder arrays.
        max_workers : int, optional
            Number of workers to run in parallel. 1 is serial and None is all
            available.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)
        fill_extend : bool
            Whether to fill data extending beyond the base meta data with
            nearest neighbor values.
        smooth_extend : float
            Option to smooth the scalar/adder data outside of the spatial
            domain set by the threshold input. This alleviates the weird seams
            far from the domain of interest. This value is the standard
            deviation for the gaussian_filter kernel
        smooth_interior : float
            Value to use to smooth the scalar/adder data inside of the spatial
            domain set by the threshold input. This can reduce the effect of
            extreme values within aggregations over large number of pixels.
            This value is the standard deviation for the gaussian_filter
            kernel.
        zero_rate_threshold : float, default=1.157e-7
            Threshold value used to determine the zero rate in the observed
            historical dataset. For instance, 0.01 means that anything less
            than that will be considered negligible, hence equal to zero. Dai
            2006 defined this as 1mm/day. Pierce 2015 used 0.01mm/day. We
            recommend 0.01mm/day (1.157e-7 kg/m2/s).

        Returns
        -------
        out : dict
            Dictionary with parameters defining the statistical distributions
            for each of the three given datasets. Each value has dimensions
            (lat, lon, n-parameters).
        """
        logger.debug('Calculating CDF parameters for QDM')

        logger.info(
            'Initialized params with shape: %s', self.bias_gid_raster.shape
        )
        self.out = self._run(
            out=self.out,
            max_workers=max_workers,
            daily_reduction=daily_reduction,
            fill_extend=fill_extend,
            smooth_extend=smooth_extend,
            smooth_interior=smooth_interior,
            zero_rate_threshold=zero_rate_threshold,
        )

        extra_attrs = {
            'zero_rate_threshold': zero_rate_threshold,
            'time_window_center': self.time_window_center,
        }
        self.write_outputs(
            fp_out,
            self.out,
            extra_attrs=extra_attrs,
        )

        return copy.deepcopy(self.out)

    def write_outputs(
        self,
        fp_out: str,
        out: Optional[dict] = None,
        extra_attrs: Optional[dict] = None,
    ):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            An HDF5 filename to write the estimated statistical distributions.
        out : dict, optional
            A dictionary with the three statistical distribution parameters.
            If not given, it uses :attr:`.out`.
        extra_attrs: dict, optional
            Extra attributes to be exported together with the dataset.

        Examples
        --------
        >>> mycalc = PresRat(...)
        >>> mycalc.write_outputs(fp_out="myfile.h5", out=mydictdataset,
        ...   extra_attrs={'zero_rate_threshold': 0.01})
        """

        out = out or self.out

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
                f.attrs['dist'] = self.dist
                f.attrs['sampling'] = self.sampling
                f.attrs['log_base'] = self.log_base
                f.attrs['base_fps'] = self.base_fps
                f.attrs['bias_fps'] = self.bias_fps
                f.attrs['bias_fut_fps'] = self.bias_fut_fps
                if extra_attrs is not None:
                    for a, v in extra_attrs.items():
                        f.attrs[a] = v
                logger.info('Wrote quantiles to file: %s', fp_out)
