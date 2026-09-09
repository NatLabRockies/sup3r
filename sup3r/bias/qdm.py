"""QDM related methods to correct and bias and trend

Procedures to apply Quantile Delta Method correction and derived methods such
as PresRat.
"""

import copy
import json
import logging
import os

import h5py
import numpy as np
from rex.utilities.bc_utils import (
    sample_q_invlog,
    sample_q_linear,
    sample_q_log,
)

from sup3r.preprocessing.utilities import expand_paths

from .abstract import AbstractBiasCorrection
from .base import DataRetrievalBase
from .mixins import FillAndSmoothMixin

logger = logging.getLogger(__name__)


class QuantileDeltaMappingCorrection(
    AbstractBiasCorrection, FillAndSmoothMixin, DataRetrievalBase
):
    """Estimate probability distributions required by Quantile Delta Mapping

    The main purpose of this class is to estimate the probability
    distributions required by Quantile Delta Mapping (QDM) ([Cannon2015]_)
    technique. Therefore, the name 'Correction' can be misleading since it is
    not the correction *per se*, but that was used to keep consistency within
    this module.

    The QDM technique corrects bias and trend by comparing the data
    distributions of three datasets: a historical reference, a biased
    reference, and a biased target to correct (in Cannon et. al. (2015)
    called: historical observed, historical modeled, and future modeled
    respectively). Those three probability distributions provided here
    can be, for instance, used by
    :func:`~sup3r.bias.bias_transforms.local_qdm_bc` to actually correct
    a dataset.
    """

    def __init__(
        self,
        base_fps,
        bias_fps,
        bias_fut_fps,
        base_dset,
        bias_feature,
        *,
        distance_upper_bound=None,
        target=None,
        shape=None,
        base_handler='Resource',
        bias_handler='DataHandlerNCforCC',
        base_handler_kwargs=None,
        bias_handler_kwargs=None,
        bias_fut_handler_kwargs=None,
        decimals=None,
        match_zero_rate=False,
        n_quantiles=101,
        dist='empirical',
        relative=True,
        sampling='linear',
        log_base=10,
        n_time_steps=24,
        window_size=120,
        pre_load=True,
    ):
        """
        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use
            to correct the biased dataset (observed historical in Cannon et.
            al. (2015)). This is typically several years of WTK or NSRDB files.
        bias_fps : list | str
            One or more biased .nc or .h5 filepaths representing the biased
            data
            to be compared with the baseline data (modeled historical in Cannon
            et. al. (2015)). This is typically several years of GCM .nc files.
        bias_fut_fps : list | str
            Consistent data to `bias_fps` but for a different time period
            (modeled
            future in Cannon et. al. (2015)). This is the dataset that would be
            corrected, while `bias_fsp` is used to provide a transformation map
            with the baseline data.
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be U_100m or V_100m which will retrieve
            windspeed and winddirection and derive the U/V component.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset
        distance_upper_bound : float
            Upper bound on the nearest neighbor distance in decimal degrees.
            This should be the approximate resolution of the low-resolution
            bias data. None (default) will calculate this based on the median
            distance between points in bias_fps
        target : tuple
            (lat, lon) lower left corner of raster to retrieve from bias_fps.
            If None then the lower left corner of the full domain will be used.
        shape : tuple
            (rows, cols) grid size to retrieve from bias_fps. If None then the
            full domain shape will be used.
        base_handler : str
            Name of rex resource handler or sup3r.preprocessing.data_handlers
            class to be retrieved from the rex/sup3r library. If a
            sup3r.preprocessing.data_handlers class is used, all data will be
            loaded in this class' initialization and the subsequent bias
            calculation will be done in serial
        bias_handler : str
            Name of the bias data handler class to be retrieved from the
            sup3r.preprocessing.data_handlers library.
        base_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the base_handler
            class
        bias_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the bias_handler
            class with the bias_fps
        bias_fut_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the
            bias_handler class with the bias_fut_fps
        decimals : int | None
            Option to round bias and base data to this number of
            decimals, this gets passed to np.around(). If decimals
            is negative, it specifies the number of positions to
            the left of the decimal point.
        match_zero_rate : bool
            Option to fix the frequency of zero values in the biased data. The
            lowest percentile of values in the biased data will be set to zero
            to match the percentile of zeros in the base data. If
            SkillAssessment is being run and this is True, the distributions
            will not be mean-centered. This helps resolve the issue where
            global climate models produce too many days with small
            precipitation totals e.g., the "drizzle problem" [Polade2014]_.
        dist : str, default="empirical",
            Define the type of distribution, which can be "empirical" or any
            parametric distribution defined in "scipy".
        n_quantiles : int, default=101
            Defines the number of quantiles (between 0 and 1) for an empirical
            distribution.
        sampling : str, default="linear",
            Defines how the quantiles are sampled. For instance, 'linear' will
            result in a linearly spaced quantiles. Other options are: 'log'
            and 'invlog'.
        log_base : int or float, default=10
            Log base value if sampling is "log" or "invlog".
        n_time_steps : int
            Number of times to calculate QDM parameters equally distributed
            along a year. For instance, `n_time_steps=1` results in a single
            set of parameters while `n_time_steps=12` is approximately every
            month.
        window_size : int
            Total time window period in days to be considered for each time QDM
            is calculated. For instance, `window_size=30` with
            `n_time_steps=12` would result in approximately monthly estimates.
        pre_load : bool
            Flag to preload all data needed for bias correction. This is
            currently recommended to improve performance with the new sup3r
            data handler access patterns

        See Also
        --------
        sup3r.bias.bias_transforms.local_qdm_bc :
            Bias correction using QDM.
        sup3r.preprocessing.data_handlers.DataHandler :
            Bias correction using QDM directly from a derived handler.
        rex.utilities.bc_utils.QuantileDeltaMapping
            Quantile Delta Mapping method and support functions. Since
            :mod:`rex.utilities.bc_utils` is used here, the arguments
            ``dist``, ``n_quantiles``, ``sampling``, and ``log_base``
            must be consitent with that package/module.

        Notes
        -----
        One way of using this class is by saving the distributions definitions
        obtained here with the method :meth:`.write_outputs` and then use that
        file with :func:`~sup3r.bias.bias_transforms.local_qdm_bc` or through
        a derived :class:`~sup3r.preprocessing.data_handlers.DataHandler`.
        **ATTENTION**, be careful handling that file of parameters. There is
        no checking process and one could missuse the correction estimated for
        the wrong dataset.

        References
        ----------
        .. [Cannon2015] Cannon, A. J., Sobie, S. R., & Murdock, T. Q. (2015).
           Bias correction of GCM precipitation by quantile mapping: how well
           do methods preserve changes in quantiles and extremes?. Journal of
           Climate, 28(17), 6938-6959.
        """

        self.n_quantiles = n_quantiles
        self.dist = dist
        self.relative = relative
        self.sampling = sampling
        self.log_base = log_base
        self.n_time_steps = n_time_steps
        self.window_size = window_size

        super().__init__(
            base_fps=base_fps,
            bias_fps=bias_fps,
            base_dset=base_dset,
            bias_feature=bias_feature,
            distance_upper_bound=distance_upper_bound,
            target=target,
            shape=shape,
            base_handler=base_handler,
            bias_handler=bias_handler,
            base_handler_kwargs=base_handler_kwargs,
            bias_handler_kwargs=bias_handler_kwargs,
            decimals=decimals,
            match_zero_rate=match_zero_rate,
            pre_load=False,
        )

        self.bias_fut_fps = bias_fut_fps
        self.bias_fut_fps = expand_paths(self.bias_fut_fps)

        self.bias_fut_handler_kwargs = bias_fut_handler_kwargs or {}
        self.bias_fut_dh = self.bias_handler(
            self.bias_fut_fps,
            [self.bias_feature],
            target=self.target,
            shape=self.shape,
            **self.bias_fut_handler_kwargs,
        )

        if pre_load:
            self.pre_load()

    def pre_load(self):
        """Preload all data needed for bias correction. This is currently
        recommended to improve performance with the new sup3r data handler
        access patterns"""
        super().pre_load()
        if hasattr(self.bias_fut_dh, 'compute'):
            logger.info('Pre loading future biased data into memory...')
            self.bias_fut_dh.data.compute()
            logger.info('Finished pre loading future biased data.')

    def _init_out(self):
        """Initialize output arrays `self.out`

        Three datasets are saved here with information to reconstruct the
        probability distributions for the three datasets (see class
        documentation).
        """
        keys = [
            f'bias_{self.bias_feature}_params',
            f'bias_fut_{self.bias_feature}_params',
            f'base_{self.base_dset}_params',
        ]
        shape = (
            *self.bias_gid_raster.shape,
            self.n_time_steps,
            self.n_quantiles,
        )
        arr = np.full(shape, np.nan, np.float32)
        self.out = {k: arr.copy() for k in keys}

        self.time_window_center = self._window_center(self.n_time_steps)

    @staticmethod
    def _window_center(ntimes: int):
        """A sequence of equally spaced `ntimes` day of year

        This is used to identify the center of moving windows to apply filters
        and masks. For instance, if `ntimes` equal to 12, it would return
        12 equal periods, i.e. approximately the months' center time.

        This is conveniently shifted one half time interval so that December
        31st would be closest to the last interval of the year instead of the
        first.

        Leap years are neglected, but if processed here, the doy 366 is
        included in the relevant windows as an extra data, which would
        cause a negligible error.

        Parameters
        ----------
        ntimes : int
            Number of intervals in one year. Choose 12 for approximately
            monthly time scales.

        Returns
        -------
        np.ndarray :
            Sequence of days of the year equally spaced and shifted by half
            window size, thus `ntimes`=12 results in approximately [15, 45,
            ...]. It includes the fraction of a day, thus 15.5 is equivalent
            to January 15th, 12:00h.
        """
        assert ntimes > 0, 'Requires a positive number of intervals'

        dt = 365 / ntimes
        return np.arange(dt / 2, 366, dt)

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
        base_dh_inst=None,
    ):
        """Estimate probability distributions at a single site"""

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

        for nt, idt in enumerate(window_center):
            base_idx = cls.window_mask(base_ti.day_of_year, idt, window_size)
            bias_idx = cls.window_mask(bias_ti.day_of_year, idt, window_size)
            bias_fut_idx = cls.window_mask(
                bias_fut_ti.day_of_year, idt, window_size
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

        return out

    @staticmethod
    def get_qdm_params(
        bias_data,
        bias_fut_data,
        base_data,
        bias_feature,
        base_dset,
        sampling,
        n_samples,
        log_base,
    ):
        """Get quantiles' cut point for given datasets

        Estimate the quantiles' cut points for each of the three given
        datasets. Lacking a good analytical approximation, such as one of
        the parametric distributions, those quantiles can be used to
        approximate the statistical distribution of those datasets.

        Parameters
        ----------
        bias_data : np.ndarray
            1D array of biased data observations.
        bias_fut_data : np.ndarray
            1D array of biased data observations.
        base_data : np.ndarray
            1D array of base data observations.
        bias_feature : str
            This is the biased feature from bias_fps to retrieve. This should
            be a single feature name corresponding to base_dset.
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be U_100m or V_100m which will retrieve
            windspeed and winddirection and derive the U/V component.
        sampling : str
            Defines how the quantiles are sampled. For instance, 'linear' will
            result in a linearly spaced quantiles. Other options are: 'log'
            and 'invlog'.
        n_samples : int
            Number of points to sample between 0 and 1, i.e. number of
            quantiles.
        log_base : int | float
            Log base value.

        Returns
        -------
        out : dict
            Dictionary of the quantiles' cut points. Note that to make sense
            of those cut point values, one need to know the given arguments
            such as `log_base`. For instance, the sequence [-1, 0, 2] are,
            if sampling was linear, the minimum, median, and maximum values
            respectively. The expected keys are "bias_{bias_feature}_params",
            "bias_fut_{bias_feature}_params", and "base_{base_dset}_params".

        See Also
        --------
        rex.utilities.bc_utils : Sampling scales, such as `sample_q_linear()`
        """

        if sampling == 'linear':
            quantiles = sample_q_linear(n_samples)
        elif sampling == 'log':
            quantiles = sample_q_log(n_samples, log_base)
        elif sampling == 'invlog':
            quantiles = sample_q_invlog(n_samples, log_base)
        else:
            msg = (
                'sampling option must be linear, log, or invlog, but '
                'received: {}'.format(sampling)
            )
            logger.error(msg)
            raise KeyError(msg)

        out = {
            f'bias_{bias_feature}_params': np.quantile(bias_data, quantiles),
            f'bias_fut_{bias_feature}_params': np.quantile(
                bias_fut_data, quantiles
            ),
            f'base_{base_dset}_params': np.quantile(base_data, quantiles),
        }

        return out

    def write_outputs(self, fp_out, out=None):
        """Write outputs to an .h5 file.

        Parameters
        ----------
        fp_out : str | None
            An HDF5 filename to write the estimated statistical distributions.
        out : dict, optional
            A dictionary with the three statistical distribution parameters.
            If not given, it uses :attr:`.out`.
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
                f.attrs['time_window_center'] = self.time_window_center
                logger.info('Wrote quantiles to file: %s', fp_out)

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
                bias_fut_data = self.get_bias_data(bias_gid, self.bias_fut_dh)
                task_kwargs[bias_gid] = {
                    'bias_data': bias_data,
                    'bias_fut_data': bias_fut_data,
                    'base_fps': self.base_fps,
                    'bias_feature': self.bias_feature,
                    'base_dset': self.base_dset,
                    'base_gid': base_gid,
                    'base_handler': self.base_handler,
                    'bias_ti': self.bias_dh.time_index,
                    'bias_fut_ti': self.bias_fut_dh.time_index,
                    'decimals': self.decimals,
                    'dist': self.dist,
                    'relative': self.relative,
                    'sampling': self.sampling,
                    'n_samples': self.n_quantiles,
                    'log_base': self.log_base,
                    'n_time_steps': self.n_time_steps,
                    'window_size': self.window_size,
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
        """Estimate the statistical distributions for each location

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
        )

        self.write_outputs(fp_out, self.out)
        return copy.deepcopy(self.out)

    @staticmethod
    def window_mask(doy, d0, window_size):
        """An index of elements within a given time window

        Create an index of days of the year within the target time window. It
        only considers the day of the year (doy), hence, it is limited to time
        scales smaller than annual.

        Parameters
        ----------
        doy : np.ndarray
            An unordered array of days of year, i.e. January 1st is 1.
        d0 : int
            Center point of the target window [day of year].
        window_size : float
            Total size of the target window, i.e. the window covers half this
            value on each side of d0. Note that it has the same units of doy,
            thus it is equal to the number of points only if doy is daily.

        Returns
        -------
        np.array
            An boolean array with the same shape of the given `doy`, where
            True means that position is within the target window.

        Notes
        -----
        Leap years have the day 366 included in the output index, but a
        precise shift is not calculated, resulting in a negligible error
        in large datasets.
        """
        d_start = d0 - window_size / 2
        d_end = d0 + window_size / 2

        if d_start < 0:
            idx = (doy > 365 + d_start) | (doy < d_end)
        elif d_end > 365:
            idx = (doy > d_start) | (doy < d_end - 365)
        else:
            idx = (doy > d_start) & (doy < d_end)

        return idx
