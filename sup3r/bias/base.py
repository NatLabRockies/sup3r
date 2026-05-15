"""Data retrieval class for performing / evaluating bias correction.

TODO: This can likely leverage the new data handling objects. Refactor
accordingly.
"""

import logging
from abc import abstractmethod

import numpy as np
import pandas as pd
import rex
from rex.utilities.fun_utils import get_fun_call_str
from scipy import stats
from scipy.spatial import KDTree

import sup3r.preprocessing
from sup3r.preprocessing import DataHandler
from sup3r.preprocessing.utilities import expand_paths
from sup3r.utilities import VERSION_RECORD, ModuleName
from sup3r.utilities.cli import BaseCLI

logger = logging.getLogger(__name__)


class DataRetrievalBase:
    """Base class to handle data retrieval for the biased data and the
    baseline data
    """

    def __init__(
        self,
        base_fps,
        bias_fps,
        base_dset,
        bias_feature,
        distance_upper_bound=None,
        target=None,
        shape=None,
        base_handler='Resource',
        bias_handler='DataHandlerNCforCC',
        base_handler_kwargs=None,
        bias_handler_kwargs=None,
        decimals=None,
        match_zero_rate=False,
        pre_load=True,
    ):
        """
        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use to correct the biased dataset. This is typically several years
            of WTK or NSRDB files.
        bias_fps : list | str
            One or more biased .nc or .h5 filepaths representing the biased
            data to be corrected based on the baseline data. This is typically
            several years of GCM .nc files.
        base_dset : str
            A single dataset from the base_fps to retrieve. In the case of wind
            components, this can be u_100m or v_100m which will retrieve
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
            Name of rex resource handler or sup3r.preprocessing class to be
            retrieved from the rex/sup3r library. If a sup3r.preprocessing
            class is used, all data will be loaded in this class'
            initialization and the subsequent bias calculation will be done in
            serial
        bias_handler : str
            Name of the bias data handler class to be retrieved from the
            sup3r.preprocessing library.
        base_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the base_handler
            class
        bias_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the bias_handler
            class
        decimals : int | None
            Option to round bias and base data to this number of decimals, this
            gets passed to np.around(). If decimals is negative, it specifies
            the number of positions to the left of the decimal point.
        match_zero_rate : bool
            Option to fix the frequency of zero values in the biased data. The
            lowest percentile of values in the biased data will be set to zero
            to match the percentile of zeros in the base data. If
            SkillAssessment is being run and this is True, the distributions
            will not be mean-centered. This helps resolve the issue where
            global climate models produce too many days with small
            precipitation totals e.g., the "drizzle problem" [Polade2014]_.
        pre_load : bool
            Flag to preload all data needed for bias correction. This is
            currently recommended to improve performance with the new sup3r
            data handler access patterns

        References
        ----------
        .. [Polade2014] Polade, S. D., Pierce, D. W., Cayan, D. R., Gershunov,
           A., & Dettineer, M. D. (2014). The key role of dry days in changing
           regional climate and precipitation regimes. Scientific reports,
           4(1), 4364. https://doi.org/10.1038/srep04364
        """

        logger.info(
            'Initializing DataRetrievalBase for base dset "%s" correcting '
            'biased dataset(s): %s',
            base_dset,
            bias_feature,
        )
        self.base_fps = base_fps
        self.bias_fps = bias_fps
        self.base_dset = base_dset
        self.bias_feature = bias_feature
        self.target = target
        self.shape = shape
        self.decimals = decimals
        self.base_handler_kwargs = base_handler_kwargs or {}
        self.bias_handler_kwargs = bias_handler_kwargs or {}
        self.bad_bias_gids = []
        self.match_zero_rate = match_zero_rate
        self.base_fps = expand_paths(self.base_fps)
        self.bias_fps = expand_paths(self.bias_fps)
        self._distance_upper_bound = distance_upper_bound

        base_sup3r_handler = getattr(sup3r.preprocessing, base_handler, None)
        base_rex_handler = getattr(rex, base_handler, None)

        if base_rex_handler is not None:
            self.base_handler = base_rex_handler
            self.base_dh = self.base_handler(
                self.base_fps[0], **self.base_handler_kwargs
            )
        elif base_sup3r_handler is not None:
            self.base_handler = base_sup3r_handler
            self.base_handler_kwargs['features'] = [self.base_dset]
            self.base_dh = self.base_handler(
                self.base_fps, **self.base_handler_kwargs
            )
            msg = (
                'Base data handler opened with a sup3r DataHandler class '
                'must load cached data!'
            )
            assert self.base_dh.data is not None, msg
        else:
            msg = f'Could not retrieve "{base_handler}" from sup3r or rex!'
            logger.error(msg)
            raise RuntimeError(msg)

        self.bias_handler = getattr(sup3r.preprocessing, bias_handler)
        self.base_meta = self.base_dh.meta
        self.bias_dh = self.bias_handler(
            self.bias_fps,
            [self.bias_feature],
            target=self.target,
            shape=self.shape,
            **self.bias_handler_kwargs,
        )
        lats = self.bias_dh.lat_lon[..., 0].flatten()
        self.bias_meta = self.bias_dh.meta
        self.bias_ti = self.bias_dh.time_index

        raster_shape = self.bias_dh.lat_lon[..., 0].shape
        bias_lat_lon = self.bias_meta[['latitude', 'longitude']].values
        self.bias_tree = KDTree(bias_lat_lon)
        self.bias_gid_raster = np.arange(lats.size)
        self.bias_gid_raster = self.bias_gid_raster.reshape(raster_shape)

        self.nn_dist, self.nn_ind = self.bias_tree.query(
            self.base_meta[['latitude', 'longitude']],
            distance_upper_bound=self.distance_upper_bound,
        )

        if pre_load:
            self.pre_load()

        self.out = None
        self._init_out()

        logger.info('Finished initializing DataRetrievalBase.')

    def pre_load(self):
        """Preload all data needed for bias correction. This is currently
        recommended to improve performance with the new sup3r data handler
        access patterns"""

        if hasattr(self.base_dh, 'compute'):
            logger.info('Pre loading baseline unbiased data into memory...')
            self.base_dh.data.compute()
            logger.info('Finished pre loading baseline unbiased data.')

        if hasattr(self.bias_dh, 'compute'):
            logger.info('Pre loading historical biased data into memory...')
            self.bias_dh.data.compute()
            logger.info('Finished pre loading historical biased data.')

    @abstractmethod
    def _init_out(self):
        """Initialize output arrays"""

    @property
    def meta(self):
        """Get a meta data dictionary on how these bias factors were
        calculated"""
        meta = {
            'base_fps': self.base_fps,
            'bias_fps': self.bias_fps,
            'base_dset': self.base_dset,
            'bias_feature': self.bias_feature,
            'target': self.target,
            'shape': self.shape,
            'class': str(self.__class__),
            'version_record': VERSION_RECORD,
        }
        return meta

    @property
    def distance_upper_bound(self):
        """Maximum distance (float) to map high-resolution data from exo_source
        to the low-resolution file_paths input."""
        if self._distance_upper_bound is None:
            diff = np.diff(
                self.bias_meta[['latitude', 'longitude']].values, axis=0
            )
            diff = np.max(np.median(diff, axis=0))
            self._distance_upper_bound = diff
            logger.info(
                'Set distance upper bound to %.4f',
                self._distance_upper_bound,
            )
        return self._distance_upper_bound

    @staticmethod
    def compare_dists(base_data, bias_data, adder=0, scalar=1):
        """Compare two distributions using the two-sample Kolmogorov-Smirnov.
        When the output is minimized, the two distributions are similar.

        Parameters
        ----------
        base_data : np.ndarray
            1D array of base data observations.
        bias_data : np.ndarray
            1D array of biased data observations.
        adder : float
            Factor to adjust the biased data before comparing distributions:
            bias_data * scalar + adder
        scalar : float
            Factor to adjust the biased data before comparing distributions:
            bias_data * scalar + adder

        Returns
        -------
        out : float
            KS test statistic
        """
        out = stats.ks_2samp(base_data, bias_data * scalar + adder)
        return out.statistic

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to call cls.run() on a single node based on an input
        config.

        Parameters
        ----------
        config : dict
            sup3r bias calc config with all necessary args and kwargs to
            initialize the class and call run() on a single node.
        """
        import_str = 'import time;\n'
        import_str += 'from gaps import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += f'from sup3r.bias import {cls.__name__}'

        if not hasattr(cls, 'run'):
            msg = (
                'I can only get you a node command for subclasses of '
                'DataRetrievalBase with a run() method.'
            )
            logger.error(msg)
            raise NotImplementedError(msg)

        # pylint: disable=E1101
        init_str = get_fun_call_str(cls, config)
        fun_str = get_fun_call_str(cls.run, config)
        fun_str = fun_str.partition('.')[-1]
        fun_str = 'bc.' + fun_str

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (
            f"python -c '{import_str};\n"
            't0 = time.time();\n'
            f'logger = init_logger({log_arg_str});\n'
            f'bc = {init_str};\n'
            f'{fun_str};\n'
            't_elap = time.time() - t0;\n'
        )

        pipeline_step = config.get('pipeline_step') or ModuleName.BIAS_CALC
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";'\n"

        return cmd.replace('\\', '/')

    def get_bias_gid(self, coord):
        """Get the bias gid from a coordinate.

        Parameters
        ----------
        coord : tuple
            (lat, lon) to get data for.

        Returns
        -------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.
        d : float
            Distance in decimal degrees from coord to bias gid
        """
        d, i = self.bias_tree.query(coord)
        bias_gid = self.bias_gid_raster.flatten()[i]
        return bias_gid, d

    def get_base_gid(self, bias_gid):
        """Get one or more base gid(s) corresponding to a bias gid.

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.

        Returns
        -------
        dist : np.ndarray
            Array of nearest neighbor distances with length equal to the number
            of high-resolution baseline gids that map to the low resolution
            bias gid pixel.
        base_gid : np.ndarray
            Array of base gids that are the nearest neighbors of bias_gid with
            length equal to the number of high-resolution baseline gids that
            map to the low resolution bias gid pixel.
        """
        base_gid = np.where(self.nn_ind == bias_gid)[0]
        dist = self.nn_dist[base_gid]
        return dist, base_gid

    def get_data_pair(self, coord, daily_reduction='avg'):
        """Get base and bias data observations based on a single bias gid.

        Parameters
        ----------
        coord : tuple
            (lat, lon) to get data for.
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged or min/max'd as well.
        bias_data : np.ndarray
            1D array of temporal data at the requested gid.
        base_dist : np.ndarray
            Array of nearest neighbor distances from coord to the base data
            sites with length equal to the number of high-resolution baseline
            gids that map to the low resolution bias gid pixel.
        bias_dist : Float
            Nearest neighbor distance from coord to the bias data site
        """
        bias_gid, bias_dist = self.get_bias_gid(coord)
        base_dist, base_gid = self.get_base_gid(bias_gid)
        bias_data = self.get_bias_data(bias_gid)
        base_data = self.get_base_data(
            self.base_fps,
            self.base_dset,
            base_gid,
            self.base_handler,
            daily_reduction=daily_reduction,
            decimals=self.decimals,
        )
        base_data = base_data[0]
        return base_data, bias_data, base_dist, bias_dist

    def get_bias_data(self, bias_gid, bias_dh=None):
        """Get data from the biased data source for a single gid

        Parameters
        ----------
        bias_gid : int
            gid of the data to retrieve in the bias data source raster data.
            The gids for this data source are the enumerated indices of the
            flattened coordinate array.
        bias_dh : DataHandler, default=self.bias_dh
            Any ``DataHandler`` from :mod:`sup3r.preprocessing`. This optional
            argument allows an alternative handler other than the usual
            :attr:`bias_dh`. For instance, the derived
            :class:`~qdm.QuantileDeltaMappingCorrection` uses it to access the
            reference biased dataset as well as the target biased dataset.

        Returns
        -------
        bias_data : np.ndarray
            1D array of temporal data at the requested gid.
        """

        row, col = np.where(self.bias_gid_raster == bias_gid)

        # This can be confusing. If the given argument `bias_dh` is None,
        # the default value for dh is `self.bias_dh`.
        dh = bias_dh or self.bias_dh
        bias_data = dh.data[self.bias_feature][row[0], col[0], ...]

        if bias_data.ndim != 1:
            msg = (
                'Found a weird number of feature channels for the bias '
                'data retrieval: {}. Need just one channel'.format(
                    bias_data.shape
                )
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if self.decimals is not None:
            bias_data = np.around(bias_data, decimals=self.decimals)

        return np.asarray(bias_data)

    @classmethod
    def get_base_data(
        cls,
        base_fps,
        base_dset,
        base_gid,
        base_handler,
        base_handler_kwargs=None,
        daily_reduction='avg',
        decimals=None,
        base_dh_inst=None,
    ):
        """Get data from the baseline data source, possibly for many high-res
        base gids corresponding to a single coarse low-res bias gid.

        Parameters
        ----------
        base_fps : list | str
            One or more baseline .h5 filepaths representing non-biased data to
            use to correct the biased dataset. This is typically several years
            of WTK or NSRDB files.
        base_dset : str
            A single dataset from the base_fps to retrieve.
        base_gid : int | np.ndarray
            One or more spatial gids to retrieve from base_fps. The data will
            be spatially averaged across all of these sites.
        base_handler : rex.Resource
            A rex data handler similar to rex.Resource or sup3r.DataHandler
            classes (if using the latter, must also input base_dh_inst)
        base_handler_kwargs : dict | None
            Optional kwargs to send to the initialization of the base_handler
            class
        daily_reduction : None | str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)
        decimals : int | None
            Option to round bias and base data to this number of
            decimals, this gets passed to np.around(). If decimals
            is negative, it specifies the number of positions to
            the left of the decimal point.
        base_dh_inst : sup3r.DataHandler
            Instantiated  DataHandler class that has already loaded the base
            data (required if base files are .nc and are not being opened by a
            rex Resource handler).

        Returns
        -------
        out_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged or min/max'd as well.
        out_ti : pd.DatetimeIndex
            DatetimeIndex object of datetimes corresponding to the
            output data.
        """

        out_data = []
        out_ti = []
        all_cs_ghi = []
        base_handler_kwargs = base_handler_kwargs or {}

        if issubclass(base_handler, DataHandler) and base_dh_inst is None:
            msg = (
                'The method `get_base_data()` is only to be used with '
                '`base_handler` as a `sup3r.DataHandler` subclass if '
                '`base_dh_inst` is also provided!'
            )
            logger.error(msg)
            raise RuntimeError(msg)

        if issubclass(base_handler, DataHandler) and base_dh_inst is not None:
            out_ti = base_dh_inst.time_index
            out_data = cls._read_base_sup3r_data(
                base_dh_inst, base_dset, base_gid
            )
            all_cs_ghi = np.ones(len(out_data), dtype=np.float32) * np.nan
        else:
            for fp in base_fps:
                with base_handler(fp, **base_handler_kwargs) as res:
                    base_ti = res.time_index
                    temp_out = cls._read_base_rex_data(
                        res, base_dset, base_gid
                    )
                    base_data, base_cs_ghi = temp_out

                out_data.append(base_data)
                out_ti.append(base_ti)
                all_cs_ghi.append(base_cs_ghi)

            out_data = np.hstack(out_data)
            out_ti = pd.DatetimeIndex(np.hstack(out_ti))
            all_cs_ghi = np.hstack(all_cs_ghi)

        if daily_reduction is not None:
            out_data, out_ti = cls._reduce_base_data(
                out_ti, out_data, all_cs_ghi, base_dset, daily_reduction
            )

        if decimals is not None:
            out_data = np.around(out_data, decimals=decimals)

        return np.asarray(out_data), out_ti

    @staticmethod
    def _match_zero_rate(bias_data, base_data):
        """The lowest percentile of values in the biased data will be set to
        zero to match the percentile of zeros in the base data. This helps
        resolve the issue where global climate models produce too many days
        with small precipitation totals e.g., the "drizzle problem".
        Ref: Polade et al., 2014 https://doi.org/10.1038/srep04364

        Parameters
        ----------
        bias_data : Union[np.ndarray, da.core.Array]
            1D array of biased data observations.
        base_data : Union[np.ndarray, da.core.Array]
            1D array of base data observations.

        Returns
        -------
        bias_data : np.ndarray
            1D array of biased data observations. Values below the quantile
            associated with zeros in base_data will be set to zero
        """

        q_zero_base_in = np.nanmean(base_data == 0)
        q_zero_bias_in = np.nanmean(bias_data == 0)

        q_bias = np.linspace(0, 1, len(bias_data))
        min_value_bias = np.interp(q_zero_base_in, q_bias, sorted(bias_data))

        bias_data[bias_data < min_value_bias] = 0

        q_zero_base_out = np.nanmean(base_data == 0)
        q_zero_bias_out = np.nanmean(bias_data == 0)

        logger.debug(
            'Input bias/base zero rate is %.3e/%.3e, output is %.3e/%.3e',
            q_zero_bias_in,
            q_zero_base_in,
            q_zero_bias_out,
            q_zero_base_out,
        )

        return bias_data

    @staticmethod
    def _read_base_sup3r_data(dh, base_dset, base_gid):
        """Read baseline data from a sup3r DataHandler

        Parameters
        ----------
        dh : sup3r.DataHandler
            sup3r DataHandler that is an open file handler of the base file(s)
        base_dset : str
            A single dataset from the base_fps to retrieve.
        base_gid : int | np.ndarray
            One or more spatial gids to retrieve from base_fps. The data will
            be spatially averaged across all of these sites.

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
        """
        gid_raster = np.arange(len(dh.meta))
        gid_raster = gid_raster.reshape(dh.shape[:2])
        idy, idx = np.where(np.isin(gid_raster, base_gid))
        if dh.data.loaded:
            # faster direct access of numpy array if loaded
            base_data = dh.data[base_dset].data[idy, idx]
        else:
            base_data = dh.data[base_dset].data.vindex[idy, idx]

        assert base_data.shape[0] == len(base_gid)
        assert base_data.shape[1] == len(dh.time_index)
        return base_data.mean(axis=0)

    @staticmethod
    def _read_base_rex_data(res, base_dset, base_gid):
        """Read baseline data from a rex resource handler with extra logic for
        special datasets (e.g. u/v wind components or clearsky_ratio)

        Parameters
        ----------
        res : rex.Resource
            rex Resource handler that is an open file handler of the base
            file(s)
        base_dset : str
            A single dataset from the base_fps to retrieve.
        base_gid : int | np.ndarray
            One or more spatial gids to retrieve from base_fps. The data will
            be spatially averaged across all of these sites.

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
        base_cs_ghi : np.ndarray
            If base_dset == "clearsky_ratio", the base_data array is GHI and
            this base_cs_ghi is clearsky GHI. Otherwise this is an array with
            same length as base_data but full of np.nan
        """

        msg = '`res` input must not be a `DataHandler` subclass!'
        assert not issubclass(res.__class__, DataHandler), msg

        base_cs_ghi = None

        if base_dset.lower().startswith(('u_', 'v_')):
            dset_ws = base_dset.lower().replace('u_', 'windspeed_')
            dset_ws = dset_ws.lower().replace('v_', 'windspeed_')
            dset_wd = dset_ws.replace('speed', 'direction')
            base_ws = res[dset_ws, :, base_gid]
            base_wd = res[dset_wd, :, base_gid]

            if base_dset.startswith('u_'):
                base_data = -base_ws * np.sin(np.radians(base_wd))
            else:
                base_data = -base_ws * np.cos(np.radians(base_wd))

        elif base_dset == 'clearsky_ratio':
            base_data = res['ghi', :, base_gid]
            base_cs_ghi = res['clearsky_ghi', :, base_gid]

        else:
            base_data = res[base_dset, :, base_gid]

        if len(base_data.shape) == 2:
            base_data = np.nanmean(base_data, axis=1)
            if base_cs_ghi is not None:
                base_cs_ghi = np.nanmean(base_cs_ghi, axis=1)

        if base_cs_ghi is None:
            base_cs_ghi = np.ones(len(base_data), dtype=np.float32) * np.nan

        return base_data, base_cs_ghi

    @staticmethod
    def _reduce_base_data(
        base_ti, base_data, base_cs_ghi, base_dset, daily_reduction
    ):
        """Reduce the base timeseries data using some sort of daily reduction
        function.

        Parameters
        ----------
        base_ti : pd.DatetimeIndex
            Time index associated with base_data
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
        base_cs_ghi : np.ndarray
            If base_dset == "clearsky_ratio", the base_data array is GHI and
            this base_cs_ghi is clearsky GHI. Otherwise this is an array with
            same length as base_data but full of np.nan
        base_dset : str
            A single dataset from the base_fps to retrieve.
        daily_reduction : str
            Option to do a reduction of the hourly+ source base data to daily
            data. Can be None (no reduction, keep source time frequency), "avg"
            (daily average), "max" (daily max), "min" (daily min),
            "sum" (daily sum/total)

        Returns
        -------
        base_data : np.ndarray
            1D array of base data spatially averaged across the base_gid input
            and possibly daily-averaged or min/max'd as well.
        daily_ti : pd.DatetimeIndex
            Daily DatetimeIndex corresponding to the daily base_data
        """

        if daily_reduction is None:
            return base_data

        daily_ti = pd.DatetimeIndex(sorted(set(base_ti.date)))
        df = pd.DataFrame(
            {
                'date': base_ti.date,
                'base_data': base_data,
                'base_cs_ghi': base_cs_ghi,
            }
        )

        cs_ratio = (
            daily_reduction.lower() in {'avg', 'average', 'mean'}
            and base_dset == 'clearsky_ratio'
        )

        if cs_ratio:
            daily_ghi = df.groupby('date').sum()['base_data'].values
            daily_cs_ghi = df.groupby('date').sum()['base_cs_ghi'].values
            daily_ghi[daily_cs_ghi == 0] = 0
            daily_cs_ghi[daily_cs_ghi == 0] = 1
            base_data = daily_ghi / daily_cs_ghi
            mask = np.isnan(base_data)
            msg = (
                'Could not calculate daily average "clearsky_ratio" with '
                'input ghi and cs ghi inputs: \n{}, \n{}'.format(
                    daily_ghi[mask], daily_cs_ghi[mask]
                )
            )
            assert not np.isnan(base_data).any(), msg

        elif daily_reduction.lower() in {'avg', 'average', 'mean'}:
            base_data = df.groupby('date').mean()['base_data'].values

        elif daily_reduction.lower() in {'max', 'maximum'}:
            base_data = df.groupby('date').max()['base_data'].values

        elif daily_reduction.lower() in {'min', 'minimum'}:
            base_data = df.groupby('date').min()['base_data'].values

        elif daily_reduction.lower() in {'sum', 'total'}:
            base_data = df.groupby('date').sum()['base_data'].values

        msg = (
            f'Daily reduced base data shape {base_data.shape} does not '
            f'match daily time index shape {daily_ti.shape}, '
            'something went wrong!'
        )
        assert len(base_data.shape) == 1, msg
        assert base_data.shape == daily_ti.shape, msg

        return base_data, daily_ti
