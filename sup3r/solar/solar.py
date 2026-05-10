"""Custom sup3r solar module. This primarily converts GAN output clearsky ratio
to GHI, DNI, and DHI using NSRDB data and utility modules like DISC

Note that clearsky_ratio is assumed to be clearsky ghi ratio and is calculated
as daily average GHI / daily average clearsky GHI.
"""

import logging
import os

import numpy as np
import pandas as pd
from farms.disc import disc
from farms.utilities import calc_dhi, dark_night
from rex import MultiTimeResource, Resource
from rex.utilities.fun_utils import get_fun_call_str
from scipy.spatial import KDTree

from sup3r.preprocessing.utilities import expand_paths
from sup3r.utilities import ModuleName
from sup3r.utilities.cli import BaseCLI
from sup3r.utilities.utilities import OUTPUT_ATTRS
from sup3r.writers import RexOutputs

logger = logging.getLogger(__name__)


class Solar:
    """Custom sup3r solar module. This primarily converts GAN output clearsky
    ratio to GHI, DNI, and DHI using NSRDB data and utility modules like
    DISC"""

    def __init__(
        self,
        sup3r_fps,
        nsrdb_fp,
        t_slice=slice(None),
        tz=-7,
        agg_factor=1,
        nn_threshold=0.5,
        cloud_threshold=0.99,
    ):
        """
        Parameters
        ----------
        sup3r_fps : str | list
            Full .h5 filepath(s) to one or more sup3r GAN output .h5 chunk
            files containing clearsky_ratio, time_index, and meta. These files
            must have the same meta data but can be sequential and ordered
            temporal chunks. The data in this file has been rolled by tz
            to a local time (assumed that the GAN was trained on local time
            solar data) and will be converted back into UTC, so it's wise to
            include some padding in the sup3r_fps file list.
        nsrdb_fp : str
            Filepath to NSRDB .h5 file containing clearsky_ghi, clearsky_dni,
            clearsky_dhi data.
        t_slice : slice
            Slicing argument to slice the temporal axis of the sup3r_fps source
            data after doing the tz roll to UTC but before returning the
            irradiance variables. This can be used to effectively pad the solar
            irradiance calculation in UTC time. For example, if sup3r_fps is 3
            files each with 24 hours of data, t_slice can be slice(24, 48) to
            only output the middle day of irradiance data, but padded by the
            other two days for the UTC output.
        tz : int
            The timezone offset for the data in sup3r_fps. It is assumed that
            the GAN is trained on data in local time and therefore the output
            in sup3r_fps should be treated as local time. For example, -7 is
            MST which is default for CONUS training data.
        agg_factor : int
            Spatial aggregation factor for nsrdb-to-GAN-meta e.g. the number of
            NSRDB spatial pixels to average for a single sup3r GAN output site.
        nn_threshold : float
            The KDTree nearest neighbor threshold that determines how far the
            sup3r GAN output data has to be from the NSRDB source data to get
            irradiance=0. Note that is value is in decimal degrees which is a
            very approximate way to determine real distance.
        cloud_threshold : float
            Clearsky ratio threshold below which the data is considered cloudy
            and DNI is calculated using DISC.
        """

        self.t_slice = t_slice
        self.agg_factor = agg_factor
        self.nn_threshold = nn_threshold
        self.cloud_threshold = cloud_threshold
        self.tz = tz
        self._nsrdb_fp = nsrdb_fp
        self._sup3r_fps = sup3r_fps
        if isinstance(self._sup3r_fps, str):
            self._sup3r_fps = [self._sup3r_fps]

        logger.debug(
            'Initializing solar module with sup3r files: {}'.format(
                [os.path.basename(fp) for fp in self._sup3r_fps]
            )
        )
        logger.debug(
            'Initializing solar module with temporal slice: {}'.format(
                self.t_slice
            )
        )
        logger.debug(
            'Initializing solar module with NSRDB source fp: {}'.format(
                self._nsrdb_fp
            )
        )

        self.gan_data = MultiTimeResource(self._sup3r_fps)
        self.nsrdb = Resource(self._nsrdb_fp)

        # cached variables
        self._nsrdb_tslice = None
        self._idnn = None
        self._dist = None
        self._sza = None
        self._cs_ratio = None
        self._ghi = None
        self._dni = None
        self._dhi = None

        self.preflight()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if exc_type is not None:
            raise

    def preflight(self):
        """Run preflight checks on source data to make sure everything will
        work together."""
        assert 'clearsky_ratio' in self.gan_data.dsets
        assert 'clearsky_ghi' in self.nsrdb.dsets
        assert 'clearsky_dni' in self.nsrdb.dsets
        assert 'solar_zenith_angle' in self.nsrdb.dsets
        assert 'surface_pressure' in self.nsrdb.dsets
        assert isinstance(self.nsrdb_tslice, slice)

        delta = pd.Series(self.time_index[1:] - self.time_index[:-1])
        delta = delta.mean().total_seconds()
        msg = (
            'Its assumed that the sup3r GAN output solar data will be '
            'hourly but received time index: {}'.format(self.time_index)
        )
        assert delta == 3600, msg

    def close(self):
        """Close all internal file handlers"""
        self.gan_data.close()
        self.nsrdb.close()

    @property
    def idnn(self):
        """Get the nearest neighbor meta data indices from the NSRDB data that
        correspond to the sup3r GAN data

        Returns
        -------
        idnn : Union[np.ndarray, da.core.Array]
            2D array of length (n_sup3r_sites, agg_factor) where the values are
            meta data indices from the NSRDB.
        """
        if self._idnn is None:
            tree = KDTree(self.nsrdb.meta[['latitude', 'longitude']])
            out = tree.query(self.gan_data.meta[['latitude', 'longitude']])
            self._dist, self._idnn = out
            if len(self._idnn.shape) == 1:
                self._dist = np.expand_dims(self._dist, axis=1)
                self._idnn = np.expand_dims(self._idnn, axis=1)
        return self._idnn

    @property
    def dist(self):
        """Get the nearest neighbor distances from the sup3r GAN data sites to
        the NSRDB nearest neighbors.

        Returns
        -------
        dist : Union[np.ndarray, da.core.Array]
            2D array of length (n_sup3r_sites, agg_factor) where the values are
            decimal degree distances from the sup3r sites to the nsrdb nearest
            neighbors.
        """
        if self._dist is None:
            _ = self.idnn
        return self._dist

    @property
    def time_index(self):
        """Time index for the sup3r GAN output data but sliced by t_slice

        Returns
        -------
        pd.DatetimeIndex
        """
        return self.gan_data.time_index[self.t_slice]

    @property
    def out_of_bounds(self):
        """Get a boolean mask for the sup3r data that is out of bounds (too far
        from the NSRDB data).

        Returns
        -------
        out_of_bounds : Union[np.ndarray, da.core.Array]
            1D boolean array with length == number of sup3r GAN sites. True if
            the site is too far from the NSRDB.
        """
        return (self.dist > self.nn_threshold).any(axis=1)

    @property
    def nsrdb_tslice(self):
        """Get the time slice of the NSRDB data corresponding to the sup3r GAN
        output."""

        if self._nsrdb_tslice is None:
            doy_nsrdb = self.nsrdb.time_index.day_of_year
            doy_gan = self.time_index.day_of_year
            mask = doy_nsrdb.isin(doy_gan)

            if mask.sum() == 0:
                msg = (
                    'Time index intersection of the NSRDB time index and '
                    'sup3r GAN output has only {} common timesteps! '
                    'Something went wrong.\nNSRDB time index: \n{}\nSup3r '
                    'GAN output time index:\n{}'.format(
                        mask.sum(), self.nsrdb.time_index, self.time_index
                    )
                )
                logger.error(msg)
                raise RuntimeError(msg)

            ilocs = np.where(mask)[0]
            t0, t1 = ilocs[0], ilocs[-1] + 1

            ti_nsrdb = self.nsrdb.time_index
            delta = (
                pd.Series(ti_nsrdb[1:] - ti_nsrdb[:-1])[1:]
                .mean()
                .total_seconds()
            )

            step = int(3600 / delta)
            self._nsrdb_tslice = slice(t0, t1, step)

            logger.debug(
                'Found nsrdb_tslice {} with corresponding '
                'time index:\n\t{}'.format(
                    self._nsrdb_tslice,
                    self.nsrdb.time_index[self._nsrdb_tslice],
                )
            )

        return self._nsrdb_tslice

    @property
    def clearsky_ratio(self):
        """Get the clearsky ghi ratio data from the GAN output, rolled from tz
        to UTC.

        Returns
        -------
        clearsky_ratio : Union[np.ndarray, da.core.Array]
            2D array with shape (time, sites) in UTC.
        """
        if self._cs_ratio is None:
            self._cs_ratio = self.gan_data['clearsky_ratio']
            self._cs_ratio = np.roll(self._cs_ratio, -self.tz, axis=0)

            # if tz is negative, roll to utc is positive, and the beginning of
            # the dataset is rolled over from the end and must be backfilled,
            # otherwise you can get seams
            self._cs_ratio[: -self.tz, :] = self._cs_ratio[-self.tz, :]

            # apply temporal slicing of source data, see docstring on t_slice
            # for more info
            self._cs_ratio = self._cs_ratio[self.t_slice, :]

        return self._cs_ratio

    @property
    def solar_zenith_angle(self):
        """Get the solar zenith angle (degrees)

        Returns
        -------
        solar_zenith_angle : Union[np.ndarray, da.core.Array]
            2D array with shape (time, sites) in UTC.
        """
        if self._sza is None:
            self._sza = self.get_nsrdb_data('solar_zenith_angle')
        return self._sza

    @property
    def ghi(self):
        """Get the ghi (W/m2) based on the GAN output clearsky ratio +
        clearsky_ghi in UTC.

        Returns
        -------
        ghi : Union[np.ndarray, da.core.Array]
            2D array with shape (time, sites) in UTC.
        """
        if self._ghi is None:
            logger.debug('Calculating GHI.')
            nsrdb_cs_ghi = self.get_nsrdb_data('clearsky_ghi')
            self._ghi = nsrdb_cs_ghi * self.clearsky_ratio
            self._ghi[:, self.out_of_bounds] = 0
        return self._ghi

    @property
    def dni(self):
        """Get the dni (W/m2) which is clearsky dni (from NSRDB) when the GAN
        output is clear (clearsky_ratio is > cloud_threshold), and calculated
        from the DISC model when cloudy

        Returns
        -------
        dni : Union[np.ndarray, da.core.Array]
            2D array with shape (time, sites) in UTC.
        """
        if self._dni is None:
            logger.debug('Calculating DNI.')
            self._dni = self.get_nsrdb_data('clearsky_dni')
            pressure = self.get_nsrdb_data('surface_pressure')
            doy = self.time_index.day_of_year.values
            cloudy_dni = disc(
                self.ghi, self.solar_zenith_angle, doy, pressure=pressure
            )
            cloudy_dni = np.minimum(self._dni, cloudy_dni)
            self._dni[self.cloud_mask] = cloudy_dni[self.cloud_mask]
            self._dni = dark_night(self._dni, self.solar_zenith_angle)
            self._dni[:, self.out_of_bounds] = 0
        return self._dni

    @property
    def dhi(self):
        """Get the dhi (W/m2) which is calculated based on the simple
        relationship between GHI, DNI and solar zenith angle.

        Returns
        -------
        dhi : Union[np.ndarray, da.core.Array]
            2D array with shape (time, sites) in UTC.
        """
        if self._dhi is None:
            logger.debug('Calculating DHI.')
            self._dhi, self._dni = calc_dhi(
                self.dni, self.ghi, self.solar_zenith_angle
            )
            self._dhi = dark_night(self._dhi, self.solar_zenith_angle)
            self._dhi[:, self.out_of_bounds] = 0
        return self._dhi

    @property
    def cloud_mask(self):
        """Get the cloud mask (True if cloudy) based on the GAN output clearsky
        ratio in UTC.

        Returns
        -------
        cloud_mask : Union[np.ndarray, da.core.Array]
            2D array with shape (time, sites) in UTC.
        """
        return self.clearsky_ratio < self.cloud_threshold

    def get_nsrdb_data(self, dset):
        """Get an NSRDB dataset with spatial index corresponding to the sup3r
        GAN output data, averaged across the agg_factor.

        Parameters
        ----------
        dset : str
            Name of dataset to retrieve from NSRDB source file

        Returns
        -------
        out : Union[np.ndarray, da.core.Array]
            Dataset of shape (time, sites) where time and sites correspond to
            the same shape as the sup3r GAN output data and if agg_factor > 1
            the sites is an average across multiple NSRDB sites.
        """

        logger.debug('Retrieving "{}" from NSRDB source data.'.format(dset))
        out = None

        for idx in range(self.idnn.shape[1]):
            temp = self.nsrdb[dset, self.nsrdb_tslice, self.idnn[:, idx]]
            temp = temp.astype(np.float32)

            if out is None:
                out = temp
            else:
                out += temp

        out /= self.idnn.shape[1]

        return out

    @staticmethod
    def get_sup3r_fps(fp_pattern, ignore=None):
        """Get a list of file chunks to run in parallel based on a file pattern

        Note
        ----
        It's assumed that all source files have the pattern
        `sup3r_file_TTTTTT_SSSSSS.h5` where TTTTTT is the zero-padded temporal
        chunk index and SSSSSS is the zero-padded spatial chunk index.

        Parameters
        ----------
        fp_pattern : str | list
            Unix-style file*pattern that matches a set of spatiotemporally
            chunked sup3r forward pass output files.
        ignore : str | None
            Ignore all files that have this string in their filenames.

        Returns
        -------
        fp_sets : list
            List of file sets where each file set is 3 temporally sequential
            files over the same spatial chunk. Each file set overlaps with its
            neighbor such that fp_sets[0][-1] == fp_sets[1][0] (this is so
            Solar can do temporal padding when rolling from GAN local time
            output to UTC).
        t_slices : list
            List of t_slice arguments corresponding to fp_sets to pass to
            Solar class initialization that will slice and reduce the
            overlapping time axis when Solar outputs irradiance data.
        temporal_ids : list
            List of temporal id strings TTTTTT corresponding to the fp_sets
        spatial_ids : list
            List of spatial id strings SSSSSS corresponding to the fp_sets
        target_fps : list
            List of actual target files corresponding to fp_sets, so for
            example the file set fp_sets[10] sliced by t_slices[10] is designed
            to process target_fps[10]
        """

        all_fps = [fp for fp in expand_paths(fp_pattern) if fp.endswith('.h5')]
        if ignore is not None:
            all_fps = [
                fp for fp in all_fps if ignore not in os.path.basename(fp)
            ]

        all_fps = sorted(all_fps)

        source_dir = os.path.dirname(all_fps[0])
        source_fn_base = os.path.basename(all_fps[0]).replace('.h5', '')
        source_fn_base = '_'.join(source_fn_base.split('_')[:-2])

        all_id_spatial = [
            fp.replace('.h5', '').split('_')[-1] for fp in all_fps
        ]
        all_id_temporal = [
            fp.replace('.h5', '').split('_')[-2] for fp in all_fps
        ]

        all_id_spatial = sorted(set(all_id_spatial))
        all_id_temporal = sorted(set(all_id_temporal))

        fp_sets = []
        t_slices = []
        temporal_ids = []
        spatial_ids = []
        target_fps = []
        for idt, id_temporal in enumerate(all_id_temporal):
            start = 0
            single_chunk_id_temps = [id_temporal]

            if idt > 0:
                start = 24
                single_chunk_id_temps.insert(0, all_id_temporal[idt - 1])

            if idt < (len(all_id_temporal) - 1):
                single_chunk_id_temps.append(all_id_temporal[idt + 1])

            for id_spatial in all_id_spatial:
                single_fp_set = []
                for t_str in single_chunk_id_temps:
                    fp = os.path.join(source_dir, source_fn_base)
                    fp += f'_{t_str}_{id_spatial}.h5'
                    single_fp_set.append(fp)

                fp_target = os.path.join(source_dir, source_fn_base)
                fp_target += f'_{id_temporal}_{id_spatial}.h5'

                fp_sets.append(single_fp_set)
                t_slices.append(slice(start, start + 24))
                temporal_ids.append(id_temporal)
                spatial_ids.append(id_spatial)
                target_fps.append(fp_target)

        return fp_sets, t_slices, temporal_ids, spatial_ids, target_fps

    @classmethod
    def get_node_cmd(cls, config):
        """Get a CLI call to run Solar.run_temporal_chunk() on a single
        node based on an input config.

        Parameters
        ----------
        config : dict
            sup3r solar config with all necessary args and kwargs to
            run Solar.run_temporal_chunk() on a single node.
        """
        import_str = 'import time;\n'
        import_str += 'from gaps import Status;\n'
        import_str += 'from rex import init_logger;\n'
        import_str += f'from sup3r.solar import {cls.__name__}'

        fun_str = get_fun_call_str(cls.run_temporal_chunks, config)

        log_file = config.get('log_file', None)
        log_level = config.get('log_level', 'INFO')
        log_arg_str = f'"sup3r", log_level="{log_level}"'
        if log_file is not None:
            log_arg_str += f', log_file="{log_file}"'

        cmd = (
            f"python -c '{import_str};\n"
            't0 = time.time();\n'
            f'logger = init_logger({log_arg_str});\n'
            f'{fun_str};\n'
            't_elap = time.time() - t0;\n'
        )

        pipeline_step = config.get('pipeline_step') or ModuleName.SOLAR
        cmd = BaseCLI.add_status_cmd(config, pipeline_step, cmd)
        cmd += ";'\n"

        return cmd.replace('\\', '/')

    def write(self, fp_out, features=('ghi', 'dni', 'dhi')):
        """Write irradiance datasets (ghi, dni, dhi) to output h5 file.

        Parameters
        ----------
        fp_out : str
            Filepath to an output h5 file to write irradiance variables to.
            Parent directory will be created if it does not exist.
        features : list | tuple
            List of features to write to disk. These have to be attributes of
            the Solar class (ghi, dni, dhi).
        """

        if not os.path.exists(os.path.dirname(fp_out)):
            os.makedirs(os.path.dirname(fp_out), exist_ok=True)

        with RexOutputs(fp_out, 'w') as fh:
            fh.meta = self.gan_data.meta
            fh.time_index = self.time_index

            for feature in features:
                attrs = OUTPUT_ATTRS[feature]
                arr = getattr(self, feature, None)
                if arr is None:
                    msg = (
                        'Feature "{}" was not available from Solar '
                        'module class.'.format(feature)
                    )
                    logger.error(msg)
                    raise AttributeError(msg)

                fh.add_dataset(
                    fp_out,
                    feature,
                    arr,
                    dtype=attrs['dtype'],
                    attrs=attrs,
                    chunks=attrs['chunks'],
                )
                logger.info(f'Added "{feature}" to output file.')
            run_attrs = self.gan_data.h5[self._sup3r_fps[0]].global_attrs
            run_attrs['nsrdb_source'] = self._nsrdb_fp
            fh.run_attrs = run_attrs

        logger.debug(f'Finished writing file: {fp_out}')

    @classmethod
    def run_temporal_chunks(
        cls,
        fp_pattern,
        nsrdb_fp,
        fp_out_suffix='irradiance',
        tz=-7,
        agg_factor=1,
        nn_threshold=0.5,
        cloud_threshold=0.99,
        features=('ghi', 'dni', 'dhi'),
        temporal_ids=None,
    ):
        """Run the solar module on all spatial chunks for each temporal
        chunk corresponding to the fp_pattern and the given list of
        temporal_ids. This typically gets run from the CLI.

        Parameters
        ----------
        fp_pattern : str
            Unix-style file*pattern that matches a set of spatiotemporally
            chunked sup3r forward pass output files.
        nsrdb_fp : str
            Filepath to NSRDB .h5 file containing clearsky_ghi, clearsky_dni,
            clearsky_dhi data.
        fp_out_suffix : str
            Suffix to add to the input sup3r source files when writing the
            processed solar irradiance data to new data files.
        tz : int
            The timezone offset for the data in sup3r_fps. It is assumed that
            the GAN is trained on data in local time and therefore the output
            in sup3r_fps should be treated as local time. For example, -7 is
            MST which is default for CONUS training data.
        agg_factor : int
            Spatial aggregation factor for nsrdb-to-GAN-meta e.g. the number of
            NSRDB spatial pixels to average for a single sup3r GAN output site.
        nn_threshold : float
            The KDTree nearest neighbor threshold that determines how far the
            sup3r GAN output data has to be from the NSRDB source data to get
            irradiance=0. Note that is value is in decimal degrees which is a
            very approximate way to determine real distance.
        cloud_threshold : float
            Clearsky ratio threshold below which the data is considered cloudy
            and DNI is calculated using DISC.
        features : list | tuple
            List of features to write to disk. These have to be attributes of
            the Solar class (ghi, dni, dhi).
        temporal_ids : list | None
            Lise of zero-padded temporal ids from the file chunks that match
            fp_pattern. This input typically gets set from the CLI. If None,
            this will run all temporal indices.
        """
        if temporal_ids is None:
            cls._run_temporal_chunk(
                fp_pattern=fp_pattern,
                nsrdb_fp=nsrdb_fp,
                fp_out_suffix=fp_out_suffix,
                tz=tz,
                agg_factor=agg_factor,
                nn_threshold=nn_threshold,
                cloud_threshold=cloud_threshold,
                features=features,
                temporal_id=temporal_ids,
            )
        else:
            for temporal_id in temporal_ids:
                cls._run_temporal_chunk(
                    fp_pattern=fp_pattern,
                    nsrdb_fp=nsrdb_fp,
                    fp_out_suffix=fp_out_suffix,
                    tz=tz,
                    agg_factor=agg_factor,
                    nn_threshold=nn_threshold,
                    cloud_threshold=cloud_threshold,
                    features=features,
                    temporal_id=temporal_id,
                )

    @classmethod
    def _run_temporal_chunk(
        cls,
        fp_pattern,
        nsrdb_fp,
        fp_out_suffix='irradiance',
        tz=-7,
        agg_factor=1,
        nn_threshold=0.5,
        cloud_threshold=0.99,
        features=('ghi', 'dni', 'dhi'),
        temporal_id=None,
    ):
        """Run the solar module on all spatial chunks for a single temporal
        chunk corresponding to the fp_pattern. This typically gets run from the
        CLI.

        See Also
        --------
        :meth:`run_temporal_chunks`
        """

        temp = cls.get_sup3r_fps(fp_pattern, ignore=f'_{fp_out_suffix}.h5')
        fp_sets, t_slices, temporal_ids, _, target_fps = temp

        if temporal_id is not None:
            fp_sets = [
                fp_set
                for i, fp_set in enumerate(fp_sets)
                if temporal_ids[i] == temporal_id
            ]
            t_slices = [
                t_slice
                for i, t_slice in enumerate(t_slices)
                if temporal_ids[i] == temporal_id
            ]
            target_fps = [
                target_fp
                for i, target_fp in enumerate(target_fps)
                if temporal_ids[i] == temporal_id
            ]

        zip_iter = zip(fp_sets, t_slices, target_fps)
        for i, (fp_set, t_slice, fp_target) in enumerate(zip_iter):
            fp_out = fp_target.replace('.h5', f'_{fp_out_suffix}.h5')

            if os.path.exists(fp_out):
                logger.info('%s already exists. Skipping.', fp_out)

            else:
                logger.info(
                    'Running temporal index {} out of {}.'.format(
                        i + 1, len(fp_sets)
                    )
                )

                kwargs = {
                    't_slice': t_slice,
                    'tz': tz,
                    'agg_factor': agg_factor,
                    'nn_threshold': nn_threshold,
                    'cloud_threshold': cloud_threshold,
                }
                tmp_out = fp_out + '.tmp'
                with Solar(fp_set, nsrdb_fp, **kwargs) as solar:
                    solar.write(tmp_out, features=features)
                os.replace(tmp_out, fp_out)
