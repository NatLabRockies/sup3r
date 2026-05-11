"""Classes to compute means from vortex and era data and compute bias
correction factors.

Vortex mean files can be downloaded from IRENA.
https://globalatlas.irena.org/workspace
"""

import calendar
import logging
import os

import dask
import numpy as np
import pandas as pd
import xarray as xr
from rex import Resource
from scipy.interpolate import interp1d

from sup3r.preprocessing.utilities import log_args
from sup3r.utilities import VERSION_RECORD
from sup3r.utilities.utilities import get_tmp_file
from sup3r.writers import OutputHandler, RexOutputs

logger = logging.getLogger(__name__)


class VortexMeanPrepper:
    """Class for converting monthly vortex tif files for each height to a
    single h5 files containing all monthly means for all requested output
    heights.
    """

    @log_args
    def __init__(self, path_pattern, in_heights, out_heights, overwrite=False):
        """
        Parameters
        ----------
        path_pattern : str
            Pattern for input tif files. Needs to include {month} and {height}
            format keys.
        in_heights : list
            List of heights for input files.
        out_heights : list
            List of output heights used for interpolation
        overwrite : bool
            Whether to overwrite intermediate netcdf files containing the
            interpolated masked monthly means.
        """
        msg = 'path_pattern needs to have {month} and {height} format keys'
        assert '{month}' in path_pattern and '{height}' in path_pattern, msg
        self.path_pattern = path_pattern
        self.in_heights = in_heights
        self.out_heights = out_heights
        self.out_dir = os.path.dirname(path_pattern)
        self.overwrite = overwrite
        self._mask = None
        self._meta = None

    @property
    def in_features(self):
        """List of features corresponding to input heights."""
        return [f'windspeed_{h}m' for h in self.in_heights]

    @property
    def out_features(self):
        """List of features corresponding to output heights"""
        return [f'windspeed_{h}m' for h in self.out_heights]

    def get_input_file(self, month, height):
        """Get vortex tif file for given month and height."""
        return self.path_pattern.format(month=month, height=height)

    def get_height_files(self, month):
        """Get set of netcdf files for given month"""
        files = []
        for height in self.in_heights:
            infile = self.get_input_file(month, height)
            out_file = infile.replace('.tif', '.nc')
            files.append(out_file)
        return files

    @property
    def input_files(self):
        """Get list of all input files used for h5 meta."""
        files = []
        for height in self.in_heights:
            for i in range(1, 13):
                month = calendar.month_name[i]
                files.append(self.get_input_file(month, height))
        return files

    def get_output_file(self, month):
        """Get name of netcdf file for a given month."""
        return os.path.join(
            self.out_dir.replace('{month}', month), f'{month}.nc'
        )

    @property
    def output_files(self):
        """List of output monthly output files each with windspeed for all
        input heights
        """
        files = []
        for i in range(1, 13):
            month = calendar.month_name[i]
            files.append(self.get_output_file(month))
        return files

    def convert_month_height_tif(self, month, height):
        """Get windspeed mean for the given month and hub height from the
        corresponding input file and write this to a netcdf file.
        """
        infile = self.get_input_file(month, height)
        logger.info('Getting mean windspeed_%sm for %s.', height, month)
        out_file = infile.replace('.tif', '.nc')
        if os.path.exists(out_file) and self.overwrite:
            os.remove(out_file)

        if not os.path.exists(out_file) or self.overwrite:
            ds = xr.open_mfdataset(infile)
            ds = ds.rename(
                {
                    'band_data': f'windspeed_{height}m',
                    'x': 'longitude',
                    'y': 'latitude',
                }
            )
            ds = ds.isel(band=0).drop_vars('band')
            ds.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf')
        return out_file

    def convert_month_tif(self, month):
        """Write netcdf files for all heights for the given month."""
        for height in self.in_heights:
            self.convert_month_height_tif(month, height)

    def convert_all_tifs(self):
        """Write netcdf files for all heights for all months."""
        for i in range(1, 13):
            month = calendar.month_name[i]
            logger.info('Converting tif files to netcdf files for %s', month)
            self.convert_month_tif(month)

    @property
    def mask(self):
        """Mask coordinates without data"""
        if self._mask is None:
            with xr.open_mfdataset(self.get_height_files('January')) as res:
                mask = (res[self.in_features[0]] != -999) & (
                    ~np.isnan(res[self.in_features[0]])
                )
                for feat in self.in_features[1:]:
                    tmp = (res[feat] != -999) & (~np.isnan(res[feat]))
                    mask &= tmp
                self._mask = np.array(mask).flatten()
        return self._mask

    def get_month(self, month):
        """Get interpolated means for all hub heights for the given month.

        Parameters
        ----------
        month : str
            Name of month to get data for

        Returns
        -------
        data : xarray.Dataset
            xarray dataset object containing interpolated monthly windspeed
            means for all input and output heights

        """
        month_file = self.get_output_file(month)
        if os.path.exists(month_file) and self.overwrite:
            os.remove(month_file)

        if os.path.exists(month_file) and not self.overwrite:
            logger.info('Loading month_file %s.', month_file)
            data = xr.open_mfdataset(month_file)
        else:
            logger.info(
                'Getting mean windspeed for all heights (%s) for %s',
                self.in_heights,
                month,
            )
            data = xr.open_mfdataset(self.get_height_files(month))
            logger.info(
                'Interpolating windspeed for all heights (%s) for %s.',
                self.out_heights,
                month,
            )
            data = self.interp(data)
            data.to_netcdf(month_file, format='NETCDF4', engine='h5netcdf')
            logger.info(
                'Saved interpolated means for all heights for %s to %s.',
                month,
                month_file,
            )
        return data

    def interp(self, data):
        """Interpolate data to requested output heights.

        Parameters
        ----------
        data : xarray.Dataset
            xarray dataset object containing windspeed for all input heights

        Returns
        -------
        data : xarray.Dataset
            xarray dataset object containing windspeed for all input and output
            heights
        """
        var_array = np.zeros(
            (
                len(data.latitude) * len(data.longitude),
                len(self.in_heights),
            ),
            dtype=np.float32,
        )
        lev_array = var_array.copy()
        for i, (h, feat) in enumerate(zip(self.in_heights, self.in_features)):
            var_array[..., i] = data[feat].values.flatten()
            lev_array[..., i] = h

        logger.info(
            'Interpolating %s to %s for %s coordinates.',
            self.in_features,
            self.out_features,
            var_array.shape[0],
        )
        tmp = [
            interp1d(h, v, fill_value='extrapolate')(self.out_heights)
            for h, v in zip(lev_array[self.mask], var_array[self.mask])
        ]
        out = np.full(
            (len(data.latitude), len(data.longitude), len(self.out_heights)),
            np.nan,
            dtype=np.float32,
        )
        out[self.mask.reshape((len(data.latitude), len(data.longitude)))] = tmp
        for i, feat in enumerate(self.out_features):
            if feat not in data:
                data[feat] = (('latitude', 'longitude'), out[..., i])
        return data

    def get_lat_lon(self):
        """Get lat lon grid"""
        with xr.open_mfdataset(self.get_height_files('January')) as res:
            lons, lats = np.meshgrid(
                res['longitude'].values, res['latitude'].values
            )
        return np.array(lats), np.array(lons)

    @property
    def meta(self):
        """Get meta with latitude/longitude"""
        if self._meta is None:
            lats, lons = self.get_lat_lon()
            self._meta = pd.DataFrame()
            self._meta['latitude'] = lats.flatten()[self.mask]
            self._meta['longitude'] = lons.flatten()[self.mask]
        return self._meta

    @property
    def time_index(self):
        """Get time index so output conforms to standard format"""
        times = [f'2000-{str(i).zfill(2)}' for i in range(1, 13)]
        time_index = pd.DatetimeIndex(times)
        return time_index

    def get_all_data(self):
        """Get interpolated monthly means for all out heights as a dictionary
        to use for h5 writing.

        Returns
        -------
        out : dict
            Dictionary of arrays containing monthly means for each hub height.
            Also includes latitude and longitude. Spatial dimensions are
            flattened
        """
        data_dict = {}
        s_num = len(self.meta)
        for i in range(1, 13):
            month = calendar.month_name[i]
            out = self.get_month(month)
            for feat in self.out_features:
                if feat not in data_dict:
                    data_dict[feat] = np.full((s_num, 12), np.nan)
                data = out[feat].values.flatten()[self.mask]
                data_dict[feat][..., i - 1] = data
        return data_dict

    @property
    def global_attrs(self):
        """Get dictionary on how this data is prepared"""
        attrs = {
            'input_files': self.input_files,
            'class': str(self.__class__),
            'version_record': str(VERSION_RECORD),
        }
        return attrs

    def write_data(self, fp_out, out):
        """Write monthly means for all heights to h5 file"""
        if fp_out is not None:
            if not os.path.exists(os.path.dirname(fp_out)):
                os.makedirs(os.path.dirname(fp_out), exist_ok=True)

            if not os.path.exists(fp_out) or self.overwrite:
                OutputHandler._init_h5(
                    fp_out, self.time_index, self.meta, self.global_attrs
                )
                with RexOutputs(fp_out, 'a') as f:
                    for dset, data in out.items():
                        OutputHandler._ensure_dset_in_output(fp_out, dset)
                        f[dset] = data.T
                        logger.info('Added %s to %s.', dset, fp_out)

                    logger.info(
                        'Wrote monthly means for all out heights: %s', fp_out
                    )
            elif os.path.exists(fp_out):
                logger.info('%s already exists and overwrite=False.', fp_out)

    @classmethod
    def run(
        cls, path_pattern, in_heights, out_heights, fp_out, overwrite=False
    ):
        """Read vortex tif files, convert these to monthly netcdf files for all
        input heights, interpolate this data to requested output heights, mask
        fill values, and write all data to h5 file.

        Parameters
        ----------
        path_pattern : str
            Pattern for input tif files. Needs to include {month} and {height}
            format keys.
        in_heights : list
            List of heights for input files.
        out_heights : list
            List of output heights used for interpolation
        fp_out : str
            Name of final h5 output file to write with means.
        overwrite : bool
            Whether to overwrite intermediate netcdf files containing the
            interpolated masked monthly means.
        """
        vprep = cls(path_pattern, in_heights, out_heights, overwrite=overwrite)
        vprep.convert_all_tifs()
        out = vprep.get_all_data()
        vprep.write_data(fp_out, out)


class BiasCorrectUpdate:
    """Class for bias correcting existing files and writing corrected files."""

    @classmethod
    def get_bc_factors(cls, bc_file, dset, month, global_scalar=1):
        """Get bias correction factors for the given dset and month

        Parameters
        ----------
        bc_file : str
            Name of h5 file containing bias correction factors
        dset : str
            Name of dataset to apply bias correction factors for
        month : int
            Index of month to bias correct
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.

        Returns
        -------
        factors : ndarray
            Array of bias correction factors for the given dset and month.
        """
        with Resource(bc_file) as res:
            logger.info(
                'Getting %s bias correction factors for month %s.',
                dset,
                month,
            )
            bc_factor = res[f'{dset}_scalar', :, month - 1]
            factors = global_scalar * bc_factor
            logger.info(
                'Retrieved %s bias correction factors for month %s. '
                'Using global_scalar=%s.',
                dset,
                month,
                global_scalar,
            )
        return factors

    @classmethod
    def _correct_month(
        cls, fh_in, month, out_file, dset, bc_file, global_scalar
    ):
        """Bias correct data for a given month.

        Parameters
        ----------
        fh_in : Resource()
            Resource handler for input file being corrected
        month : int
            Index of month to be corrected
        out_file : str
            Name of h5 file containing bias corrected data
        dset : str
            Name of dataset to bias correct
        bc_file : str
            Name of file containing bias correction factors for the given dset
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.
        """
        with RexOutputs(out_file, 'a') as fh:
            mask = fh.time_index.month == month
            mask = np.arange(len(fh.time_index))[mask]
            mask = slice(mask[0], mask[-1] + 1)
            bc_factors = cls.get_bc_factors(
                bc_file=bc_file,
                dset=dset,
                month=month,
                global_scalar=global_scalar,
            )
            logger.info(
                'Applying bias correction factors for month %s', month
            )
            fh[dset, mask, :] = bc_factors * fh_in[dset, mask, :]

    @classmethod
    def update_file(
        cls,
        in_file,
        out_file,
        dset,
        bc_file,
        global_scalar=1,
        max_workers=None,
    ):
        """Update the in_file with bias corrected values for the given dset
        and write to out_file.

        Parameters
        ----------
        in_file : str
            Name of h5 file containing data to bias correct
        out_file : str
            Name of h5 file containing bias corrected data
        dset : str
            Name of dataset to bias correct
        bc_file : str
            Name of file containing bias correction factors for the given dset
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.
        max_workers : int | None
            Number of workers to use for parallel processing.
        """
        tmp_file = get_tmp_file(out_file)
        logger.info(
            'Bias correcting %s in %s with %s.', dset, in_file, bc_file
        )
        with Resource(in_file) as fh_in:
            OutputHandler._init_h5(
                tmp_file, fh_in.time_index, fh_in.meta, fh_in.global_attrs
            )
            OutputHandler._ensure_dset_in_output(tmp_file, dset)

            tasks = []
            for i in range(1, 13):
                task = dask.delayed(cls._correct_month)(
                    fh_in,
                    month=i,
                    out_file=tmp_file,
                    dset=dset,
                    bc_file=bc_file,
                    global_scalar=global_scalar,
                )
                tasks.append(task)

            logger.info('Added %s bias correction futures', len(tasks))
            if max_workers == 1:
                dask.compute(*tasks, scheduler='single-threaded')
            else:
                dask.compute(
                    *tasks, scheduler='threads', num_workers=max_workers
                )
            logger.info('Finished bias correcting %s in %s', dset, in_file)

        os.replace(tmp_file, out_file)
        logger.info('Saved bias corrected %s to: %s', dset, out_file)

    @classmethod
    def run(
        cls,
        in_file,
        out_file,
        dset,
        bc_file,
        overwrite=False,
        global_scalar=1,
        max_workers=None,
    ):
        """Run bias correction update.

        Parameters
        ----------
        in_file : str
            Name of h5 file containing data to bias correct
        out_file : str
            Name of h5 file containing bias corrected data
        dset : str
            Name of dataset to bias correct
        bc_file : str
            Name of file containing bias correction factors for the given dset
        overwrite : bool
            Whether to overwrite the output file if it already exists.
        global_scalar : float
            Optional global scalar to multiply all bias correction
            factors. This can be used to improve systemic bias against
            observation data.
        max_workers : int | None
            Number of workers to use for parallel processing.
        """
        if os.path.exists(out_file) and not overwrite:
                logger.info(
                    '%s already exists and overwrite=False. Skipping.',
                    out_file,
                )
        else:
            if os.path.exists(out_file) and overwrite:
                logger.info(
                    '%s exists but overwrite=True. Removing %s.',
                    out_file,
                    out_file,
                )
                os.remove(out_file)
            cls.update_file(
                in_file,
                out_file,
                dset,
                bc_file,
                global_scalar=global_scalar,
                max_workers=max_workers,
            )
