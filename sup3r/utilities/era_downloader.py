"""Download ERA5 file for the given year and month

Note
----
To use this you need to have ``cdsapi`` package installed and a ``~/.cdsapirc``
file with a url and api key. Follow the instructions here:
https://cds.climate.copernicus.eu/how-to-api
"""

import logging
import os
import pprint
from calendar import monthrange
from warnings import warn

import dask
import dask.array as da
import numpy as np
import pandas as pd
from rex import init_logger

from sup3r.preprocessing import Loader
from sup3r.preprocessing.loaders.utilities import (
    standardize_names,
    standardize_values,
)
from sup3r.preprocessing.names import (
    ERA_NAME_MAP,
    LEVEL_VARS,
    SFC_VARS,
    Dimension,
)
from sup3r.preprocessing.utilities import log_args, ordered_dims
from sup3r.utilities.utilities import get_tmp_file
from sup3r.writers import Cacher

# these are occasionally included in downloaded files, more often with cds-beta
IGNORE_VARS = ('number', 'expver')


logger = logging.getLogger(__name__)


class EraDownloader:
    """Class to handle ERA5 downloading, variable renaming, and file
    combinations.

    Examples
    --------
    Download u / v at 10m, 100m, and specified pressure levels, for CONUS.
    Include orog (surface elevation), and zg (geopotential height), which are
    required for converting from pressure levels to heights. This downloads
    the data for all months, combine into a yearly file, and standardize the
    data for use in the sup3r package.

    >>> area = [53, -132, 20, -60] # [max_lat, min_lon, min_lat, max_lon]
    >>> pressure_levels = [700, 800, 900, 925, 950, 975, 1000] # in hPa
    >>> monthly_fpattern = "./{year}/{month}/{year}_{month}_{var}.nc"
    >>> yearly_fpattern = "./{year}/{year}_{var}.nc"
    >>> variables = ['u', 'v', 'orog', 'zg']
    >>> EraDownloader.run(
            year=2021,
            area=area,
            months=list(range(1, 13)),
            levels=pressure_levels,
            monthly_file_pattern=monthly_fpattern,
            yearly_file_pattern=yearly_fpattern,
            variables=variables,
            max_workers=3,
            product_type='reanalysis'
        )
    """

    @log_args
    def __init__(
        self,
        year,
        month,
        days,
        area,
        levels,
        file_pattern,
        overwrite=False,
        variables=None,
        product_type='reanalysis',
    ):
        """Initialize the class.

        Parameters
        ----------
        year : int
            Year of data to download.
        month : int
            Month of data to download.
        days : list
            List of days to download data for the given month.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        file_pattern : str
            Pattern for combined monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_combined.nc'
        overwrite : bool
            Whether to overwrite existing files.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members', 'monthly_averaged_reanalysis',
            'monthly_averaged_ensemble_members'
        """
        self.year = year
        self.month = month
        self.days = [str(day).zfill(2) for day in days]
        self.area = area
        self.levels = levels
        self.overwrite = overwrite
        self.file_pattern = file_pattern
        self._variables = variables
        self.sfc_file_variables = []
        self.level_file_variables = []
        self.prep_var_lists(self.variables)
        self.product_type = product_type
        self.hours = self.get_hours()

    def get_hours(self):
        """ERA5 is hourly and EDA is 3-hourly. Check and warn for incompatible
        requests."""
        if self.product_type in (
            'monthly_averaged_reanalysis',
            'monthly_averaged_ensemble_members',
        ):
            hours = ['00:00']
        elif self.product_type in (
            'reanalysis',
            'monthly_averaged_reanalysis_by_hour_of_day',
        ):
            hours = [str(n).zfill(2) + ':00' for n in range(0, 24)]
        else:
            hours = [str(n).zfill(2) + ':00' for n in range(0, 24, 3)]
        return hours

    @property
    def variables(self):
        """Get list of requested variables"""
        if self._variables is None:
            raise OSError('Received empty variable list.')
        return self._variables

    @property
    def monthly_file(self):
        """Name of file with all surface and level variables for a given month
        and year."""
        monthly_file = self.file_pattern.replace(
            '{var}', '_'.join(self.variables)
        ).format(year=self.year, month=str(self.month).zfill(2))
        os.makedirs(os.path.dirname(monthly_file), exist_ok=True)
        return monthly_file

    @property
    def surface_file(self):
        """Get name of file with variables from single level download"""
        basedir = os.path.dirname(self.monthly_file)
        basename = os.path.basename(self.monthly_file)
        return os.path.join(basedir, f'sfc_{basename}')

    @property
    def level_file(self):
        """Get name of file with variables from pressure level download"""
        basedir = os.path.dirname(self.monthly_file)
        basename = os.path.basename(self.monthly_file)
        return os.path.join(basedir, f'level_{basename}')

    def _prep_var_lists(self, variables):
        """Add all downloadable variables for the generic requested variables.
        e.g. if variable = 'u' add all downloadable u variables to list.
        """
        d_vars = []
        var_list = variables.copy()
        for i, v in enumerate(var_list):
            if v in ('u', 'v'):
                var_list[i] = f'{v}_'

        all_vars = SFC_VARS + LEVEL_VARS + ['zg', 'orog']
        for var in var_list:
            d_vars.extend([d_var for d_var in all_vars if var in d_var])
        return d_vars

    def prep_var_lists(self, variables):
        """Create surface and level variable lists based on requested
        variables.
        """
        variables = self._prep_var_lists(variables)
        for var in variables:
            if var in SFC_VARS and var not in self.sfc_file_variables:
                self.sfc_file_variables.append(var)
            elif var in LEVEL_VARS and var not in self.level_file_variables:
                self.level_file_variables.append(var)
            elif var not in SFC_VARS + LEVEL_VARS + ['zg', 'orog']:
                msg = f'Requested {var} is not available for download.'
                logger.warning(msg)
                warn(msg)

        sfc_and_level_check = (
            len(self.sfc_file_variables) > 0
            and len(self.level_file_variables) > 0
            and 'orog' not in variables
            and 'zg' not in variables
        )
        if sfc_and_level_check:
            msg = (
                f'Both surface and pressure level variables ({variables}) '
                'were requested without requesting "orog" and "zg".'
            )
            logger.warning(msg)
            warn(msg)

        else:
            if 'orog' in variables:
                self.sfc_file_variables.append('geopotential')
            if 'zg' in variables:
                self.level_file_variables.append('geopotential')

    @staticmethod
    def get_cds_client():
        """Get the copernicus climate data store (CDS) API object for ERA
        downloads."""

        try:
            import cdsapi  # noqa
        except ImportError as e:
            msg = f'Could not import cdsapi package. {e}'
            raise ImportError(msg) from e

        msg = (
            'To download ERA5 data you need to have a ~/.cdsapirc file '
            'with a valid url and api key. Follow the instructions here: '
            'https://cds.climate.copernicus.eu/how-to-api'
        )
        req_file = os.path.join(os.path.expanduser('~'), '.cdsapirc')
        assert os.path.exists(req_file), msg

        return cdsapi.Client()

    def download_process_combine(self):
        """Run the download routine."""
        sfc_check = len(self.sfc_file_variables) > 0
        level_check = (
            len(self.level_file_variables) > 0
            and self.levels is not None
            and len(self.levels) > 0
        )
        if self.level_file_variables:
            msg = (
                f'{self.level_file_variables} requested but no levels'
                ' were provided.'
            )
            if self.levels is None:
                logger.warning(msg)
                warn(msg)

        time_dict = {
            'year': self.year,
            'month': self.month,
            'time': self.hours,
        }
        if 'monthly' not in self.product_type:
            time_dict['day'] = self.days

        if sfc_check:
            tmp_file = get_tmp_file(self.surface_file)
            self.download_file(
                self.sfc_file_variables,
                time_dict=time_dict,
                area=self.area,
                out_file=tmp_file,
                level_type='single',
                overwrite=self.overwrite,
                product_type=self.product_type,
            )
            os.replace(tmp_file, self.surface_file)
            logger.info('Moved %s to %s', tmp_file, self.surface_file)
        if level_check:
            tmp_file = get_tmp_file(self.level_file)
            self.download_file(
                self.level_file_variables,
                time_dict=time_dict,
                area=self.area,
                out_file=tmp_file,
                level_type='pressure',
                levels=self.levels,
                overwrite=self.overwrite,
                product_type=self.product_type,
            )
            os.replace(tmp_file, self.level_file)
            logger.info('Moved %s to %s', tmp_file, self.level_file)
        if sfc_check or level_check:
            self.process_and_combine()

    @classmethod
    def download_file(
        cls,
        variables,
        time_dict,
        area,
        out_file,
        level_type,
        levels=None,
        product_type='reanalysis',
        overwrite=False,
    ):
        """Download either single-level or pressure-level file

        Parameters
        ----------
        variables : list
            List of variables to download
        time_dict : dict
            Dictionary with year, month, day, time entries.
        area : list
            List of bounding box coordinates.
            e.g. [max_lat, min_lon, min_lat, max_lon]
        out_file : str
            Name of output file
        level_type : str
            Either 'single' or 'pressure'
        levels : list
            List of pressure levels to download, if level_type == 'pressure'
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members', 'monthly_averaged_reanalysis',
            'monthly_averaged_ensemble_members'
        overwrite : bool
            Whether to overwrite existing file
        """
        if os.path.exists(out_file) and not cls._can_skip_file(out_file):
            logger.info(
                'Previous download of %s failed. Removing %s.',
                out_file,
                out_file,
            )
            os.remove(out_file)

        if not cls._can_skip_file(out_file) or overwrite:
            logger.info(
                'Downloading %s to %s with levels = %s.',
                variables,
                out_file,
                levels,
            )
            dataset = f'reanalysis-era5-{level_type}-levels'
            if 'monthly' in product_type:
                dataset += '-monthly-means'
            entry = {
                'product_type': [product_type],
                'data_format': 'netcdf',
                'download_format': 'unarchived',
                'variable': variables,
                'area': area,
            }
            entry.update(time_dict)
            if level_type == 'pressure':
                entry['pressure_level'] = levels
            logger.info(
                'Calling CDS-API with dataset=%s, entry=%s.', dataset, entry
            )
            cds_api_client = cls.get_cds_client()

            cds_api_client.retrieve(dataset, entry, out_file)
        else:
            logger.info('File already exists: %s.', out_file)

    def process_surface_file(self):
        """Rename variables and convert geopotential to geopotential height."""
        tmp_file = get_tmp_file(self.surface_file)
        ds = Loader(self.surface_file)

        if 'z' in ds.data_vars:
            logger.info(
                'Converting "z" var to "orog" for %s', self.surface_file
            )
            ds = self.convert_z(ds, name='orog')

        ds = standardize_names(ds, ERA_NAME_MAP)
        ds = standardize_values(ds)

        if 'monthly' in self.product_type:
            ds['time'] = pd.DatetimeIndex([
                f'{self.year}-{str(self.month).zfill(2)}-01'
            ])
        ds.compute().to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')
        os.replace(tmp_file, self.surface_file)
        logger.info(
            'Finished processing %s. Moved %s to %s.',
            self.surface_file,
            tmp_file,
            self.surface_file,
        )

    def add_pressure(self, ds):
        """Add pressure to dataset

        Parameters
        ----------
        ds : Dataset
            xr.Dataset() object for which to add pressure

        Returns
        -------
        ds : Dataset
        """
        if 'pressure' in self.variables and 'pressure' not in ds.data_vars:
            logger.info('Adding pressure variable.')
            pres = 100 * ds[Dimension.PRESSURE_LEVEL].values.astype(np.float32)

            # if trailing dimensions don't match this is for an ensemble
            # download
            dims = {k: ds.sizes[k] for k in ordered_dims(ds.dims)}
            arr_shape = list(dims.values())
            if len(pres) != arr_shape[-1]:
                pres = np.repeat(pres[..., None], arr_shape[-1], axis=-1)
            ds['pressure'] = (
                dims,
                da.broadcast_to(pres, arr_shape),
            )
            ds['pressure'].attrs['units'] = 'Pa'
        return ds

    def convert_z(self, ds, name):
        """Convert z to given height variable

        Parameters
        ----------
        ds : Dataset
            xr.Dataset() object for new file
        name : str
            Variable name. e.g. zg or orog, typically

        Returns
        -------
        ds : Dataset
            xr.Dataset() object for new file with new height variable written.
        """
        if name not in ds.data_vars and 'z' in ds.data_vars:
            ds['z'] = (ds['z'].dims, ds['z'].values / 9.81)
            ds['z'].attrs['units'] = 'm'
            ds = ds.rename({'z': name})
        return ds

    def process_level_file(self):
        """Convert geopotential to geopotential height."""
        tmp_file = get_tmp_file(self.level_file)
        ds = Loader(self.level_file)

        if 'z' in ds.data_vars:
            logger.info('Converting "z" var to "zg" for %s', self.level_file)
            ds = self.convert_z(ds, name='zg')

        ds = standardize_names(ds, ERA_NAME_MAP)
        ds = standardize_values(ds)
        ds = self.add_pressure(ds)
        if 'monthly' in self.product_type:
            ds['time'] = pd.DatetimeIndex([
                f'{self.year}-{str(self.month).zfill(2)}-01'
            ])
        ds.compute().to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')
        os.replace(tmp_file, self.level_file)
        logger.info(
            'Finished processing %s. Moved %s to %s.',
            self.level_file,
            tmp_file,
            self.level_file,
        )

    def process_and_combine(self):
        """Process variables and combine surface and level files into a single
        monthly file. Proccesing includes renaming variables and converting to
        standard units."""

        if os.path.exists(self.monthly_file) and not self._can_skip_file(
            self.monthly_file
        ):
            os.remove(self.monthly_file)

        if not self._can_skip_file(self.monthly_file) or self.overwrite:
            files = []
            if os.path.exists(self.level_file):
                logger.info('Processing %s.', self.level_file)
                self.process_level_file()
                files.append(self.level_file)
            if os.path.exists(self.surface_file):
                logger.info('Processing %s.', self.surface_file)
                self.process_surface_file()
                files.append(self.surface_file)

            kwargs = {'compat': 'override'}
            self._combine_files(files, self.monthly_file, kwargs)

            if os.path.exists(self.level_file):
                os.remove(self.level_file)
            if os.path.exists(self.surface_file):
                os.remove(self.surface_file)
        else:
            logger.info('%s already exists.', self.monthly_file)

    def get_monthly_file(self):
        """Download level and surface files, process variables, and combine
        processed files. Includes checks for shape and variables."""
        if os.path.exists(self.monthly_file) and (
            not self._can_skip_file(self.monthly_file) or self.overwrite
        ):
            os.remove(self.monthly_file)

        if not os.path.exists(self.monthly_file):
            self.download_process_combine()

    @classmethod
    def all_vars_exist(cls, year, file_pattern, variables):
        """Check if all yearly variable files for the requested year exist.

        Parameters
        ----------
        year : int
            Year used for data download.
        file_pattern : str
            Pattern for variable file. Must include year and
            var format keys. e.g. 'era5_{year}_{var}_combined.nc'
        variables : list
            Variables that should have been downloaded

        Returns
        -------
        bool
            True if all monthly variable files for the requested year and month
            exist.
        """
        return all(
            os.path.exists(file_pattern.format(year=year, var=var))
            for var in variables
        )

    @classmethod
    def run_month(
        cls,
        year,
        month,
        days,
        area,
        levels,
        file_pattern,
        overwrite=False,
        variables=None,
        product_type='reanalysis',
    ):
        """Run routine for the given month and year.

        Parameters
        ----------
        year : int
            Year of data to download.
        month : int
            Month of data to download.
        days : list
            List of days to download data for the given month.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        file_pattern : str
            Pattern for combined monthly output file. Must include year and
            month format keys.  e.g. 'era5_{year}_{month}_combined.nc'
        overwrite : bool
            Whether to overwrite existing files.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members', 'monthly_averaged_reanalysis',
            'monthly_averaged_ensemble_members'
        """
        variables = variables if isinstance(variables, list) else [variables]
        for var in variables:
            downloader = cls(
                year=year,
                month=month,
                days=days,
                area=area,
                levels=levels,
                file_pattern=file_pattern,
                overwrite=overwrite,
                variables=[var],
                product_type=product_type,
            )
            downloader.get_monthly_file()

    @classmethod
    def run_for_var(
        cls,
        year,
        area,
        levels,
        monthly_file_pattern,
        yearly_file_pattern=None,
        months=None,
        days=None,
        overwrite=False,
        max_workers=None,
        variable=None,
        product_type='reanalysis',
        chunks='auto',
        res_kwargs=None,
    ):
        """Run routine for all requested months in the requested year for the
        given variable.

        Parameters
        ----------
        year : int
            Year of data to download.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        monthly_file_pattern : str
            Pattern for monthly output files. Must include year, month, and var
            format keys.  e.g. 'era5_{year}_{month}_{var}.nc'
        yearly_file_pattern : str
            Pattern for yearly output files. Must include year and var format
            keys.  e.g. 'era5_{year}_{var}.nc'
        months : list | None
            List of months to download data for. If None then all months for
            the given year will be downloaded.
        days : list | None
            List of days to download data for. If None then all days for the
            given months will be downloaded. This should be a list of lists
            with an entry for each month. e.g. [[1, 2], [1, 2, 3]]
        overwrite : bool
            Whether to overwrite existing files.
        max_workers : int
            Max number of workers to use for downloading and processing monthly
            files.
        variable : str
            Variable to download.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members', 'monthly_averaged_reanalysis',
            'monthly_averaged_ensemble_members'
        chunks : str | dict
            Dictionary of chunksizes used when writing data to netcdf files.
            Can also be 'auto'.
        """
        if yearly_file_pattern is not None:
            yearly_var_file = yearly_file_pattern.format(
                year=year, var=variable
            )
            if os.path.exists(yearly_var_file) and not overwrite:
                logger.info(
                    '%s already exists and overwrite=False.', yearly_var_file
                )
        msg = 'file_pattern must have {year}, {month}, and {var} format keys'
        assert all(
            key in monthly_file_pattern
            for key in ('{year}', '{month}', '{var}')
        ), msg

        tasks = []
        months = range(1, 13) if months is None else months
        if days is None:
            days = [
                range(1, monthrange(year, month)[1] + 1)
                for month in months
            ]
        days = [[str(day).zfill(2) for day in d] for d in days]
        for i, month in enumerate(months):
            task = dask.delayed(cls.run_month)(
                year=year,
                month=month,
                days=days[i],
                area=area,
                levels=levels,
                file_pattern=monthly_file_pattern,
                overwrite=overwrite,
                variables=[variable],
                product_type=product_type,
            )
            tasks.append(task)

        if max_workers == 1:
            dask.compute(*tasks, scheduler='single-threaded')
        else:
            dask.compute(*tasks, scheduler='threads', num_workers=max_workers)

        if yearly_file_pattern is not None and len(months) == 12:
            cls.make_yearly_var_file(
                year,
                monthly_file_pattern,
                yearly_file_pattern,
                variable,
                chunks=chunks,
                res_kwargs=res_kwargs,
            )

    @classmethod
    def run(
        cls,
        year,
        area,
        levels,
        monthly_file_pattern,
        yearly_file_pattern=None,
        months=None,
        days=None,
        overwrite=False,
        max_workers=None,
        variables=None,
        product_type='reanalysis',
        chunks='auto',
        combine_all_files=False,
        res_kwargs=None,
    ):
        """Run routine for all requested months in the requested year.

        Parameters
        ----------
        year : int
            Year of data to download.
        area : list
            Domain area of the data to download.
            [max_lat, min_lon, min_lat, max_lon]
        levels : list
            List of pressure levels to download.
        monthly_file_pattern : str
            Pattern for monthly output file. Must include year, month, and var
            format keys.  e.g. 'era5_{year}_{month}_{var}_combined.nc'
        yearly_file_pattern : str
            Pattern for yearly output file. Must include year and var
            format keys.  e.g. 'era5_{year}_{var}_combined.nc'
        months : list | None
            List of months to download data for. If None then all months for
            the given year will be downloaded.
        days : list | None
            List of days to download data for. If None then all days for the
            given months will be downloaded. This should be a list of lists
            with an entry for each month. e.g. [[1, 2], [1, 2, 3]]
        overwrite : bool
            Whether to overwrite existing files.
        max_workers : int
            Max number of workers to use for downloading and processing monthly
            files.
        variables : list | None
            Variables to download. If None this defaults to just gepotential
            and wind components.
        product_type : str
            Can be 'reanalysis', 'ensemble_mean', 'ensemble_spread',
            'ensemble_members', 'monthly_averaged_reanalysis',
            'monthly_averaged_ensemble_members'
        chunks : str | dict
            Dictionary of chunksizes used when writing data to netcdf files.
            Can also be 'auto'
        combine_all_files : bool
            Whether to combine separate yearly variable files into a single
            yearly file with all variables included
        """
        for var in variables:
            cls.run_for_var(
                year=year,
                area=area,
                levels=levels,
                months=months,
                days=days,
                monthly_file_pattern=monthly_file_pattern,
                yearly_file_pattern=yearly_file_pattern,
                overwrite=overwrite,
                variable=var,
                product_type=product_type,
                max_workers=max_workers,
                chunks=chunks,
                res_kwargs=res_kwargs,
            )

        if yearly_file_pattern is not None and (
            cls.all_vars_exist(
                year=year,
                file_pattern=yearly_file_pattern,
                variables=variables,
            )
            and combine_all_files
        ):
            cls.make_yearly_file(
                year,
                yearly_file_pattern,
                variables,
                chunks=chunks,
                res_kwargs=res_kwargs,
            )

    @classmethod
    def make_yearly_var_file(
        cls,
        year,
        monthly_file_pattern,
        yearly_file_pattern,
        variable,
        chunks='auto',
        res_kwargs=None,
    ):
        """Combine monthly variable files into a single yearly variable file.

        Parameters
        ----------
        year : int
            Year used to download data
        monthly_file_pattern : str
            File pattern for monthly variable files. Must have year, month, and
            var format keys. e.g. './era_{year}_{month}_{var}_combined.nc'
        yearly_file_pattern : str
            File pattern for yearly variable files. Must have year and var
            format keys. e.g. './era_{year}_{var}_combined.nc'
        variable : string
            Variable name for the files to be combined.
        chunks : str | dict
            Dictionary of chunksizes used when writing data to netcdf files.
            Can also be 'auto'.
        res_kwargs : None | dict
            Keyword arguments for base resource handler, like
            ``xr.open_mfdataset.`` This is passed to a ``Loader`` object and
            then used in the base loader contained by that obkect.
        """
        files = [
            monthly_file_pattern.format(
                year=year, month=str(month).zfill(2), var=variable
            )
            for month in range(1, 13)
        ]

        out_file = yearly_file_pattern.format(year=year, var=variable)
        default_kwargs = {
            'combine': 'nested',
            'concat_dim': 'time',
            'coords': 'minimal',
        }
        res_kwargs = res_kwargs or {}
        default_kwargs.update(res_kwargs)
        cls._combine_files(
            files, out_file, chunks=chunks, res_kwargs=res_kwargs
        )

    @classmethod
    def _can_skip_file(cls, file):
        """Make sure existing file has successfully downloaded and can be
        opened."""
        if not os.path.exists(file):
            return False

        logger.info(
            '%s already exists. Making sure it downloaded successfully.', file
        )
        openable = True
        try:
            _ = Loader(file)
        except Exception as e:
            msg = 'Could not open %s. %s. Will redownload.'
            logger.warning(msg, file, e)
            warn(msg % (file, e))
            openable = False

        return openable

    @classmethod
    def _combine_files(cls, files, out_file, chunks='auto', res_kwargs=None):
        if not os.path.exists(out_file):
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            logger.info('Combining %s into %s.', files, out_file)
            try:
                res_kwargs = res_kwargs or {}
                loader = Loader(files, res_kwargs=res_kwargs)
                for ignore_var in IGNORE_VARS:
                    if ignore_var in loader.coords:
                        loader.data = loader.data.drop_vars(ignore_var)
                Cacher._write_single(
                    data=loader.data,
                    out_file=out_file,
                    max_workers=1,
                    chunks=chunks,
                )
            except Exception as e:
                msg = 'Error combining %s. %s'
                logger.error(msg, files, e)
                raise RuntimeError(msg % (files, e)) from e
        else:
            logger.info('%s already exists.', out_file)

    @classmethod
    def make_yearly_file(
        cls, year, file_pattern, variables, chunks='auto', res_kwargs=None
    ):
        """Combine yearly variable files into a single file.

        Parameters
        ----------
        year : int
            Year for the data to make into a yearly file.
        file_pattern : str
            File pattern for output files. Must have year and var
            format keys. e.g. './era_{year}_{var}_combined.nc'
        variables : list
            List of variables corresponding to the yearly variable files to
            combine.
        chunks : str | dict
            Dictionary of chunksizes used when writing data to netcdf files.
            Can also be 'auto'.
        res_kwargs : None | dict
            Keyword arguments for base resource handler, like
            ``xr.open_mfdataset.`` This is passed to a ``Loader`` object and
            then used in the base loader contained by that obkect.
        """
        msg = (
            f'Not all variable files with file_patten {file_pattern} for '
            f'year {year} exist.'
        )
        assert cls.all_vars_exist(
            year=year, file_pattern=file_pattern, variables=variables
        ), msg

        files = [file_pattern.format(year=year, var=var) for var in variables]
        yearly_file = (
            file_pattern.replace('_{var}_', '')
            .replace('_{var}', '')
            .format(year=year)
        )
        cls._combine_files(
            files, yearly_file, res_kwargs=res_kwargs, chunks=chunks
        )

    @classmethod
    def run_qa(cls, file, res_kwargs=None, log_file=None):
        """Check for NaN values and log min / max / mean / stds for all
        variables."""

        logger = init_logger(__name__, log_level='DEBUG', log_file=log_file)
        with Loader(file, res_kwargs=res_kwargs) as res:
            logger.info('Running qa on file: %s', file)
            logger.info('\n%s', pprint.pformat(res.qa(), indent=2))
