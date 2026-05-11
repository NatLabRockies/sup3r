"""pytests for general utilities"""

import os

import numpy as np
import pandas as pd

from sup3r.preprocessing.names import FEATURE_NAMES
from sup3r.utilities.era_downloader import EraDownloader
from sup3r.utilities.pytest.helpers import make_fake_dset
from sup3r.utilities.utilities import xr_open_mfdataset


class EraDownloaderTester(EraDownloader):
    """Testing version of era downloader with download_file method overridden
    since we wont include a cdsapi key in tests."""

    # pylint: disable=unused-argument
    @classmethod
    def download_file(
        cls,
        variables,
        time_dict,
        area,  # noqa
        out_file,
        level_type,
        levels=None,
        **kwargs,  # noqa
    ):
        """Download either single-level or pressure-level file"""
        n_days = pd.Period(
            f'{time_dict["year"]}-{time_dict["month"]}-01'
        ).days_in_month
        ti = pd.date_range(
            f'{time_dict["year"]}-{time_dict["month"]}-01',
            f'{time_dict["year"]}-{time_dict["month"]}-{n_days}',
            freq='D',
        )
        shape = (10, 10, len(ti))
        if levels is not None:
            shape = (*shape, len(levels))

        features = []

        name_map = {
            '10m_u_component_of_wind': 'u10',
            '10m_v_component_of_wind': 'v10',
            '100m_u_component_of_wind': 'u100',
            '100m_v_component_of_wind': 'v100',
            'u_component_of_wind': 'u',
            'v_component_of_wind': 'v',
        }

        if 'geopotential' in variables:
            features.append('z')
        features.extend([v for f, v in name_map.items() if f in variables])

        nc = make_fake_dset(shape=shape, features=features)
        nc['time'] = ti
        if 'z' in nc:
            if level_type == 'single':
                nc['z'] = (nc['z'].dims, np.zeros(nc['z'].shape))
            else:
                arr = np.zeros(nc['z'].shape)
                for i in range(nc['z'].shape[1]):
                    arr[:, i, ...] = i * 100
                nc['z'] = (nc['z'].dims, arr)
        nc.to_netcdf(out_file)


def test_era_dl(tmpdir_factory):
    """Test basic post proc for era downloader."""

    variables = ['zg', 'orog', 'u', 'v', 'pressure']
    file_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_{var}.nc'
    )
    year = 2000
    month = 1
    area = [50, -130, 23, -65]
    levels = [1000, 900, 800]
    EraDownloaderTester.run_month(
        year=year,
        month=month,
        days=[[1, 2, 3]],
        area=area,
        levels=levels,
        file_pattern=file_pattern,
        variables=variables,
    )
    for v in variables:
        standard_name = FEATURE_NAMES.get(v, v)
        tmp = xr_open_mfdataset(
            file_pattern.format(year=2000, month='01', var=v)
        )
        assert standard_name in tmp


def test_era_dl_year(tmpdir_factory):
    """Test post proc for era downloader, including log interpolation, for full
    year."""

    variables = ['zg', 'orog', 'u', 'v', 'pressure']
    file_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{month}_{var}.nc'
    )
    yearly_file_pattern = os.path.join(
        tmpdir_factory.mktemp('tmp'), 'era5_{year}_{var}_final.nc'
    )
    EraDownloaderTester.run(
        year=2000,
        area=[50, -130, 23, -65],
        levels=[1000, 900, 800],
        variables=variables,
        monthly_file_pattern=file_pattern,
        yearly_file_pattern=yearly_file_pattern,
        max_workers=1,
        combine_all_files=True,
        res_kwargs={'compat': 'no_conflicts'},
    )

    combined_file = yearly_file_pattern.replace('_{var}_', '').format(
        year=2000
    )
    tmp = xr_open_mfdataset(combined_file)
    for v in variables:
        standard_name = FEATURE_NAMES.get(v, v)
        assert standard_name in tmp
