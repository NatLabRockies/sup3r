"""Test caching by :class:`Deriver` objects"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.preprocessing import DataHandler, DataHandlerH5WindCC, Loader
from sup3r.utilities.pytest.helpers import make_fake_dset
from sup3r.writers import Cacher

target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']


@pytest.mark.parametrize('ext', ['h5', 'nc'])
def test_cacher_attrs(ext):
    """Make sure attributes are preserved in cached data."""
    with tempfile.TemporaryDirectory() as td:
        nc = make_fake_dset(shape=(10, 10, 10), features=['ws_100m'])
        nc['ws_100m'].attrs = {'attrs': 'test'}
        other_attrs = {
            'GRIB_centre': 'ecmf',
            'Description': 'European Centre for Medium-Range Weather...',
            'GRIB_subCentre': 0,
            'Conventions': 'CF-1.7',
            'date_modified': '2024-10-20T22:02:04.598215',
            'global_attrs': '',
        }
        nc.attrs.update(other_attrs)
        tmp_file = td + '/test.nc'
        nc.to_netcdf(tmp_file)
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        DataHandler(
            tmp_file,
            features=['windspeed_100m'],
            feature_aliases={'ws_100m': 'windspeed_100m'},
            cache_kwargs={
                'cache_pattern': cache_pattern,
                'max_workers': 1,
                'attrs': {
                    'windspeed_100m': {
                        'units': 'm/s',
                        'long_name': 'Wind Speed at 100m',
                    }
                },
            },
        )

        out = Loader(cache_pattern.format(feature='windspeed_100m'))
        assert out.data['windspeed_100m'].attrs == {
            'attrs': 'test',
            'units': 'm/s',
            'long_name': 'Wind Speed at 100m'
        }
        global_attrs = {**other_attrs, 'source_files': tmp_file}
        for k, v in global_attrs.items():
            assert out.attrs[k] == v


@pytest.mark.parametrize(
    ('input_files', 'derive_features', 'ext', 'shape', 'target'),
    [
        (
            pytest.FP_WTK,
            ['u_100m', 'v_100m', 'sza'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            pytest.FP_ERA,
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_derived_data_caching(
    input_files, derive_features, ext, shape, target
):
    """Test feature derivation followed by caching/loading"""

    chunks = {'time': 1000, 'south_north': 5, 'west_east': 5}
    res_kwargs = {'engine': 'netcdf4'} if ext == 'nc' else {}
    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandler(
            file_paths=input_files,
            features=derive_features,
            shape=shape,
            target=target,
            chunks=chunks,
            res_kwargs=res_kwargs,
        )

        cacher = Cacher(
            deriver.data,
            cache_kwargs={
                'cache_pattern': cache_pattern,
                'chunks': chunks,
                'max_workers': 1,
            },
        )

        assert deriver.shape[:3] == (shape[0], shape[1], deriver.shape[2])
        assert all(
            deriver[f].shape == (*shape, deriver.shape[2])
            for f in derive_features
        )

        loader = DataHandler(cacher.out_files, features=derive_features)
        assert all(
            loader[feature].dtype == deriver[feature].dtype
            for feature in derive_features
        )
        loaded = loader.as_array().compute()
        derived = deriver.as_array().compute()
        assert np.array_equal(loaded, derived)


@pytest.mark.parametrize('ext', ['h5', 'nc'])
def test_time_feature_cache_round_trip(ext):
    """Cached time features preserve their full precision."""
    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandler(
            pytest.FP_WTK,
            features=['time_feature'],
            time_slice=slice(0, 5),
            shape=shape,
            target=target,
        )
        cacher = Cacher(
            deriver.data,
            cache_kwargs={
                'cache_pattern': cache_pattern,
                'max_workers': 1,
            },
        )
        loader = DataHandler(cacher.out_files, features=['time_feature'])

        derived = deriver['time_feature'].values
        loaded = loader['time_feature'].values
        assert loaded.dtype == np.float64
        np.testing.assert_array_equal(loaded, derived)


@pytest.mark.parametrize(
    ['input_files', 'derive_features', 'ext', 'shape', 'target'],
    [
        (
            pytest.FP_WTK,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        ),
        (
            pytest.FP_ERA,
            ['windspeed_100m', 'winddirection_100m'],
            'nc',
            (10, 10),
            (37.25, -107),
        ),
    ],
)
def test_caching_with_dh_loading(
    input_files, derive_features, ext, shape, target
):
    """Test feature derivation followed by caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandler(
            file_paths=input_files,
            features=derive_features,
            shape=shape,
            target=target,
        )

        cacher = Cacher(
            deriver.data,
            cache_kwargs={'cache_pattern': cache_pattern, 'max_workers': 1},
        )

        assert deriver.shape[:3] == (shape[0], shape[1], deriver.shape[2])
        assert all(
            deriver[f].shape == (*shape, deriver.shape[2])
            for f in derive_features
        )

        loader = DataHandler(cacher.out_files, features=derive_features)
        assert all(
            loader[feature].dtype == deriver[feature].dtype
            for feature in derive_features
        )
        assert np.array_equal(
            loader.as_array().compute(), deriver.as_array().compute()
        )


@pytest.mark.parametrize(
    ['input_files', 'derive_features', 'ext', 'shape', 'target'],
    [
        (
            pytest.FP_WTK,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        )
    ],
)
def test_caching_and_loading_with_daily_dh(
    input_files, derive_features, ext, shape, target
):
    """Test caching and loading for daily data handler which has both hourly
    and daily data."""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandlerH5WindCC(
            file_paths=input_files,
            features=derive_features,
            shape=shape,
            target=target,
            cache_kwargs={'cache_pattern': cache_pattern, 'max_workers': 1},
        )

        assert deriver.shape[:3] == (shape[0], shape[1], deriver.shape[2])
        assert all(
            deriver.hourly[f].shape == (*shape, deriver.shape[2])
            for f in derive_features
        )
        assert deriver.data.dtype == np.dtype(np.float32)

        loader = DataHandlerH5WindCC(
            [cache_pattern.format(feature=f) for f in derive_features],
            features=derive_features,
        )
        assert np.array_equal(loader.hourly.values, deriver.hourly.values)
        ld = loader.daily.values
        dd = deriver.daily.values
        assert np.allclose(ld, dd, atol=1e-6)


@pytest.mark.parametrize(
    ['input_files', 'derive_features', 'ext', 'shape', 'target'],
    [
        (
            pytest.FP_WTK,
            ['u_100m', 'v_100m'],
            'h5',
            (20, 20),
            (39.01, -105.15),
        )
    ],
)
def test_caching_and_loading_with_daily_dh_hr_coarsen(
    input_files, derive_features, ext, shape, target
):
    """Test caching and loading for daily data handler which is also spatially
    coarsened before caching."""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        deriver = DataHandlerH5WindCC(
            file_paths=input_files,
            features=derive_features,
            shape=shape,
            hr_spatial_coarsen=2,
            target=target,
            cache_kwargs={'cache_pattern': cache_pattern, 'max_workers': 1},
        )

        assert deriver.shape[:3] == (
            shape[0] / 2,
            shape[1] / 2,
            deriver.shape[2],
        )
        assert all(
            deriver.hourly[f].shape
            == (shape[0] / 2, shape[1] / 2, deriver.shape[2])
            for f in derive_features
        )
        assert deriver.data.dtype == np.dtype(np.float32)

        loader = DataHandlerH5WindCC(
            file_paths=input_files,
            features=derive_features,
            shape=shape,
            hr_spatial_coarsen=2,
            target=target,
            cache_kwargs={'cache_pattern': cache_pattern, 'max_workers': 1},
        )
        assert np.array_equal(loader.hourly.values, deriver.hourly.values)
        ld = loader.daily.values
        dd = deriver.daily.values
        assert np.allclose(ld, dd, atol=1e-6)
