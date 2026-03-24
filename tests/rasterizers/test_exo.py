"""Test correct functioning of exogenous data specific rasterizers"""

import os
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from rex import Resource

from sup3r.preprocessing import (
    BaseExoRasterizer,
    Dimension,
    ExoData,
    ExoDataHandler,
    ExoRasterizer,
)
from sup3r.utilities.pytest.helpers import make_fake_dset
from sup3r.utilities.utilities import RANDOM_GENERATOR, xr_open_mfdataset
from sup3r.writers import RexOutputs

TARGET = (13.67, 125.0)
SHAPE = (8, 8)
S_ENHANCE = [1, 4]
T_ENHANCE = [1, 2]


def test_exo_data_init():
    """Make sure `ExoData` raises the correct error with bad input."""
    with pytest.raises(ValueError):
        ExoData(steps=['dummy'])


@pytest.mark.parametrize(
    'feature',
    [
        'topography',
        'sza',
        'latitude_feature',
        'longitude_feature',
        'time_feature',
    ],
)
def test_exo_cache(feature):
    """Test exogenous data caching and re-load"""
    # no cached data
    steps = []
    for s_en, t_en in zip(S_ENHANCE, T_ENHANCE):
        steps.append({
            's_enhance': s_en,
            't_enhance': t_en,
            'combine_type': 'input',
            'model': 0,
        })
    with TemporaryDirectory() as td:
        fp_topo = make_topo_file(pytest.FPS_GCM[0], td)
        base = ExoDataHandler(
            file_paths=pytest.FPS_GCM,
            feature=feature,
            source_files=fp_topo,
            steps=steps,
            input_handler_kwargs={'target': TARGET, 'shape': SHAPE},
            input_handler_name='Rasterizer',
            cache_dir=os.path.join(td, 'exo_cache'),
        )
        for i, arr in enumerate(base.data[feature]['steps']):
            assert arr.shape[0] == SHAPE[0] * S_ENHANCE[i]
            assert arr.shape[1] == SHAPE[1] * S_ENHANCE[i]

        assert len(os.listdir(f'{td}/exo_cache')) == 2

        # load cached data
        cache = ExoDataHandler(
            file_paths=pytest.FPS_GCM,
            feature=feature,
            source_files=pytest.FP_WTK,
            steps=steps,
            input_handler_kwargs={'target': TARGET, 'shape': SHAPE},
            input_handler_name='Rasterizer',
            cache_dir=os.path.join(td, 'exo_cache'),
        )
        assert len(os.listdir(f'{td}/exo_cache')) == 2

        for arr1, arr2 in zip(
            base.data[feature]['steps'], cache.data[feature]['steps']
        ):
            assert np.allclose(arr1['data'], arr2['data'])


def get_lat_lon_range_h5(fp):
    """Get the min/max lat/lon from an h5 file"""
    with Resource(fp) as wtk:
        lat_range = (
            wtk.meta[Dimension.LATITUDE].min(),
            wtk.meta[Dimension.LATITUDE].max(),
        )
        lon_range = (
            wtk.meta[Dimension.LONGITUDE].min(),
            wtk.meta[Dimension.LONGITUDE].max(),
        )
    return lat_range, lon_range


def get_lat_lon_range_nc(fp):
    """Get the min/max lat/lon from a netcdf file"""

    dset = xr_open_mfdataset(fp)
    lat_range = (dset['lat'].values.min(), dset['lat'].values.max())
    lon_range = (dset['lon'].values.min(), dset['lon'].values.max())
    return lat_range, lon_range


def make_topo_file(fp, td, N=100, offset=0.1):
    """Make a dummy h5 file with high-res topo for testing"""

    if fp.endswith('.h5'):
        lat_range, lon_range = get_lat_lon_range_h5(fp)
    else:
        lat_range, lon_range = get_lat_lon_range_nc(fp)

    lat = np.linspace(lat_range[0] - offset, lat_range[1] + offset, N)
    lon = np.linspace(lon_range[0] - offset, lon_range[1] + offset, N)
    idy, idx = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))
    lon, lat = np.meshgrid(lon, lat)
    lon, lat = lon.flatten(), lat.flatten()
    idy, idx = idy.flatten(), idx.flatten()
    scale = 30
    elevation = np.sin(scale * np.deg2rad(idy) + scale * np.deg2rad(idx))
    meta = pd.DataFrame({
        Dimension.LATITUDE: lat,
        Dimension.LONGITUDE: lon,
        'elevation': elevation,
    })

    fp_temp = os.path.join(td, 'elevation.h5')
    with RexOutputs(fp_temp, mode='w') as out:
        out.meta = meta

    return fp_temp


def make_srl_file(fp, td, N=100, offset=0.1):
    """Make a dummy h5 file with high-res srl for testing"""

    if fp.endswith('.h5'):
        lat_range, lon_range = get_lat_lon_range_h5(fp)
    else:
        lat_range, lon_range = get_lat_lon_range_nc(fp)

    lat = np.linspace(lat_range[0] - offset, lat_range[1] + offset, N)
    lon = np.linspace(lon_range[0] - offset, lon_range[1] + offset, N)
    idy, idx = np.meshgrid(np.arange(len(lon)), np.arange(len(lat)))
    lon, lat = np.meshgrid(lon, lat)
    lon, lat = lon.flatten(), lat.flatten()
    idy, idx = idy.flatten(), idx.flatten()
    srl = RANDOM_GENERATOR.uniform(0, 1, len(lat))
    meta = pd.DataFrame({
        Dimension.LATITUDE: lat,
        Dimension.LONGITUDE: lon,
    })

    fp_temp = os.path.join(td, 'srl.h5')
    with RexOutputs(fp_temp, mode='w') as out:
        out.meta = meta
        out.add_dataset(fp_temp, 'srl', srl, dtype=np.float32)

    return fp_temp


@pytest.mark.parametrize('s_enhance', [1, 2])
def test_srl_extraction_h5(s_enhance):
    """Test the spatial enhancement of a test grid and then the lookup of the
    srl data. Tests general exo rasterization for new feature"""
    with tempfile.TemporaryDirectory() as td:
        fp_exo_srl = make_srl_file(pytest.FP_WTK, td)

        kwargs = {
            'file_paths': pytest.FP_WTK,
            'source_files': fp_exo_srl,
            'feature': 'srl',
            's_enhance': s_enhance,
            't_enhance': 1,
            'input_handler_kwargs': {
                'target': (39.01, -105.15),
                'shape': (20, 20),
            },
            'cache_dir': f'{td}/exo_cache/',
        }

        te = BaseExoRasterizer(**kwargs)

        te_gen = ExoRasterizer(**{
            k: v for k, v in kwargs.items() if k != 'cache_dir'
        })

        assert np.array_equal(te.data.as_array(), te_gen.data.as_array())

        hr_srl = np.asarray(te.data.as_array())

        lat = te.hr_lat_lon[..., 0].flatten()
        lon = te.hr_lat_lon[..., 1].flatten()
        hr_wtk_meta = np.vstack((lat, lon)).T
        hr_wtk_ind = np.arange(len(lat)).reshape(te.hr_shape[:-1])
        assert te.nn.max() == len(hr_wtk_meta)

        for gid in RANDOM_GENERATOR.choice(
            len(hr_wtk_meta), 50, replace=False
        ):
            idy, idx = np.where(hr_wtk_ind == gid)
            iloc = np.where(te.nn == gid)[0]
            exo_coords = te.source_lat_lon[iloc]

            # make sure all mapped high-res exo coordinates are closest to gid
            # pylint: disable=consider-using-enumerate
            for i in range(len(exo_coords)):
                dist = hr_wtk_meta - exo_coords[i]
                dist = np.hypot(dist[:, 0], dist[:, 1])
                assert np.argmin(dist) == gid

            # make sure the mean srlation makes sense
            test_out = hr_srl[idy, idx, 0]
            true_out = te.source_data[iloc].mean()
            assert np.allclose(test_out, true_out)


@pytest.mark.parametrize('s_enhance', [1, 2])
def test_topo_extraction_h5(s_enhance):
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WTK file (also the same file for the test)"""
    with tempfile.TemporaryDirectory() as td:
        fp_exo_topo = make_topo_file(pytest.FP_WTK, td)

        kwargs = {
            'file_paths': pytest.FP_WTK,
            'source_files': fp_exo_topo,
            'feature': 'topography',
            's_enhance': s_enhance,
            't_enhance': 1,
            'input_handler_kwargs': {
                'target': (39.01, -105.15),
                'shape': (20, 20),
            },
            'cache_dir': f'{td}/exo_cache/',
        }

        te = BaseExoRasterizer(**kwargs)

        te_gen = ExoRasterizer(**{
            k: v for k, v in kwargs.items() if k != 'cache_dir'
        })

        assert np.array_equal(te.data.as_array(), te_gen.data.as_array())

        hr_elev = np.asarray(te.data.as_array())

        lat = te.hr_lat_lon[..., 0].flatten()
        lon = te.hr_lat_lon[..., 1].flatten()
        hr_wtk_meta = np.vstack((lat, lon)).T
        hr_wtk_ind = np.arange(len(lat)).reshape(te.hr_shape[:-1])
        assert te.nn.max() == len(hr_wtk_meta)

        for gid in RANDOM_GENERATOR.choice(
            len(hr_wtk_meta), 50, replace=False
        ):
            idy, idx = np.where(hr_wtk_ind == gid)
            iloc = np.where(te.nn == gid)[0]
            exo_coords = te.source_lat_lon[iloc]

            # make sure all mapped high-res exo coordinates are closest to gid
            # pylint: disable=consider-using-enumerate
            for i in range(len(exo_coords)):
                dist = hr_wtk_meta - exo_coords[i]
                dist = np.hypot(dist[:, 0], dist[:, 1])
                assert np.argmin(dist) == gid

            # make sure the mean elevation makes sense
            test_out = hr_elev[idy, idx, 0]
            true_out = te.source_data[iloc].mean()
            assert np.allclose(test_out, true_out)


def test_bad_s_enhance(s_enhance=10):
    """Test a large s_enhance factor that results in a bad mapping with
    enhanced grid pixels not having source exo data points"""
    with tempfile.TemporaryDirectory() as td:
        fp_exo_topo = make_topo_file(pytest.FP_WTK, td)

        with pytest.warns(UserWarning) as warnings:
            te = BaseExoRasterizer(
                file_paths=pytest.FP_WTK,
                source_files=fp_exo_topo,
                feature='topography',
                s_enhance=s_enhance,
                t_enhance=1,
                input_handler_kwargs={
                    'target': (39.01, -105.15),
                    'shape': (20, 20),
                },
                cache_dir=f'{td}/exo_cache/',
            )
            _ = te.data

    good = [
        'target pixels did not have unique' in str(w.message)
        for w in warnings.list
    ]
    assert any(good)


def test_topo_extraction_nc():
    """Test the spatial enhancement of a test grid and then the lookup of the
    elevation data to a reference WRF file (also the same file for the test)

    We already test proper topo mapping and aggregation in the h5 test so this
    just makes sure that the data can be rasterized from a WRF file.
    """
    with TemporaryDirectory() as td:
        te = BaseExoRasterizer(
            file_paths=pytest.FP_WRF,
            source_files=pytest.FP_WRF,
            feature='topography',
            s_enhance=1,
            t_enhance=1,
            cache_dir=f'{td}/exo_cache/',
        )
        hr_elev = np.asarray(te.data.as_array())

        te_gen = ExoRasterizer(
            file_paths=pytest.FP_WRF,
            source_files=pytest.FP_WRF,
            feature='topography',
            s_enhance=1,
            t_enhance=1,
        )
        assert np.array_equal(te.data.as_array(), te_gen.data.as_array())
    assert np.allclose(te.source_data.flatten(), hr_elev.flatten())


@pytest.mark.parametrize(
    ['s_enhance', 'with_nans'],
    [
        (1, False),
        (2, False),
        (4, False),
        (1, True),
        (2, True),
        (4, True),
    ],
)
def test_obs_agg(s_enhance, with_nans):
    """Test the aggregation of 3D data with local mean"""
    with TemporaryDirectory() as td:
        dset = make_fake_dset(
            (8, 8, 1),
            ['u_10m'],
            lat_range=(20, 24),
            lon_range=(-120, -116),
        )

        if with_nans:
            mask = RANDOM_GENERATOR.choice(
                [True, False], dset['u_10m'].shape, p=[0.6, 0.4]
            )
            u_10m = dset['u_10m'].values
            u_10m[mask] = np.nan
            dset['u_10m'] = (dset['u_10m'].dims, u_10m)

        res = dset.sx.coarsen({'south_north': 4, 'west_east': 4}).mean()
        res.to_netcdf(f'{td}/lr.nc')
        dset.to_netcdf(f'{td}/hr.nc')

        te = ExoRasterizer(
            file_paths=f'{td}/lr.nc',
            source_files=f'{td}/hr.nc',
            feature='u_10m_obs',
            s_enhance=s_enhance,
            t_enhance=1,
        )
        agg_obs = np.asarray(te._get_data_3d())
        true_obs = (
            te
            .source_handler['u_10m']
            .coarsen({
                'south_north': 4 // s_enhance,
                'west_east': 4 // s_enhance,
            })
            .mean()
            .values
        )
        mask = np.isnan(agg_obs) | np.isnan(true_obs)
        assert np.allclose(agg_obs[~mask], true_obs[~mask], rtol=0.01)
