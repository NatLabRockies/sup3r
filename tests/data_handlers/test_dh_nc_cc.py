"""Test data handler for netcdf climate change data"""

import os
import tempfile
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from rex import Resource
from scipy.spatial import KDTree

from sup3r import TEST_DATA_DIR
from sup3r.preprocessing import (
    DataHandlerNCforCC,
    DataHandlerNCforCCwithPowerLaw,
    Dimension,
    Loader,
    LoaderX,
)
from sup3r.preprocessing.derivers.methods import UWindPowerLaw, VWindPowerLaw
from sup3r.utilities.pytest.helpers import make_fake_dset
from sup3r.utilities.utilities import xr_open_mfdataset


def test_get_just_coords_nc():
    """Test data handling without features, target, shape, or raster_file
    input"""

    handler = DataHandlerNCforCC(file_paths=pytest.FP_UAS, features=[])
    nc_res = LoaderX(pytest.FP_UAS)
    shape = (len(nc_res[Dimension.LATITUDE]), len(nc_res[Dimension.LONGITUDE]))
    target = (
        nc_res[Dimension.LATITUDE].min(),
        nc_res[Dimension.LONGITUDE].min(),
    )
    assert np.array_equal(
        np.asarray(handler.lat_lon[-1, 0, :]),
        (
            handler.rasterizer.data[Dimension.LATITUDE].min(),
            handler.rasterizer.data[Dimension.LONGITUDE].min(),
        ),
    )
    assert not handler.features
    assert handler.grid_shape == shape
    assert np.array_equal(handler.target, target)


def test_reload_cache():
    """Test auto reloading of cached data."""

    with xr_open_mfdataset(pytest.FPS_GCM) as fh:
        min_lat = np.min(fh.lat.values.astype(np.float32))
        min_lon = np.min(fh.lon.values.astype(np.float32))
        target = (min_lat, min_lon)

    features = ['u_100m', 'v_100m']
    with tempfile.TemporaryDirectory() as td:
        dummy_file = os.path.join(td, 'dummy.nc')
        dummy = make_fake_dset((20, 20, 20), features=['dummy'])
        loader = Loader(pytest.FPS_GCM)
        loader.data['dummy'] = dummy['dummy'].values
        out = loader.data[['dummy']]
        out.to_netcdf(dummy_file, format='NETCDF4', engine='h5netcdf')
        cache_pattern = os.path.join(td, 'cache_{feature}.nc')
        cache_kwargs = {'cache_pattern': cache_pattern}
        handler = DataHandlerNCforCC(
            pytest.FPS_GCM,
            features=features,
            target=target,
            shape=(20, 20),
            cache_kwargs=cache_kwargs
        )

        # reload from cache
        cached = DataHandlerNCforCC(
            file_paths=dummy_file,
            features=features,
            target=target,
            shape=(20, 20),
            cache_kwargs=cache_kwargs,
        )
        assert all(f in cached for f in features)
        harr = handler.as_array().compute()
        carr = cached.as_array().compute()
        assert np.array_equal(harr, carr)


@pytest.mark.parametrize(
    ('features', 'feat_class', 'src_name'),
    [(['u_100m'], UWindPowerLaw, 'uas'), (['v_100m'], VWindPowerLaw, 'vas')],
)
def test_data_handling_nc_cc_power_law(features, feat_class, src_name):
    """Make sure the power law extrapolation of wind operates correctly"""

    with tempfile.TemporaryDirectory() as td, xr_open_mfdataset(
        pytest.FP_UAS
    ) as fh:
        tmp_file = os.path.join(td, f'{src_name}.nc')
        if src_name not in fh:
            fh[src_name] = fh['uas']
        fh.to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')

        scalar = (100 / feat_class.NEAR_SFC_HEIGHT) ** feat_class.ALPHA
        var_hh = fh[src_name].values * scalar
        var_hh = np.transpose(var_hh, axes=(1, 2, 0))
        dh = DataHandlerNCforCCwithPowerLaw(tmp_file, features=features)
        if fh['lat'][-1] > fh['lat'][0]:
            var_hh = var_hh[::-1]
        mask = np.isnan(dh.data[features[0]].data)
        masked_u = dh.data[features[0]].data[~mask].compute_chunk_sizes()
        np.array_equal(masked_u, var_hh[~mask])


def test_data_handling_nc_cc():
    """Make sure the netcdf cc data handler operates correctly"""

    with xr_open_mfdataset(pytest.FPS_GCM) as fh:
        min_lat = np.min(fh.lat.values.astype(np.float32))
        min_lon = np.min(fh.lon.values.astype(np.float32))
        target = (min_lat, min_lon)
        plevel = fh.plev[-1]
        ua = np.transpose(fh['ua'][:, -1, ...].values, (1, 2, 0))
        va = np.transpose(fh['va'][:, -1, ...].values, (1, 2, 0))

    handler = DataHandlerNCforCC(
        pytest.FPS_GCM,
        features=['u_100m', 'v_100m', 'latitude_feature', 'longitude_feature'],
        target=target,
        shape=(20, 20),
    )
    assert handler.data.shape == (20, 20, 20, 4)

    # upper case features warning
    features = [f'U_{int(plevel)}pa', f'V_{int(plevel)}pa']
    with pytest.warns(match='Received some upper case features'):
        handler = DataHandlerNCforCC(
            pytest.FPS_GCM,
            features=features,
            target=target,
            shape=(20, 20),
        )

    assert handler.data.shape == (20, 20, 20, 2)
    assert np.allclose(ua[::-1], handler.data[features[0]])
    assert np.allclose(va[::-1], handler.data[features[1]])


def test_nc_cc_temp():
    """Make sure the netcdf cc data handler operates correctly on temperature
    derivations, including unit conversions."""

    with TemporaryDirectory() as td:
        tmp_file = os.path.join(td, 'tas.nc')
        nc = make_fake_dset((10, 10, 10), features=['tas', 'tasmin', 'tasmax'])
        for f in nc.data_vars:
            nc[f].attrs['units'] = 'K'
        nc.to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')
        dh = DataHandlerNCforCC(
            tmp_file,
            features=[
                'temperature_2m',
                'temperature_min_2m',
                'temperature_max_2m',
            ],
        )
        for f in dh.features:
            assert dh[f].attrs['units'] == 'C'

        tmp_file = os.path.join(td, 'ta.nc')
        nc = make_fake_dset((10, 10, 10, 10), features=['ta'])
        nc['ta'].attrs['units'] = 'K'
        nc = nc.swap_dims({'level': 'height'})
        nc.to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')

        DataHandlerNCforCC.FEATURE_REGISTRY.update({'temperature': 'ta'})
        dh = DataHandlerNCforCC(
            tmp_file, features=['temperature_100m']
        )
        assert dh['temperature_100m'].attrs['units'] == 'C'
        nc.close()


def test_nc_cc_rh():
    """Make sure the netcdf cc data handler operates correctly on
    relativehumidity_2m lookup"""

    features = [
        'relativehumidity_2m',
        'relativehumidity_min_2m',
        'relativehumidity_max_2m',
    ]
    with TemporaryDirectory() as td:
        tmp_file = os.path.join(td, 'hurs.nc')
        nc = make_fake_dset(
            (10, 10, 10), features=['hurs', 'hursmin', 'hursmax']
        )
        nc.to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')
        dh = DataHandlerNCforCC(tmp_file, features=features)
        assert all(f in dh.features for f in features)


@pytest.mark.parametrize('agg', (1, 4))
def test_solar_cc(agg):
    """Test solar data handling from CC data file with clearsky ratio
    calculated using clearsky ratio from NSRDB h5 file."""

    features = ['clearsky_ratio', 'rsds', 'clearsky_ghi']
    input_files = [os.path.join(TEST_DATA_DIR, 'rsds_test.nc')]
    nsrdb_source_fp = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')

    with xr_open_mfdataset(input_files) as fh:
        min_lat = np.min(fh.lat.values.astype(np.float32))
        min_lon = np.min(fh.lon.values.astype(np.float32)) - 360
        target = (min_lat, min_lon)
        shape = (len(fh.lat.values), len(fh.lon.values))

    with pytest.raises(AssertionError):
        handler = DataHandlerNCforCC(
            input_files, features=features, target=target, shape=shape
        )

    handler = DataHandlerNCforCC(
        input_files,
        features=features,
        nsrdb_source_fp=nsrdb_source_fp,
        nsrdb_agg=agg,
        target=target,
        shape=shape,
        time_slice=slice(0, 1),
        scale_clearsky_ghi=True,
    )

    cs_ratio = handler.data['clearsky_ratio']
    ghi = handler.data['rsds']
    cs_ghi = handler.data['clearsky_ghi']
    cs_ratio_truth = ghi / cs_ghi

    assert cs_ratio.max() <= 1
    assert cs_ratio.min() >= 0
    assert np.allclose(cs_ratio, cs_ratio_truth)

    with Resource(nsrdb_source_fp) as res:
        meta = res.meta
        tree = KDTree(meta[[Dimension.LATITUDE, Dimension.LONGITUDE]])
        cs_ghi_true = res['clearsky_ghi']

    # check a few sites against NSRDB source file
    for i in range(4):
        for j in range(4):
            test_coord = handler.lat_lon[i, j]
            _, inn = tree.query(test_coord, k=agg)
            true = cs_ghi_true[0:48, inn].mean()
            scaled_true = true * handler._cs_ghi_scale[i, j]
            assert np.allclose(scaled_true, cs_ghi[i, j])
