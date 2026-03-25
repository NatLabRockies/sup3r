"""pytests for :class:`Loader` objects"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rex import Resource

from sup3r.preprocessing import Dimension, Loader, LoaderH5, LoaderX
from sup3r.preprocessing.loaders.utilities import standardize_values
from sup3r.utilities.pytest.helpers import (
    make_fake_dset,
    make_fake_nc_file,
)

features = ['windspeed_100m', 'winddirection_100m']


def test_load_zarr():
    """Test simple zarr file loading. Make sure general loader matches zarr
    specific loader (which uses LoaderNC with zarr engine)"""

    with TemporaryDirectory() as td:
        temp_file = os.path.join(td, 'test.zarr')
        nc = make_fake_dset((10, 10, 20), features=['u_100m', 'v_100m'])
        encoding = {}
        for var in list(nc.data_vars) + list(nc.coords):
            if nc[var].ndim > 0:  # Only chunk variables with dimensions
                chunks = tuple(min(5, s) for s in nc[var].shape)
                encoding[var] = {'chunks': chunks}
        nc.to_zarr(temp_file, mode='w', encoding=encoding)
        chunks = {'time': 5, 'south_north': 5, 'west_east': 5}
        loader = LoaderX(
            temp_file, chunks=chunks, res_kwargs={'engine': 'zarr'}
        )
        assert loader.shape == (10, 10, 20, 2)
        assert all(
            loader[f].data.chunksize == tuple(chunks.values())
            for f in loader.features
        )

        gen_loader = Loader(
            temp_file, chunks=chunks, res_kwargs={'engine': 'zarr'}
        )

        assert np.array_equal(loader.as_array(), gen_loader.as_array())


def test_time_independent_loading():
    """Make sure loaders work with time independent files."""
    with TemporaryDirectory() as td:
        out_file = os.path.join(td, 'topo.nc')
        nc = make_fake_dset((20, 20, 1), features=['topography'])
        nc = nc.isel(time=0)
        nc = nc.drop_vars(Dimension.TIME)
        assert Dimension.TIME not in nc.dims
        assert Dimension.TIME not in nc.coords
        nc.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf')
        loader = LoaderX(out_file)
        assert tuple(loader.dims) == (
            Dimension.SOUTH_NORTH,
            Dimension.WEST_EAST,
        )


def test_time_independent_loading_h5():
    """Make sure loaders work with time independent features."""
    loader = LoaderH5(pytest.FP_WTK, features=['topography'])
    assert len(loader['topography'].shape) == 2


def test_dim_ordering():
    """Make sure standard reordering works with dimensions not in the standard
    list."""
    loader = LoaderX(pytest.FPS_GCM)
    assert tuple(loader.to_dataarray().dims) == (
        Dimension.SOUTH_NORTH,
        Dimension.WEST_EAST,
        Dimension.TIME,
        Dimension.PRESSURE_LEVEL,
        'nbnd',
        Dimension.VARIABLE,
    )


def test_standard_values():
    """Make sure standardization of values works."""
    with TemporaryDirectory() as td:
        tmp_file = os.path.join(td, 'ta.nc')
        nc = make_fake_dset((10, 10, 10), features=['ta'])
        nc['ta'].attrs['units'] = 'K'
        old_vals = standardize_values(nc)['ta'].values
        nc.to_netcdf(tmp_file, format='NETCDF4', engine='h5netcdf')
        loader = Loader(tmp_file)
        assert loader.data['ta'].attrs['units'] == 'C'
        ta_vals = loader.data['ta'].transpose(*nc.dims).values
        assert np.allclose(ta_vals, old_vals, rtol=0.01)


def test_lat_inversion():
    """Write temp file with ascending lats and load. Needs to be corrected to
    descending lats."""
    with TemporaryDirectory() as td:
        nc = make_fake_dset(
            (20, 20, 100, 5), features=['u', 'v'], lat_range=(-70, 70)
        )
        out_file = os.path.join(td, 'inverted.nc')
        nc.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf')
        loader = LoaderX(out_file)
        assert nc[Dimension.LATITUDE][0, 0] < nc[Dimension.LATITUDE][-1, 0]
        assert loader.lat_lon[-1, 0, 0] < loader.lat_lon[0, 0, 0]

        assert np.allclose(
            nc['u']
            .transpose(
                Dimension.SOUTH_NORTH,
                Dimension.WEST_EAST,
                Dimension.TIME,
                Dimension.PRESSURE_LEVEL,
            )
            .data[::-1],
            loader['u'],
            rtol=0.01,
        )


def test_lon_range():
    """Write temp file with lons 0 - 360 and load. Needs to be corrected to
    -180 - 180."""
    with TemporaryDirectory() as td:
        nc = make_fake_dset((20, 20, 100, 5), features=['u', 'v'])
        nc[Dimension.LONGITUDE] = (
            nc[Dimension.LONGITUDE].dims,
            (nc[Dimension.LONGITUDE].data + 360) % 360.0,
        )
        out_file = os.path.join(td, 'bad_lons.nc')
        nc.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf')
        loader = LoaderX(out_file)
        assert (nc[Dimension.LONGITUDE] > 180).any()
        assert (loader[Dimension.LONGITUDE] <= 180).all()
        assert (loader[Dimension.LONGITUDE] >= -180).all()


def test_level_inversion():
    """Write temp file with descending pressure levels and load. Needs to be
    corrected so surface pressure is first."""
    with TemporaryDirectory() as td:
        nc = make_fake_dset((20, 20, 100, 5), features=['u', 'v'])
        nc[Dimension.PRESSURE_LEVEL] = (
            nc[Dimension.PRESSURE_LEVEL].dims,
            nc[Dimension.PRESSURE_LEVEL].data[::-1],
        )
        nc['u'] = (
            nc['u'].dims,
            nc['u']
            .isel({Dimension.PRESSURE_LEVEL: slice(None, None, -1)})
            .data,
        )
        out_file = os.path.join(td, 'inverted.nc')
        nc.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf')
        loader = LoaderX(out_file, res_kwargs={'chunks': None})
        assert (
            nc[Dimension.PRESSURE_LEVEL][0] < nc[Dimension.PRESSURE_LEVEL][-1]
        )

        og = nc['u'].transpose(*Dimension.dims_4d_pres()).values[..., ::-1]
        corrected = loader['u'].values
        assert np.allclose(og, corrected, rtol=0.01)


def test_load_cc():
    """Test simple era5 file loading."""
    chunks = {'south_north': 5, 'west_east': 5, 'time': 5}
    loader = LoaderX(pytest.FP_UAS, chunks=chunks)
    assert all(
        loader[f].data.chunksize == tuple(chunks.values())
        for f in loader.features
        if len(loader[f].data.shape) == 3
    )
    assert isinstance(loader.time_index, pd.DatetimeIndex)
    assert loader.to_dataarray().dims[:3] == (
        Dimension.SOUTH_NORTH,
        Dimension.WEST_EAST,
        Dimension.TIME,
    )


@pytest.mark.parametrize('fp', (pytest.FP_ERA, Path(pytest.FP_ERA)))
def test_load_era5(fp):
    """Test simple era5 file loading. Make sure general loader matches the type
    specific loader and that it works with pathlib"""
    chunks = {'south_north': 10, 'west_east': 10, 'time': 1000}
    loader = LoaderX(fp, chunks=chunks)
    assert all(
        loader[f].data.chunksize == tuple(chunks.values())
        for f in loader.features
        if len(loader[f].data.shape) == 3
    )
    assert isinstance(loader.time_index, pd.DatetimeIndex)
    assert loader.to_dataarray().dims[:3] == (
        Dimension.SOUTH_NORTH,
        Dimension.WEST_EAST,
        Dimension.TIME,
    )


def test_load_flattened_nc():
    """Test simple netcdf file loading when nc data is spatially flattened."""
    with TemporaryDirectory() as td:
        temp_file = os.path.join(td, 'test.nc')
        coords = {
            'time': np.array(range(5)),
            'latitude': ('space_dummy', np.array(range(100))),
            'longitude': ('space_dummy', np.array(range(100))),
        }
        data_vars = {
            'u_100m': (('time', 'space_dummy'), np.zeros((5, 100))),
            'v_100m': (('time', 'space_dummy'), np.zeros((5, 100))),
        }
        nc = xr.Dataset(coords=coords, data_vars=data_vars)
        nc.to_netcdf(temp_file)
        chunks = {'time': 5, 'space': 5}
        loader = LoaderX(temp_file, chunks=chunks)
        assert loader.shape == (100, 5, 2)
        assert 'space' in loader['latitude'].dims
        assert 'space' in loader['longitude'].dims
        assert all(
            loader[f].data.chunksize == tuple(chunks.values())
            for f in loader.features
        )

        gen_loader = Loader(temp_file, chunks=chunks)

        assert np.array_equal(loader.as_array(), gen_loader.as_array())


def test_load_nc():
    """Test simple netcdf file loading. Make sure general loader matches nc
    specific loader"""
    with TemporaryDirectory() as td:
        temp_file = os.path.join(td, 'test.nc')
        make_fake_nc_file(
            temp_file, shape=(10, 10, 20), features=['u_100m', 'v_100m']
        )
        chunks = {'time': 5, 'south_north': 5, 'west_east': 5}
        loader = LoaderX(temp_file, chunks=chunks)
        assert loader.shape == (10, 10, 20, 2)
        assert all(
            loader[f].data.chunksize == tuple(chunks.values())
            for f in loader.features
        )

        gen_loader = Loader(temp_file, chunks=chunks)

        assert np.array_equal(loader.as_array(), gen_loader.as_array())


def test_load_h5():
    """Test simple h5 file loading. Also checks renaming elevation ->
    topography. Also makes sure that general loader matches type specific
    loader. Also checks that meta data is carried into loader object"""

    chunks = {'space': 200, 'time': 200}
    loader = LoaderH5(pytest.FP_WTK, chunks=chunks)
    feats = [
        'pressure_100m',
        'temperature_100m',
        'winddirection_100m',
        'winddirection_80m',
        'windspeed_100m',
        'windspeed_80m',
        'topography',
    ]
    assert loader.data.shape == (400, 8784, len(feats))
    assert sorted(loader.features) == sorted(feats)
    assert all(
        loader[f].data.chunksize == tuple(chunks.values()) for f in feats[:-1]
    )
    gen_loader = Loader(pytest.FP_WTK, chunks=chunks)
    assert np.array_equal(loader.as_array(), gen_loader.as_array())
    loader_attrs = {f: loader[f].attrs for f in feats}
    resource_attrs = Resource(pytest.FP_WTK).attrs
    assert np.array_equal(loader.meta, loader._res.meta)
    matching_feats = set(Resource(pytest.FP_WTK).datasets).intersection(feats)
    assert all(loader_attrs[f] == resource_attrs[f] for f in matching_feats)


def test_multi_file_load_nc():
    """Test multi file loading with all features the same shape."""
    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file, shape=(10, 10, 20), features=['u_100m', 'v_100m']
        )
        press_file = os.path.join(td, 'press.nc')
        make_fake_nc_file(
            press_file,
            shape=(10, 10, 20),
            features=['pressure_0m', 'pressure_100m'],
        )
        loader = LoaderX([wind_file, press_file])
        assert loader.shape == (10, 10, 20, 4)


def test_5d_load_nc():
    """Test loading netcdf data with some multi level features. This also
    check renaming of orog -> topography"""
    with TemporaryDirectory() as td:
        wind_file = os.path.join(td, 'wind.nc')
        make_fake_nc_file(
            wind_file,
            shape=(10, 10, 20),
            features=['orog', 'u_100m', 'v_100m'],
        )
        level_file = os.path.join(td, 'wind_levs.nc')
        make_fake_nc_file(
            level_file, shape=(10, 10, 20, 3), features=['zg', 'u']
        )
        loader = LoaderX([wind_file, level_file])

        assert loader.shape == (10, 10, 20, 3, 5)
        assert sorted(loader.features) == sorted([
            'topography',
            'u_100m',
            'v_100m',
            'zg',
            'u',
        ])
        assert loader['u_100m'].shape == (10, 10, 20)
        assert loader['u'].shape == (10, 10, 20, 3)
        assert loader[['u', 'topography']].shape == (10, 10, 20, 3, 2)
        assert loader.data.dtype == np.float32
