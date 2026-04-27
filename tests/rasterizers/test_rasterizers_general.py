"""Tests across general functionality of :class:`Rasterizer` objects"""

import numpy as np
import pytest
from rex import Resource

from sup3r.preprocessing import Dimension, Rasterizer
from sup3r.utilities.utilities import xr_open_mfdataset

features = ['windspeed_100m', 'winddirection_100m']


def test_get_full_domain_nc():
    """Test data handling without target, shape, or raster_file input"""

    rasterizer = Rasterizer(file_paths=pytest.FP_ERA)
    nc_res = xr_open_mfdataset(pytest.FP_ERA)
    shape = (len(nc_res[Dimension.LATITUDE]), len(nc_res[Dimension.LONGITUDE]))
    target = (
        nc_res[Dimension.LATITUDE].values.min(),
        nc_res[Dimension.LONGITUDE].values.min(),
    )
    assert np.array_equal(
        rasterizer.lat_lon[-1, 0, :],
        (
            rasterizer.loader[Dimension.LATITUDE].min(),
            rasterizer.loader[Dimension.LONGITUDE].min(),
        ),
    )
    dim_order = (Dimension.SOUTH_NORTH, Dimension.WEST_EAST, Dimension.TIME)

    # raise warning about upper case features
    with pytest.warns(match='Received some upper case features'):
        assert np.array_equal(
            rasterizer['U_100m'],
            nc_res['u_100m'].transpose(*dim_order).data.astype(np.float32),
        )
        assert np.array_equal(
            rasterizer['V_100m'],
            nc_res['v_100m'].transpose(*dim_order).data.astype(np.float32),
        )
    assert rasterizer.grid_shape == shape
    assert np.array_equal(rasterizer.target, target)


def test_get_target_nc():
    """Test data handling without target or raster_file input"""
    rasterizer = Rasterizer(file_paths=pytest.FP_ERA, shape=(4, 4))
    nc_res = xr_open_mfdataset(pytest.FP_ERA)
    target = (
        nc_res[Dimension.LATITUDE].values.min(),
        nc_res[Dimension.LONGITUDE].values.min(),
    )
    assert rasterizer.grid_shape == (4, 4)
    assert np.array_equal(rasterizer.target, target)


@pytest.mark.parametrize(
    ['input_files', 'shape', 'target'],
    [
        (pytest.FP_WTK, (20, 20), (39.01, -105.15)),
        (pytest.FP_ERA, (10, 10), (37.25, -107)),
    ],
)
def test_data_extraction(input_files, shape, target):
    """Test extraction of raw features"""
    rasterizer = Rasterizer(file_paths=input_files, target=target, shape=shape)
    assert rasterizer.shape[:3] == (shape[0], shape[1], rasterizer.shape[2])
    assert rasterizer.data.dtype == np.dtype(np.float32)


def test_topography_h5():
    """Test that topography is rasterized correctly"""

    with Resource(pytest.FP_WTK) as res:
        rasterizer = Rasterizer(
            file_paths=pytest.FP_WTK, target=(39.01, -105.15), shape=(20, 20)
        )
        ri = rasterizer.raster_index
        topo = res.get_meta_arr('elevation')[ri.flatten(),]
        topo = topo.reshape((ri.shape[0], ri.shape[1]))
    assert np.allclose(topo, rasterizer['topography'][..., 0])


def test_preloaded_h5():
    """Test preload of h5 file"""
    rasterizer = Rasterizer(
        file_paths=pytest.FP_WTK,
        target=(39.01, -105.15),
        shape=(20, 20),
        chunks=None,
    )
    for f in list(rasterizer.data.data_vars) + list(Dimension.coords_2d()):
        assert isinstance(rasterizer[f].data, np.ndarray)
