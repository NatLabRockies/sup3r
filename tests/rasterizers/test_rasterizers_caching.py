"""Ensure correct functions of :class:`Cacher` objects"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.preprocessing import Loader, Rasterizer
from sup3r.writers import Cacher

target = (39.01, -105.15)
shape = (20, 20)
features = ['windspeed_100m', 'winddirection_100m']


def test_raster_index_caching():
    """Test raster index caching by saving file and then loading"""

    # saving raster file
    with tempfile.TemporaryDirectory() as td:
        raster_file = os.path.join(td, 'raster.txt')
        rasterizer = Rasterizer(
            pytest.FP_WTK, raster_file=raster_file, target=target, shape=shape
        )
        # loading raster file
        rasterizer = Rasterizer(pytest.FP_WTK, raster_file=raster_file)
    assert np.allclose(rasterizer.target, target, atol=1)
    assert rasterizer.shape[:3] == (shape[0], shape[1], rasterizer.shape[2])


@pytest.mark.parametrize(
    ['input_files', 'ext', 'shape', 'target', 'features', 'attrs'],
    [
        (
            pytest.FP_WTK,
            'h5',
            (20, 20),
            (39.01, -105.15),
            ['windspeed_100m', 'winddirection_100m'],
            {'windspeed_100m': {'scale_factor': 10}},
        ),
        (
            pytest.FP_ERA,
            'nc',
            (10, 10),
            (37.25, -107),
            ['u_100m', 'v_100m'],
            {'source': 'ERA5', 'u_100m': {'scale_factor': 10}},
        ),
    ],
)
def test_data_caching(input_files, ext, shape, target, features, attrs):
    """Test data extraction with caching/loading"""

    with tempfile.TemporaryDirectory() as td:
        cache_pattern = os.path.join(td, 'cached_{feature}.' + ext)
        rasterizer = Rasterizer(input_files, shape=shape, target=target)
        cacher = Cacher(
            rasterizer,
            cache_kwargs={'cache_pattern': cache_pattern, 'attrs': attrs},
        )

        good_shape = (shape[0], shape[1], rasterizer.shape[2])
        assert rasterizer.shape[:3] == good_shape
        assert rasterizer.data.dtype == np.dtype(np.float32)
        loader = Loader(cacher.out_files)
        ldata = loader.data[features][...].compute()
        rdata = rasterizer.data[features][...].compute()
        assert np.allclose(ldata, rdata)

        # make sure full domain can be loaded with rasterizers
        rasterizer = Rasterizer(cacher.out_files)
        rdata = rasterizer.data[features][...].compute()
        assert np.allclose(ldata, rdata)
