"""Test the :class:`DualRasterizer` objects."""

import os
import tempfile

import numpy as np
import pytest

from sup3r.preprocessing import DataHandler, DualRasterizer, Loader

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']


def test_dual_rasterizer_shapes(full_shape=(20, 20)):
    """Test for consistent lr / hr shapes."""

    # need to reduce the number of temporal examples to test faster
    hr_container = DataHandler(
        file_paths=pytest.FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 10),
    )
    lr_container = DataHandler(
        file_paths=pytest.FP_ERA,
        features=FEATURES,
        time_slice=slice(None, None, 10),
    )

    pair_rasterizer = DualRasterizer(
        {'low_res': lr_container.data, 'high_res': hr_container.data},
        s_enhance=2,
        t_enhance=1,
    )
    assert pair_rasterizer.data.low_res.shape == (
        pair_rasterizer.data.high_res.shape[0] // 2,
        pair_rasterizer.data.high_res.shape[1] // 2,
        *pair_rasterizer.data.high_res.shape[2:],
    )


def test_dual_nan_fill(full_shape=(20, 20)):
    """Test interpolate_na nan fill."""

    # need to reduce the number of temporal examples to test faster
    hr_container = DataHandler(
        file_paths=pytest.FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(0, 5),
    )
    lr_container = DataHandler(
        file_paths=pytest.FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(0, 5),
    )

    assert not np.isnan(lr_container.data.as_array()).any()
    lr_container.data[FEATURES[0]][slice(5, 10), slice(5, 10), 2] = np.nan
    assert np.isnan(lr_container.data.as_array()).any()

    pair_rasterizer = DualRasterizer(
        {'low_res': lr_container.data, 'high_res': hr_container.data},
        s_enhance=1,
        t_enhance=1,
    )

    assert not np.isnan(pair_rasterizer.data.low_res.as_array()).any()


def test_regrid_caching(full_shape=(20, 20)):
    """Test caching and loading of regridded data"""

    # need to reduce the number of temporal examples to test faster
    with tempfile.TemporaryDirectory() as td:
        hr_container = DataHandler(
            file_paths=pytest.FP_WTK,
            features=FEATURES,
            target=TARGET_COORD,
            shape=full_shape,
            time_slice=slice(None, None, 10),
        )
        lr_container = DataHandler(
            file_paths=pytest.FP_ERA,
            features=FEATURES,
            time_slice=slice(None, None, 10),
        )
        lr_cache_pattern = os.path.join(td, 'lr_{feature}.h5')
        hr_cache_pattern = os.path.join(td, 'hr_{feature}.h5')
        pair_rasterizer = DualRasterizer(
            {'low_res': lr_container.data, 'high_res': hr_container.data},
            s_enhance=2,
            t_enhance=1,
            lr_cache_kwargs={'cache_pattern': lr_cache_pattern},
            hr_cache_kwargs={'cache_pattern': hr_cache_pattern},
        )

        # Load handlers again
        lr_container_new = Loader(
            [lr_cache_pattern.format(feature=f) for f in lr_container.features]
        )
        hr_container_new = Loader(
            [hr_cache_pattern.format(feature=f) for f in hr_container.features]
        )

        assert np.array_equal(
            lr_container_new.data[FEATURES][...],
            pair_rasterizer.data.low_res[FEATURES][...],
        )
        assert np.array_equal(
            hr_container_new.data[FEATURES][...],
            pair_rasterizer.data.high_res[FEATURES][...],
        )
