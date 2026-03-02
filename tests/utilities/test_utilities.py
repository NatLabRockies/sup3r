"""pytests for general utilities"""

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import interp1d

from sup3r.models.utilities import st_interp
from sup3r.pipeline.utilities import get_chunk_slices
from sup3r.preprocessing.derivers.utilities import transform_rotate_wind
from sup3r.preprocessing.samplers.utilities import (
    uniform_box_sampler,
    uniform_time_sampler,
    weighted_box_sampler,
    weighted_time_sampler,
)
from sup3r.preprocessing.utilities import (
    get_date_range_kwargs,
    make_time_index_from_kws,
)
from sup3r.utilities.interpolation import Interpolator
from sup3r.utilities.utilities import (
    RANDOM_GENERATOR,
    spatial_coarsening,
    temporal_coarsening,
)
from sup3r.writers import OutputHandler


def test_leap_day_handling():
    """Test get_date_range_kwargs utility function"""

    start = '2012-02-17 00:00:00'
    end = '2012-03-25 00:00:00'
    time_index = pd.date_range(start=start, end=end)
    mask = (time_index.month == 2) & (time_index.day == 29)
    time_index = time_index[~mask]
    dr_kwargs = get_date_range_kwargs(time_index)
    assert 'drop_leap' in dr_kwargs

    new_ti = make_time_index_from_kws(dr_kwargs)
    assert all(new_ti == time_index)

    hr_ti = OutputHandler.get_times(new_ti, 24 * len(new_ti))
    mask = (hr_ti.month == 2) & (hr_ti.day == 29)
    assert not any(mask)

    dr_kwargs = get_date_range_kwargs(hr_ti)
    assert 'drop_leap' in dr_kwargs

    new_ti = make_time_index_from_kws(dr_kwargs)
    assert all(new_ti == hr_ti)


def test_log_interp():
    """Make sure log interp generates reasonable output (e.g. between input
    levels)"""
    shape = (3, 3, 5)
    lower = RANDOM_GENERATOR.uniform(-10, 10, shape)
    upper = RANDOM_GENERATOR.uniform(-10, 10, shape)

    hgt_array = da.stack(
        [np.full(shape, 10), np.full(shape, 100)],
        axis=-1,
    )
    u = da.stack([lower, upper], axis=-1)
    out = Interpolator.interp_to_level(
        hgt_array, u, [40], interp_kwargs={'method': 'log'}
    )

    def between_check(first, mid, second):
        return (first <= mid <= second) or (second <= mid <= first)

    u_check = all(
        between_check(lower, mid, higher)
        for lower, mid, higher in zip(
            lower.flatten(),
            out.flatten(),
            upper.flatten(),
        )
    )
    assert u_check


def test_get_chunk_slices():
    """Test get_chunk_slices function for correct start/end"""

    arr = np.arange(10, 100)
    index_slice = slice(20, 80)

    slices = get_chunk_slices(len(arr), chunk_size=10, index_slice=index_slice)

    assert slices[0].start == index_slice.start
    assert slices[-1].stop == index_slice.stop


def test_weighted_box_sampler():
    """Test weighted_box_sampler for correct start point based on weights"""

    data = np.zeros((100, 1, 1))
    shape = (10, 1)
    max_shape = (data.shape[0] - shape[0] + 1) * (data.shape[1] - shape[1] + 1)
    chunks = np.array_split(np.arange(0, max_shape), 10)
    weights = np.zeros(len(chunks))
    weights_1 = weights.copy()
    weights_1[0] = 1

    weights_2 = weights.copy()
    weights_2[-1] = 1

    weights_3 = weights.copy()
    weights_3[2] = 0.5
    weights_3[5] = 0.5

    for _ in range(100):
        slice_1, _ = weighted_box_sampler(data.shape, shape, weights_1)
        assert chunks[0][0] <= slice_1.start <= chunks[0][-1]

        slice_2, _ = weighted_box_sampler(data.shape, shape, weights_2)
        assert chunks[-1][0] <= slice_2.start <= chunks[-1][-1]

        slice_3, _ = weighted_box_sampler(data.shape, shape, weights_3)
        assert (
            chunks[2][0] <= slice_3.start <= chunks[2][-1]
            or chunks[5][0] <= slice_3.start <= chunks[5][-1]
        )

    data = np.zeros((2, 100, 1))
    shape = (2, 10)
    max_shape = (data.shape[0] - shape[0] + 1) * (data.shape[1] - shape[1] + 1)
    chunks = np.array_split(np.arange(0, max_shape), 10)
    weights = np.zeros(len(chunks))
    weights_1 = weights.copy()
    weights_1[0] = 1

    weights_2 = weights.copy()
    weights_2[-1] = 1

    weights_3 = weights.copy()
    weights_3[2] = 0.5
    weights_3[5] = 0.5

    for _ in range(100):
        _, slice_1 = weighted_box_sampler(data.shape, shape, weights_1)
        assert chunks[0][0] <= slice_1.start <= chunks[0][-1]

        _, slice_2 = weighted_box_sampler(data.shape, shape, weights_2)
        assert chunks[-1][0] <= slice_2.start <= chunks[-1][-1]

        _, slice_3 = weighted_box_sampler(data.shape, shape, weights_3)
        assert (
            chunks[2][0] <= slice_3.start <= chunks[2][-1]
            or chunks[5][0] <= slice_3.start <= chunks[5][-1]
        )

    shape = (1, 1)
    weights = np.zeros(np.prod(data.shape))
    weights_4 = weights.copy()
    weights_4[5] = 1

    _, slice_4 = weighted_box_sampler(data.shape, shape, weights_4)
    assert weights_4[slice_4.start] == 1


def test_weighted_time_sampler():
    """Test weighted_time_sampler for correct start point based on weights"""

    data = np.zeros((1, 1, 100))
    shape = 10
    chunks = np.array_split(np.arange(0, 100 - (shape - 1)), 10)
    weights = np.zeros(len(chunks))
    weights_1 = weights.copy()
    weights_1[0] = 1

    weights_2 = weights.copy()
    weights_2[-1] = 1

    weights_3 = weights.copy()
    weights_3[2] = 0.5
    weights_3[5] = 0.5

    for _ in range(100):
        slice_1 = weighted_time_sampler(data.shape, shape, weights_1)
        assert chunks[0][0] <= slice_1.start <= chunks[0][-1]

        slice_2 = weighted_time_sampler(data.shape, shape, weights_2)
        assert chunks[-1][0] <= slice_2.start <= chunks[-1][-1]

        slice_3 = weighted_time_sampler(data.shape, 10, weights_3)
        assert (
            chunks[2][0] <= slice_3.start <= chunks[2][-1]
            or chunks[5][0] <= slice_3.start <= chunks[5][-1]
        )

    shape = 1
    weights = np.zeros(data.shape[2])
    weights_4 = weights.copy()
    weights_4[5] = 1

    slice_4 = weighted_time_sampler(data.shape, shape, weights_4)
    assert weights_4[slice_4.start] == 1


def test_uniform_time_sampler():
    """Test uniform_time_sampler for correct start point for edge case"""

    data = np.zeros((1, 1, 10))
    shape = 10
    t_slice = uniform_time_sampler(data.shape, shape)
    assert t_slice.start == 0
    assert t_slice.stop == data.shape[2]


def test_uniform_box_sampler():
    """Test uniform_box_sampler for correct start point for edge case"""

    data = np.zeros((10, 10, 1))
    shape = (10, 10)
    [s1, s2] = uniform_box_sampler(data.shape, shape)
    assert s1.start == s2.start == 0
    assert s1.stop == data.shape[0]
    assert s2.stop == data.shape[1]


def test_s_coarsen_errors():
    """Negative tests of spatial coarsening method"""

    arr = np.arange(28800).reshape((2, 20, 20, 12, 3))
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=3)
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=7)
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=40)

    arr = np.ones(10)
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=5)

    arr = np.ones((4, 4))
    with pytest.raises(ValueError):
        spatial_coarsening(arr, s_enhance=2)


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_coarsen_5D(s_enhance):
    """Test the spatial enhancement of a 5D array"""
    arr = np.arange(28800).reshape((2, 20, 20, 12, 3))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=True)

    for o in range(arr.shape[0]):
        for t in range(arr.shape[3]):
            for f in range(arr.shape[4]):
                for i_lr in range(coarse.shape[1]):
                    for j_lr in range(coarse.shape[2]):
                        i_hr = i_lr * s_enhance
                        i_hr = slice(i_hr, i_hr + s_enhance)

                        j_hr = j_lr * s_enhance
                        j_hr = slice(j_hr, j_hr + s_enhance)

                        assert np.allclose(
                            coarse[o, i_lr, j_lr, t, f],
                            arr[o, i_hr, j_hr, t, f].mean(),
                        )


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_coarsen_4D(s_enhance):
    """Test the spatial enhancement of a 4D array"""
    arr = np.arange(28800).reshape((24, 20, 20, 3))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=True)

    for o in range(arr.shape[0]):
        for f in range(arr.shape[3]):
            for i_lr in range(coarse.shape[1]):
                for j_lr in range(coarse.shape[2]):
                    i_hr = i_lr * s_enhance
                    i_hr = slice(i_hr, i_hr + s_enhance)

                    j_hr = j_lr * s_enhance
                    j_hr = slice(j_hr, j_hr + s_enhance)

                    assert np.allclose(
                        coarse[o, i_lr, j_lr, f], arr[o, i_hr, j_hr, f].mean()
                    )


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_coarsen_4D_no_obs(s_enhance):
    """Test the spatial enhancement of a 4D array without the obs axis"""
    arr = np.arange(28800).reshape((20, 20, 24, 3))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=False)

    for t in range(arr.shape[2]):
        for f in range(arr.shape[3]):
            for i_lr in range(coarse.shape[0]):
                for j_lr in range(coarse.shape[1]):
                    i_hr = i_lr * s_enhance
                    i_hr = slice(i_hr, i_hr + s_enhance)

                    j_hr = j_lr * s_enhance
                    j_hr = slice(j_hr, j_hr + s_enhance)

                    assert np.allclose(
                        coarse[i_lr, j_lr, t, f], arr[i_hr, j_hr, t, f].mean()
                    )


@pytest.mark.parametrize('s_enhance', [1, 2, 4, 5])
def test_s_coarsen_3D_no_obs(s_enhance):
    """Test the spatial enhancement of a 3D array without the obs axis"""
    arr = np.arange(28800).reshape((20, 20, 72))
    coarse = spatial_coarsening(arr, s_enhance=s_enhance, obs_axis=False)

    for f in range(arr.shape[2]):
        for i_lr in range(coarse.shape[0]):
            for j_lr in range(coarse.shape[1]):
                i_hr = i_lr * s_enhance
                i_hr = slice(i_hr, i_hr + s_enhance)

                j_hr = j_lr * s_enhance
                j_hr = slice(j_hr, j_hr + s_enhance)

                assert np.allclose(
                    coarse[i_lr, j_lr, f], arr[i_hr, j_hr, f].mean()
                )


def test_t_coarsen():
    """Test temporal coarsening of 5D array"""
    t_enhance = 4
    hr_shape = (3, 10, 10, 48, 2)
    arr = np.arange(np.prod(hr_shape)).reshape(hr_shape).astype(float)

    # test 4x temporal enhancement averaging
    arr_lr = temporal_coarsening(arr, t_enhance=t_enhance, method='average')
    for i, idt0 in enumerate(range(0, arr.shape[3], t_enhance)):
        truth = arr[:, :, :, slice(idt0, idt0 + t_enhance), :].mean(axis=3)
        test = arr_lr[:, :, :, i, :]
        assert np.allclose(truth, test)

    # test 4x temporal enhancement total
    arr_lr = temporal_coarsening(arr, t_enhance=t_enhance, method='total')
    for i, idt0 in enumerate(range(0, arr.shape[3], t_enhance)):
        truth = arr[:, :, :, slice(idt0, idt0 + t_enhance), :].sum(axis=3)
        test = arr_lr[:, :, :, i, :]
        assert np.allclose(truth, test)

    # test 4x temporal enhancement subsampling
    arr_lr = temporal_coarsening(arr, t_enhance=t_enhance, method='subsample')
    for i, idt0 in enumerate(range(0, arr.shape[3], t_enhance)):
        truth = arr[:, :, :, idt0, :]
        test = arr_lr[:, :, :, i, :]
        assert np.allclose(truth, test)


def test_transform_rotate():
    """Make sure inverse uv transform returns inputs"""
    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [np.expand_dims(lats, axis=-1), np.expand_dims(lons, axis=-1)], axis=-1
    )
    windspeed = np.ones((lat_lon.shape[0], lat_lon.shape[1], 1))

    # wd = 0 -> u = 0 and v = -1
    winddirection = np.zeros((lat_lon.shape[0], lat_lon.shape[1], 1))

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )
    u_target = np.zeros(u.shape)
    u_target[...] = 0
    v_target = np.zeros(v.shape)
    v_target[...] = -1

    assert np.allclose(u, u_target, atol=1e-5)
    assert np.allclose(v, v_target, atol=1e-5)

    # wd = 90 -> u = -1 and v = 0
    winddirection = np.zeros((lat_lon.shape[0], lat_lon.shape[1], 1))
    winddirection[...] = 90

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )
    u_target = np.zeros(u.shape)
    u_target[...] = -1
    v_target = np.zeros(v.shape)
    v_target[...] = 0

    assert np.allclose(u, u_target, atol=1e-5)
    assert np.allclose(v, v_target, atol=1e-5)

    # wd = 270 -> u = 1 and v = 0
    winddirection = np.zeros((lat_lon.shape[0], lat_lon.shape[1], 1))
    winddirection[...] = 270

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )
    u_target = np.zeros(u.shape)
    u_target[...] = 1
    v_target = np.zeros(v.shape)
    v_target[...] = 0

    assert np.allclose(u, u_target, atol=1e-5)
    assert np.allclose(v, v_target, atol=1e-5)

    # wd = 180 -> u = 0 and v = 1
    winddirection = np.zeros((lat_lon.shape[0], lat_lon.shape[1], 1))
    winddirection[...] = 180

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )
    u_target = np.zeros(u.shape)
    u_target[...] = 0
    v_target = np.zeros(v.shape)
    v_target[...] = 1

    assert np.allclose(u, u_target, atol=1e-5)
    assert np.allclose(v, v_target, atol=1e-5)

    # wd = 45 -> u = -1/sqrt(2) and v = -1/sqrt(2)
    winddirection = np.zeros((lat_lon.shape[0], lat_lon.shape[1], 1))
    winddirection[...] = 45

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )
    u_target = np.zeros(u.shape)
    u_target[...] = -1 / np.sqrt(2)
    v_target = np.zeros(v.shape)
    v_target[...] = -1 / np.sqrt(2)

    assert np.allclose(u, u_target, atol=1e-5)
    assert np.allclose(v, v_target, atol=1e-5)


def test_st_interpolation(plot=False):
    """Test spatiotemporal linear interpolation"""

    X, Y, T = np.meshgrid(np.arange(10), np.arange(10), np.arange(1, 11))
    arr = 100 * np.exp(-((X - 5) ** 2 + (Y - 5) ** 2) / T)

    s_interp = st_interp(arr, s_enhance=3, t_enhance=1)

    if plot:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        a = ax1.imshow(np.mean(s_interp, axis=-1))
        b = ax2.imshow(np.mean(arr, axis=-1))
        fig.suptitle('Spatial Interpolation')
        ax1.set_title('Interpolated')
        ax2.set_title('Not Interpolated')
        fig.colorbar(a, ax=ax1, shrink=0.5)
        fig.colorbar(b, ax=ax2, shrink=0.5)
        plt.show()

    err = np.mean(s_interp) - np.mean(arr)
    err = np.abs(err / np.mean(arr))
    assert err < 0.15

    t_interp = st_interp(arr, s_enhance=1, t_enhance=4, t_centered=True)

    if plot:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        a = ax1.imshow(np.mean(t_interp, axis=-1))
        b = ax2.imshow(np.mean(arr, axis=-1))
        fig.suptitle('Temporal Interpolation')
        ax1.set_title('Interpolated')
        ax2.set_title('Not Interpolated')
        fig.colorbar(a, ax=ax1, shrink=0.5)
        fig.colorbar(b, ax=ax2, shrink=0.5)
        plt.show()

    err = np.mean(t_interp) - np.mean(arr)
    err = np.abs(err / np.mean(arr))
    assert err < 0.01

    # spatial test
    s_vals = RANDOM_GENERATOR.uniform(0, 100, 3)
    lr = np.transpose(np.array([[s_vals, s_vals]]), axes=(1, 2, 0))
    lr = np.repeat(lr, 2, axis=-1)
    hr = st_interp(lr, s_enhance=2, t_enhance=1)
    x = np.linspace(-(1 / 4), 2 + (1 / 4), 6)
    ifun = interp1d(np.arange(3), lr[0, :, 0], fill_value='extrapolate')
    truth = ifun(x)
    assert np.allclose(lr[0], lr[1])
    assert not np.allclose(lr[:, 0], lr[:, 1])
    assert np.allclose(hr[0, :, 0], truth)

    # temporal test
    t_vals = RANDOM_GENERATOR.uniform(0, 100, 3)
    lr = np.ones((2, 2, 3)) * t_vals
    hr = st_interp(lr, s_enhance=1, t_enhance=3, t_centered=True)
    x = np.linspace(-(1 / 3), 2 + (1 / 3), 9)
    ifun = interp1d(np.arange(3), lr[0, 0], fill_value='extrapolate')
    truth = ifun(x)
    assert np.allclose(hr[0, 0], truth)

    # check vs. lat/lon interpolation
    lats = np.array([-30, 0, 30])
    lons = np.array([-30, 0, 30])
    lons, lats = np.meshgrid(lons, lats)
    lr_lat_lon = np.dstack((lats, lons))
    lats = np.repeat(np.expand_dims(lats, -1), 2, axis=-1)
    lons = np.repeat(np.expand_dims(lons, -1), 2, axis=-1)

    hr_lats_0 = st_interp(lats, s_enhance=2, t_enhance=1)[..., 0]
    hr_lons_0 = st_interp(lons, s_enhance=2, t_enhance=1)[..., 0]

    hr_lat_lon_1 = OutputHandler.get_lat_lon(lr_lat_lon, (6, 6))
    hr_lats_1 = hr_lat_lon_1[..., 0]
    hr_lons_1 = hr_lat_lon_1[..., 1]

    assert np.allclose(hr_lats_0, hr_lats_1)
    assert np.allclose(hr_lons_0, hr_lons_1)


def test_st_interp_floating_point_error_shape():
    """Test that floating point routing does not affect output array shape"""
    # 122 and 61 known to cause issues with `np.arange(0, 10, 10 / x)`
    x, y, t = 122, 61, 14
    s_enhance, t_enhance = 5, 11
    low_res = np.ones((x, y, t))
    high_res = st_interp(low_res, s_enhance=s_enhance, t_enhance=t_enhance)
    assert high_res.shape == (x * s_enhance, y * s_enhance, t * t_enhance)
