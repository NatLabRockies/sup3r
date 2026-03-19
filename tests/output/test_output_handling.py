"""Output method tests"""

import os
import tempfile

import numpy as np
import pandas as pd
from rex import ResourceX

from sup3r.postprocessing import CollectorH5, CollectorNC
from sup3r.preprocessing import DataHandler, Loader
from sup3r.preprocessing.derivers.utilities import (
    invert_uv,
    transform_rotate_wind,
)
from sup3r.utilities.pytest.helpers import (
    make_collect_chunks,
    make_fake_h5_chunks,
)
from sup3r.utilities.utilities import RANDOM_GENERATOR
from sup3r.writers import OutputHandler, OutputHandlerH5, OutputHandlerNC


def test_get_lat_lon():
    """Check that regridding works correctly"""
    low_res_lats = np.array([[1, 1, 1], [0, 0, 0]])
    low_res_lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [
            np.expand_dims(low_res_lats, axis=-1),
            np.expand_dims(low_res_lons, axis=-1),
        ],
        axis=-1,
    )
    shape = (4, 6)
    new_lat_lon = OutputHandlerNC.get_lat_lon(lat_lon, shape)

    new_lats = np.round(new_lat_lon[..., 0], 2)
    new_lons = np.round(new_lat_lon[..., 1], 2)

    assert np.allclose(new_lats[0, :], 1.25)
    assert np.allclose(new_lats[-1, :], -0.25)

    assert np.allclose(new_lons[:, 0], -125)
    assert np.allclose(new_lons[:, -1], -75)

    lat_lon = lat_lon[::-1]
    new_lat_lon = OutputHandlerNC.get_lat_lon(lat_lon, shape)

    new_lats = np.round(new_lat_lon[..., 0], 2)
    new_lons = np.round(new_lat_lon[..., 1], 2)

    assert np.allclose(new_lats[0, :], -0.25)
    assert np.allclose(new_lats[-1, :], 1.25)

    assert np.allclose(new_lons[:, 0], -125)
    assert np.allclose(new_lons[:, -1], -75)


def test_invert_uv():
    """Make sure inverse uv transform returns inputs"""
    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [np.expand_dims(lats, axis=-1), np.expand_dims(lons, axis=-1)], axis=-1
    )
    windspeed = RANDOM_GENERATOR.random((*lat_lon.shape[:2], 5))
    winddirection = 360 * RANDOM_GENERATOR.random((*lat_lon.shape[:2], 5))

    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(windspeed, ws)
    assert np.allclose(winddirection, wd)

    lat_lon = lat_lon[::-1]
    u, v = transform_rotate_wind(
        np.array(windspeed, dtype=np.float32),
        np.array(winddirection, dtype=np.float32),
        lat_lon,
    )

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(windspeed, ws)
    assert np.allclose(winddirection, wd)


def test_invert_uv_inplace():
    """Make sure inverse uv transform in output handler returns same as direct
    transform"""

    lats = np.array([[1, 1, 1], [0, 0, 0]])
    lons = np.array([[-120, -100, -80], [-120, -100, -80]])
    lat_lon = np.concatenate(
        [np.expand_dims(lats, axis=-1), np.expand_dims(lons, axis=-1)], axis=-1
    )
    u = RANDOM_GENERATOR.random((*lat_lon.shape[:2], 5))
    v = RANDOM_GENERATOR.random((*lat_lon.shape[:2], 5))

    data = np.concatenate(
        [np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)], axis=-1
    )
    OutputHandlerH5.invert_uv_features(data, ['u_100m', 'v_100m'], lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(data[..., 0], ws)
    assert np.allclose(data[..., 1], wd)

    lat_lon = lat_lon[::-1]
    data = np.concatenate(
        [np.expand_dims(u, axis=-1), np.expand_dims(v, axis=-1)], axis=-1
    )
    OutputHandlerH5.invert_uv_features(data, ['u_100m', 'v_100m'], lat_lon)

    ws, wd = invert_uv(u, v, lat_lon)

    assert np.allclose(data[..., 0], ws)
    assert np.allclose(data[..., 1], wd)


def test_general_h5_collect():
    """Make sure general file collection gives complete meta, time_index, and
    data array."""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')

        out = make_collect_chunks(td, ext='h5')
        out_files, data, features, hr_lat_lon, hr_times = (
            out[0],
            out[1],
            out[-3],
            out[-2],
            out[-1],
        )

        CollectorH5.collect(out_files, fp_out, features=features)

        with ResourceX(fp_out) as res:
            lat_lon = res['meta'][['latitude', 'longitude']].values
            time_index = res['time_index'].values
            collect_data = np.dstack([res[f, :, :] for f in features])
            base_data = data.transpose(2, 0, 1, 3).reshape(
                (len(hr_times), -1, len(features))
            )
            base_data = np.around(base_data.astype(np.float32), 2)
            hr_lat_lon = hr_lat_lon.astype(np.float32)
            assert np.array_equal(hr_times, time_index)
            assert np.array_equal(hr_lat_lon.reshape((-1, 2)), lat_lon)
            assert np.array_equal(base_data, collect_data)


def test_general_nc_collect():
    """Make sure general file collection gives complete meta, time_index, and
    data array."""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.nc')

        out = make_collect_chunks(td, ext='nc')
        out_files, base_data, features, hr_lat_lon, hr_times = (
            out[0],
            out[1],
            out[-3],
            out[-2],
            out[-1],
        )

        CollectorNC.collect(out_files, fp_out, features=features)

        with Loader(fp_out) as res:
            assert np.array_equal(hr_times, res.time_index.values)
            assert np.allclose(hr_lat_lon, res.lat_lon)
            assert np.allclose(base_data, res.values)


def test_h5_out_and_collect(collect_check):
    """Test h5 file output writing and collection with dummy data"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')

        out = make_fake_h5_chunks(td)
        out_files, features = out[0], out[4]

        CollectorH5.collect(out_files, fp_out, features=features)

        collect_check(out, fp_out)


def test_h5_collect_mask():
    """Test h5 file collection with mask meta"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')
        fp_out_mask = os.path.join(td, 'out_combined_masked.h5')
        mask_file = os.path.join(td, 'mask.csv')

        out = make_fake_h5_chunks(td)
        (out_files, data, _, _, features, _, _, _, _, _, _) = out

        CollectorH5.collect(out_files, fp_out, features=features)
        indices = np.arange(np.prod(data.shape[:2]))
        indices = indices[slice(-len(indices) // 2, None)]
        removed = [RANDOM_GENERATOR.choice(indices) for _ in range(10)]
        mask_slice = [i for i in indices if i not in removed]
        with ResourceX(fp_out) as fh:
            mask_meta = fh.meta
            mask_meta = mask_meta.iloc[mask_slice].reset_index(drop=True)
            mask_meta['gid'][:] = np.arange(len(mask_meta))
            mask_meta.to_csv(mask_file, index=False)

        CollectorH5.collect(
            out_files,
            fp_out_mask,
            features=features,
            target_meta_file=mask_file,
            max_workers=1,
        )
        with ResourceX(fp_out_mask) as fh:
            mask_meta = pd.read_csv(mask_file, dtype=np.float32)
            assert np.array_equal(mask_meta['gid'], fh.meta.index.values)
            assert np.array_equal(mask_meta['longitude'], fh.meta['longitude'])
            assert np.array_equal(mask_meta['latitude'], fh.meta['latitude'])

            with ResourceX(fp_out) as fh_o:
                assert np.array_equal(
                    fh_o['windspeed_100m', :, mask_slice], fh['windspeed_100m']
                )


def test_enforce_limits():
    """Make sure clearsky ratio is capped to [0, 1] by netcdf OutputHandler."""

    data = RANDOM_GENERATOR.uniform(-100, 100, (10, 10, 10, 1))
    lon, lat = np.meshgrid(np.arange(10), np.arange(10))
    lat_lon = np.dstack([lat, lon])
    times = pd.date_range('2021-01-01', '2021-01-10', 10)
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_csr.nc')
        OutputHandlerNC._write_output(
            data=data,
            features=['clearsky_ratio'],
            lat_lon=lat_lon,
            times=times,
            out_file=fp_out,
        )
        with Loader(fp_out) as res:
            assert res.data['clearsky_ratio'].max() <= 1.0
            assert res.data['clearsky_ratio'].max() >= 0.0


def test_netcdf_uv_invert():
    """Make windspeed and direction are inverted and written correctly to
    netcdf files."""

    data = RANDOM_GENERATOR.uniform(-10, 10, (10, 10, 5, 2))
    lon, lat = np.meshgrid(np.arange(10), np.arange(10)[::-1])
    lat_lon = np.dstack([lat, lon])
    times = pd.date_range('2021-01-01', '2021-01-05', 5)
    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_ws.nc')
        OutputHandlerNC._write_output(
            data=data.copy(),
            features=['u_10m', 'v_10m'],
            lat_lon=lat_lon,
            times=times,
            out_file=fp_out,
            invert_uv=True,
        )
        dh = DataHandler(
            fp_out, features=['windspeed_10m', 'winddirection_10m']
        )
        uvals = dh.derive('u_10m').values
        vvals = dh.derive('v_10m').values
        assert np.allclose(data[..., 0], uvals, atol=1e-5)
        assert np.allclose(data[..., 1], vvals, atol=1e-5)


def test_get_lat_lon_floating_point_error_shape():
    """Test that floating point routing does not affect output array shape"""
    # 122 and 61 known to cause issues with `np.arange(0, 10, 10 / x)`
    x, y = 122, 61
    enhance = 5
    low_res_lat_lon = np.ones((x, y, 2))
    out_shape = (x * enhance, y * enhance)
    hr_lat_lon = OutputHandler.get_lat_lon(low_res_lat_lon, shape=out_shape)
    assert hr_lat_lon.shape == (*out_shape, 2)
