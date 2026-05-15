"""pytests for H5 climate change data batch handlers"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from sup3r.preprocessing import (
    BatchHandlerCC,
    DataHandlerH5SolarCC,
    DataHandlerH5WindCC,
)
from sup3r.preprocessing.utilities import numpy_if_tensor
from sup3r.utilities.pytest.helpers import BatchHandlerTesterCC

SHAPE = (20, 20)
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)
FEATURES_W = ['u_100m', 'v_100m', 'temperature_100m']
TARGET_W = (39.01, -105.15)
TARGET_SURF = (39.1, -105.4)

dh_kwargs = {
    'target': TARGET_S,
    'shape': SHAPE,
    'time_slice': slice(None, None, 2),
    'time_roll': -7,
}


@pytest.mark.parametrize(
    ('hr_tsteps', 't_enhance', 'features'),
    [
        (72, 24, ['clearsky_ratio']),
        (24, 8, ['clearsky_ratio']),
        (12, 3, ['clearsky_ratio']),
        (72, 24, FEATURES_S),
        (72, 8, FEATURES_S),
        (24, 8, FEATURES_S),
        (33, 3, FEATURES_S),
    ],
)
def test_solar_batching(hr_tsteps, t_enhance, features):
    """Test batching of nsrdb data with and without down sampling to day
    hours"""
    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB,
        features=features,
        nan_method_kwargs={'method': 'nearest', 'dim': 'time'},
        **dh_kwargs,
    )
    batcher = BatchHandlerTesterCC(
        [handler],
        val_containers=[],
        batch_size=1,
        n_batches=5,
        s_enhance=1,
        queue_cap=0,
        t_enhance=t_enhance,
        means=dict.fromkeys(features, 0),
        stds=dict.fromkeys(features, 1),
        sample_shape=(20, 20, hr_tsteps),
    )

    assert not np.isnan(handler.data.hourly[...]).all()
    assert not np.isnan(handler.data.daily[...]).any()
    high_res_source = np.asarray(handler.data.hourly[...])
    for counter, batch in enumerate(batcher):
        assert batch.high_res.shape[3] == hr_tsteps
        assert batch.low_res.shape[3] == hr_tsteps // t_enhance

        # make sure the high res sample is found in the source handler data
        daily_idx, hourly_idx = batcher.containers[0].index_record[counter]
        hr_source = high_res_source[:, :, hourly_idx[2], :]
        found = False
        for i in range(hr_source.shape[2] - hr_tsteps + 1):
            check = hr_source[..., i : i + hr_tsteps, :]
            mask = np.isnan(check)
            if np.allclose(
                numpy_if_tensor(batch.high_res[0][~mask]), check[~mask]
            ):
                found = True
                break
        assert found

        # make sure the daily avg data corresponds to the high res data slice
        day_start = int(hourly_idx[2].start / 24)
        day_stop = int(hourly_idx[2].stop / 24)
        check = handler.data.daily[:, :, slice(day_start, day_stop)]
        assert np.allclose(numpy_if_tensor(batch.low_res[0]), check)
        check = handler.data.daily[:, :, daily_idx[2]]
        assert np.allclose(numpy_if_tensor(batch.low_res[0]), check)
    batcher.stop()


def test_solar_batching_spatial():
    """Test batching of nsrdb data with spatial only enhancement"""
    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, FEATURES_S, **dh_kwargs)

    batcher = BatchHandlerTesterCC(
        [handler],
        val_containers=[],
        batch_size=8,
        n_batches=10,
        s_enhance=2,
        t_enhance=1,
        sample_shape=(20, 20, 1),
        feature_sets={
            'lr_features': FEATURES_S,
            'hr_out_features': ['clearsky_ratio'],
        },
    )

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 1)
        assert batch.low_res.shape == (8, 10, 10, len(FEATURES_S))
    batcher.stop()


def test_solar_batch_nan_stats():
    """Test that the batch handler calculates the correct statistics even with
    NaN data present"""
    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, FEATURES_S, **dh_kwargs)

    true_csr_mean = np.nanmean(handler.data.hourly['clearsky_ratio'][...])
    true_csr_stdev = np.nanstd(handler.data.hourly['clearsky_ratio'][...])

    batcher = BatchHandlerTesterCC(
        [handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(10, 10, 9),
    )

    assert np.allclose(batcher.means[FEATURES_S[0]], true_csr_mean)
    assert np.allclose(batcher.stds[FEATURES_S[0]], true_csr_stdev)

    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, FEATURES_S, **dh_kwargs)

    batcher = BatchHandlerCC(
        [handler, handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(10, 10, 9),
    )

    assert np.allclose(true_csr_mean, batcher.means[FEATURES_S[0]])
    assert np.allclose(true_csr_stdev, batcher.stds[FEATURES_S[0]])


def test_solar_multi_day_coarse_data():
    """Test a multi day sample with only 9 hours of high res data output"""
    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, FEATURES_S, **dh_kwargs)

    batcher = BatchHandlerTesterCC(
        train_containers=[handler],
        val_containers=[handler],
        batch_size=4,
        n_batches=10,
        s_enhance=4,
        t_enhance=3,
        sample_shape=(20, 20, 9),
        feature_sets={
            'lr_features': FEATURES_S,
            'hr_out_features': ['clearsky_ratio'],
        },
    )

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, len(FEATURES_S))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, len(FEATURES_S))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)
    batcher.stop()

    # run another test with u/v on low res side but not high res
    features = ['clearsky_ratio', 'u', 'v', 'ghi', 'clearsky_ghi']
    feature_sets = {
        'lr_features': features,
        'hr_out_features': ['clearsky_ratio'],
    }
    handler = DataHandlerH5SolarCC(pytest.FP_NSRDB, features, **dh_kwargs)

    batcher = BatchHandlerTesterCC(
        train_containers=[handler],
        val_containers=[handler],
        batch_size=4,
        n_batches=10,
        s_enhance=4,
        t_enhance=3,
        sample_shape=(20, 20, 9),
        feature_sets=feature_sets,
        mode='eager',
    )

    for batch in batcher:
        assert batch.low_res.shape == (4, 5, 5, 3, len(features))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)

    for batch in batcher.val_data:
        assert batch.low_res.shape == (4, 5, 5, 3, len(features))
        assert batch.high_res.shape == (4, 20, 20, 9, 1)
    batcher.stop()


def test_wind_batching():
    """Test the wind climate change data batching object."""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['time_slice'] = slice(None)
    handler = DataHandlerH5WindCC(pytest.FP_WTK, FEATURES_W, **dh_kwargs_new)

    batcher = BatchHandlerTesterCC(
        [handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(20, 20, 72),
    )

    for batch in batcher:
        assert batch.high_res.shape[3] == 72
        assert batch.low_res.shape[3] == 3

        assert batch.high_res.shape[-1] == len(FEATURES_W)
        assert batch.low_res.shape[-1] == len(FEATURES_W)

        # make sure all of the hours correspond to their daily averages
        slices = [slice(0, 24), slice(24, 48), slice(48, 72)]
        for i, islice in enumerate(slices):
            hourly = batch.high_res[:, :, :, islice, :]
            truth = np.mean(hourly, axis=3)
            daily = batch.low_res[:, :, :, i, :]
            assert np.allclose(daily, truth, atol=1e-6)

    # make sure that each daily/hourly time slices corresponds to each other,
    # and that each hourly time slice starts/ends at daily 24-hour boundaries
    index_record = batcher.containers[0].index_record
    for idx in index_record:
        idx_lr, idx_hr = idx
        idt_lr = idx_lr[2]
        idt_hr = idx_hr[2]
        assert idt_lr.start * 24 == idt_hr.start
        assert idt_lr.stop * 24 == idt_hr.stop
        assert idt_hr.start % 24 == 0
        assert idt_hr.stop % 24 == 0

    batcher.stop()


def test_wind_batching_spatial(plot=False):
    """Test batching of wind data with spatial only enhancement"""
    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_W
    dh_kwargs_new['time_slice'] = slice(None)
    handler = DataHandlerH5WindCC(pytest.FP_WTK, FEATURES_W, **dh_kwargs_new)

    batcher = BatchHandlerTesterCC(
        [handler],
        [],
        batch_size=8,
        n_batches=10,
        s_enhance=5,
        t_enhance=1,
        sample_shape=(20, 20),
        mode='eager',
    )

    for batch in batcher:
        assert batch.high_res.shape == (8, 20, 20, 3)
        assert batch.low_res.shape == (8, 4, 4, 3)

    if plot:
        for p, batch in enumerate(batcher):
            for i in range(batch.high_res.shape[3]):
                _, axes = plt.subplots(1, 2, figsize=(10, 4))

                tmp = (
                    batch.high_res[i, :, :, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[0].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[0])
                axes[0].set_title('Batch high res cs ratio')

                tmp = (
                    batch.low_res[i, :, :, 0] * batcher.stds[0]
                    + batcher.means[0]
                )
                a = axes[1].imshow(tmp, vmin=tmp.min(), vmax=tmp.max())
                plt.colorbar(a, ax=axes[1])
                axes[1].set_title('Batch low res cs ratio')

                plt.savefig(
                    './test_wind_batch_{}_{}.png'.format(p, i),
                    dpi=300,
                    bbox_inches='tight',
                )
                plt.close()

            if p > 4:
                break
    batcher.stop()


def test_surf_min_max_vars():
    """Test data handling of min / max training only variables"""
    surf_features = [
        'temperature_2m',
        'relativehumidity_2m',
        'temperature_min_2m',
        'temperature_max_2m',
        'relativehumidity_min_2m',
        'relativehumidity_max_2m',
    ]

    dh_kwargs_new = dh_kwargs.copy()
    dh_kwargs_new['target'] = TARGET_SURF
    dh_kwargs_new['time_slice'] = slice(None, None, 1)
    handler = DataHandlerH5WindCC(
        pytest.FP_WTK_SURF, surf_features, **dh_kwargs_new
    )

    batcher = BatchHandlerTesterCC(
        [handler],
        [],
        batch_size=1,
        n_batches=10,
        s_enhance=1,
        t_enhance=24,
        sample_shape=(20, 20, 72),
        feature_sets={
            'lr_features': surf_features,
            'hr_out_features': ['temperature_2m', 'relativehumidity_2m'],
        },
        mode='eager',
    )

    assert (
        batcher.low_res['temperature_2m'].data
        > batcher.low_res['temperature_min_2m'].data
    ).all()
    assert (
        batcher.low_res['temperature_2m'].data
        < batcher.low_res['temperature_max_2m'].data
    ).all()
    assert (
        batcher.low_res['relativehumidity_2m'].data
        > batcher.low_res['relativehumidity_min_2m'].data
    ).all()
    assert (
        batcher.low_res['relativehumidity_2m'].data
        < batcher.low_res['relativehumidity_max_2m'].data
    ).all()

    assert (
        batcher.means['temperature_2m']
        == batcher.means['temperature_min_2m']
        == batcher.means['temperature_max_2m']
    )
    assert (
        batcher.stds['temperature_2m']
        == batcher.stds['temperature_min_2m']
        == batcher.stds['temperature_max_2m']
    )

    for _, batch in enumerate(batcher):
        assert batch.high_res.shape[3] == 72
        assert batch.low_res.shape[3] == 3

        assert batch.high_res.shape[-1] == len(surf_features) - 4
        assert batch.low_res.shape[-1] == len(surf_features)

        # compare daily avg temp vs min and max
        blr = batch.low_res
        assert (blr[..., 0] > blr[..., 2]).all()
        assert (blr[..., 0] < blr[..., 3]).all()

        # compare daily avg rh vs min and max
        assert (blr[..., 1] > blr[..., 4]).all()
        assert (blr[..., 1] < blr[..., 5]).all()
    batcher.stop()
