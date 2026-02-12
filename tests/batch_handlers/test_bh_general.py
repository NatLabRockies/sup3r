"""Smoke tests for batcher objects. Just make sure things run without errors"""
import copy
import os
import time
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from sup3r.preprocessing import BatchHandler, DataHandler
from sup3r.preprocessing.base import Container
from sup3r.utilities.pytest.helpers import (
    BatchHandlerTesterFactory,
    DummyData,
    SamplerTester,
)
from sup3r.utilities.utilities import (
    RANDOM_GENERATOR,
    Timer,
    spatial_coarsening,
    temporal_coarsening,
)

FEATURES = ['windspeed', 'winddirection']
means = dict.fromkeys(FEATURES, 0)
stds = dict.fromkeys(FEATURES, 1)


BatchHandlerTester = BatchHandlerTesterFactory(BatchHandler, SamplerTester)


def test_batch_sampling_workers():
    """Check that it is faster to sample batches with max_workers > 1 than with
    max_workers = 1. This does not include enqueueing and dequeueing."""

    timer = Timer()
    sample_shape = (100, 100, 30)
    chunk_shape = (
        2 * sample_shape[0],
        2 * sample_shape[1],
        2 * sample_shape[-1],
    )
    n_obs = 10
    max_workers = 10
    n_batches = 50
    n_epochs = 3
    chunks = dict(zip(['south_north', 'west_east', 'time'], chunk_shape))

    with TemporaryDirectory() as td:
        ds = DummyData((200, 200, 2000), ['u_100m', 'v_100m'])
        ds.to_netcdf(os.path.join(td, 'test.nc'))
        ds = DataHandler(os.path.join(td, 'test.nc'), chunks=chunks)

        batcher = BatchHandler(
            [ds],
            n_batches=n_batches,
            batch_size=n_obs,
            sample_shape=sample_shape,
            max_workers=max_workers,
            means={'u_100m': 0, 'v_100m': 0},
            stds={'u_100m': 1, 'v_100m': 1},
        )
        timer.start()
        queue_time = 0
        for _ in range(n_epochs):
            batches = batcher.sample_batches(n_batches)
            batches = [batch.result() for batch in batches]
            queue_start = time.time()
            for batch in batches:
                batcher.queue.enqueue(batch)
                _ = batcher.queue.dequeue()
            queue_time += (time.time() - queue_start)
        timer.stop()
        parallel_time = timer.elapsed / (n_batches * n_epochs)
        parallel_queue_time = queue_time / (n_batches * n_epochs)
        batcher.stop()

        batcher = BatchHandler(
            [ds],
            n_batches=n_batches,
            batch_size=n_obs,
            sample_shape=sample_shape,
            max_workers=1,
            means={'u_100m': 0, 'v_100m': 0},
            stds={'u_100m': 1, 'v_100m': 1},
        )
        timer.start()
        queue_time = 0
        for _ in range(n_epochs):
            batches = batcher.sample_batches(n_batches)
            queue_start = time.time()
            for batch in batches:
                batcher.queue.enqueue(batch)
                _ = batcher.queue.dequeue()
            queue_time += time.time() - queue_start
        timer.stop()
        serial_time = timer.elapsed / (n_batches * n_epochs)
        serial_queue_time = queue_time / (n_batches * n_epochs)
        batcher.stop()

        print(
            'Elapsed total time (serial / parallel): {} / {}'.format(
                serial_time, parallel_time
            )
        )
        print(
            'Elapsed queue time (serial / parallel): {} / {}'.format(
                serial_queue_time, parallel_queue_time
            )
        )
        assert serial_time > parallel_time


def test_eager_vs_lazy():
    """Make sure eager and lazy loading agree. We use queue_cap = 0 here so
    there is no disagreement that results from dequeuing vs direct batch
    requests. e.g. when the queue is empty the batch handler will directly
    sample from the contained data."""

    eager_data = DummyData((10, 10, 100), FEATURES)
    lazy_data = Container(copy.deepcopy(eager_data.data))
    kwargs = {
        'val_containers': [],
        'sample_shape': (8, 8, 4),
        'batch_size': 4,
        'n_batches': 4,
        's_enhance': 2,
        't_enhance': 1,
        'queue_cap': 0,
        'means': means,
        'stds': stds,
        'max_workers': 1,
    }

    lazy_batcher = BatchHandlerTester(
        [lazy_data],
        **kwargs,
        mode='lazy',
    )
    eager_batcher = BatchHandlerTester(
        train_containers=[eager_data],
        **kwargs,
        mode='eager',
    )

    assert eager_batcher.loaded
    assert eager_batcher.containers[0].mode == 'eager'
    assert not lazy_batcher.loaded
    assert lazy_batcher.containers[0].mode == 'lazy'

    assert np.array_equal(
        eager_batcher.data[0].as_array(),
        lazy_batcher.data[0].as_array().compute(),
    )

    state = RANDOM_GENERATOR.bit_generator.state
    eager_batches = list(eager_batcher)
    eager_batcher.stop()
    RANDOM_GENERATOR.bit_generator.state = state
    lazy_batches = list(lazy_batcher)
    lazy_batcher.stop()

    for eb, lb in zip(eager_batches, lazy_batches):
        assert np.allclose(eb.high_res, lb.high_res)
        assert np.allclose(eb.low_res, lb.low_res)


def test_not_enough_stats():
    """Negative test for not enough means / stds for given features."""

    dat = DummyData((10, 10, 100), FEATURES)

    with pytest.warns():
        batcher = BatchHandler(
            train_containers=[dat],
            val_containers=[dat],
            sample_shape=(8, 8, 4),
            n_batches=3,
            batch_size=4,
            s_enhance=2,
            t_enhance=2,
            means={'windspeed': 4},
            stds={'windspeed': 2},
            queue_cap=10,
            max_workers=1,
        )
        assert all(f in batcher.means for f in FEATURES)
        assert all(f in batcher.stds for f in FEATURES)


def test_multi_container_normalization():
    """Make sure stats are the same for 2 of the same container as a single
    one"""

    dat = DummyData((10, 10, 100), FEATURES)

    stored_data = dat.as_array()

    batcher1 = BatchHandler(
        train_containers=[dat],
        val_containers=[],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        max_workers=1,
    )

    dat.data['windspeed'] = stored_data[..., 0]
    dat.data['winddirection'] = stored_data[..., 1]

    batcher2 = BatchHandler(
        train_containers=[dat, dat],
        val_containers=[dat],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        max_workers=1,
    )

    assert batcher1.means == batcher2.means
    assert batcher1.stds == batcher2.stds


def test_normalization():
    """Make sure batch handler normalization works correctly."""

    means = {'windspeed': 2, 'winddirection': 5}
    stds = {'windspeed': 6.5, 'winddirection': 8.2}

    dat = DummyData((10, 10, 100), FEATURES)
    ws_tmp = dat.data['windspeed']
    ws_tmp[:] = 1
    ws_tmp[0:4] = np.nan
    dat.data['windspeed'] = ws_tmp
    wd_tmp = dat.data['winddirection']
    wd_tmp[:] = 1
    wd_tmp[0:4] = np.nan
    dat.data['winddirection'] = wd_tmp

    val_dat = DummyData((10, 10, 100), FEATURES)
    val_dat.data['windspeed'] = dat.data['windspeed']
    val_dat.data['winddirection'] = dat.data['winddirection']

    batcher = BatchHandler(
        train_containers=[dat],
        val_containers=[val_dat],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
    )

    means = list(means.values())
    stds = list(stds.values())

    assert (
        np.nanmean(
            batcher.containers[0].as_array()[..., 0] * stds[0] + means[0]
        )
        == 1
    )
    assert (
        np.nanmean(
            batcher.containers[0].as_array()[..., 1] * stds[1] + means[1]
        )
        == 1
    )
    assert (
        np.nanmean(
            batcher.val_data.containers[0].as_array()[..., 0] * stds[0]
            + means[0]
        )
        == 1
    )
    assert (
        np.nanmean(
            batcher.val_data.containers[0].as_array()[..., 1] * stds[1]
            + means[1]
        )
        == 1
    )

    assert len(batcher) == 3
    for b in batcher:
        assert round(np.nanmean(b.low_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.low_res[..., 1]) * stds[1] + means[1]) == 1
        assert round(np.nanmean(b.high_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.high_res[..., 1]) * stds[1] + means[1]) == 1

    assert len(batcher.val_data) == 3
    for b in batcher.val_data:
        assert round(np.nanmean(b.low_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.low_res[..., 1]) * stds[1] + means[1]) == 1
        assert round(np.nanmean(b.high_res[..., 0]) * stds[0] + means[0]) == 1
        assert round(np.nanmean(b.high_res[..., 1]) * stds[1] + means[1]) == 1
    batcher.stop()


def test_batch_handler_with_validation():
    """Smoke test for batch queue."""

    transform_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = BatchHandler(
        train_containers=[DummyData((10, 10, 100), FEATURES)],
        val_containers=[DummyData((10, 10, 100), FEATURES)],
        sample_shape=(8, 8, 4),
        batch_size=4,
        n_batches=3,
        s_enhance=2,
        t_enhance=1,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        transform_kwargs=transform_kwargs,
    )

    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 4, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 4, len(FEATURES))
        assert b.low_res.dtype == np.float32
        assert b.high_res.dtype == np.float32

    assert len(batcher.val_data) == 3
    for b in batcher.val_data:
        assert b.low_res.shape == (4, 4, 4, 4, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 4, len(FEATURES))
        assert b.low_res.dtype == np.float32
        assert b.high_res.dtype == np.float32
    batcher.stop()


@pytest.mark.parametrize(
    'method, t_enhance',
    [
        ('subsample', 2),
        ('average', 2),
        ('total', 2),
        ('subsample', 3),
        ('average', 3),
        ('total', 3),
        ('subsample', 4),
        ('average', 4),
        ('total', 4),
    ],
)
def test_temporal_coarsening(method, t_enhance):
    """Test temporal coarsening of batches"""

    sample_shape = (8, 8, 12)
    s_enhance = 2
    batch_size = 4
    transform_kwargs = {
        'smoothing_ignore': [],
        'smoothing': None,
        'temporal_coarsening_method': method,
    }
    batcher = BatchHandler(
        train_containers=[DummyData((10, 10, 100), FEATURES)],
        val_containers=[DummyData((10, 10, 100), FEATURES)],
        sample_shape=sample_shape,
        batch_size=batch_size,
        n_batches=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        transform_kwargs=transform_kwargs,
    )

    for batch in batcher:
        assert batch.low_res.shape[0] == batch.high_res.shape[0]
        assert batch.low_res.shape == (
            batch_size,
            sample_shape[0] // s_enhance,
            sample_shape[1] // s_enhance,
            sample_shape[2] // t_enhance,
            len(FEATURES),
        )
        assert batch.high_res.shape == (
            batch_size,
            sample_shape[0],
            sample_shape[1],
            sample_shape[2],
            len(FEATURES),
        )
    batcher.stop()


def test_smoothing():
    """Check gaussian filtering on low res"""

    transform_kwargs = {
        'smoothing_ignore': [],
        'smoothing': 0.6,
    }
    s_enhance = 2
    t_enhance = 2
    sample_shape = (10, 10, 12)
    batch_size = 4
    batcher = BatchHandler(
        train_containers=[DummyData((10, 10, 100), FEATURES)],
        val_containers=[DummyData((10, 10, 100), FEATURES)],
        sample_shape=sample_shape,
        batch_size=batch_size,
        n_batches=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        queue_cap=10,
        means=means,
        stds=stds,
        max_workers=1,
        transform_kwargs=transform_kwargs,
    )

    for batch in batcher:
        high_res = batch.high_res
        low_res = spatial_coarsening(high_res, s_enhance)
        low_res = temporal_coarsening(low_res, t_enhance)
        low_res_no_smooth = low_res.copy()
        for i in range(low_res_no_smooth.shape[0]):
            for j in range(low_res_no_smooth.shape[-1]):
                for t in range(low_res_no_smooth.shape[-2]):
                    low_res[i, ..., t, j] = gaussian_filter(
                        low_res_no_smooth[i, ..., t, j], 0.6, mode='nearest'
                    )
        assert np.array_equal(batch.low_res, low_res)
        assert not np.array_equal(low_res, low_res_no_smooth)
    batcher.stop()
