"""Smoke tests for batcher objects. Just make sure things run without errors"""

import pytest

from sup3r.preprocessing import (
    DualBatchQueue,
    DualSampler,
    SingleBatchQueue,
)
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import (
    DummyData,
    DummySampler,
)

FEATURES = ['windspeed', 'winddirection']


def test_batch_queue():
    """Smoke test for batch queue."""

    sample_shape = (8, 8, 10)
    samplers = [
        DummySampler(
            sample_shape,
            data_shape=(10, 10, 20),
            batch_size=4,
            features=FEATURES,
        ),
        DummySampler(
            sample_shape,
            data_shape=(12, 12, 15),
            batch_size=4,
            features=FEATURES,
        ),
    ]
    transform_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    batcher = SingleBatchQueue(
        samplers=samplers,
        n_batches=3,
        batch_size=4,
        s_enhance=2,
        t_enhance=2,
        queue_cap=10,
        max_workers=1,
        transform_kwargs=transform_kwargs,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, 4, 4, 5, len(FEATURES))
        assert b.high_res.shape == (4, 8, 8, 10, len(FEATURES))
    batcher.stop()


def test_spatial_batch_queue():
    """Smoke test for spatial batch queue. A batch queue returns batches for
    spatial models if the sample shapes have 1 for the time axis"""
    sample_shape = (8, 8)
    s_enhance = 2
    t_enhance = 1
    batch_size = 4
    queue_cap = 10
    n_batches = 3
    transform_kwargs = {'smoothing_ignore': [], 'smoothing': None}
    samplers = [
        DummySampler(
            sample_shape,
            data_shape=(10, 10, 20),
            batch_size=4,
            features=FEATURES,
        ),
        DummySampler(
            sample_shape,
            data_shape=(12, 12, 15),
            batch_size=4,
            features=FEATURES,
        ),
    ]
    batcher = SingleBatchQueue(
        samplers=samplers,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        batch_size=batch_size,
        queue_cap=queue_cap,
        max_workers=1,
        transform_kwargs=transform_kwargs,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (
            batch_size,
            sample_shape[0] // s_enhance,
            sample_shape[1] // s_enhance,
            len(FEATURES),
        )
        assert b.high_res.shape == (batch_size, *sample_shape, len(FEATURES))
    batcher.stop()


def test_dual_batch_queue():
    """Smoke test for paired batch queue."""
    lr_sample_shape = (4, 4, 5)
    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=FEATURES,
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=FEATURES,
        ),
    ]
    sampler_pairs = [
        DualSampler(
            Sup3rDataset(low_res=lr.data, high_res=hr.data),
            hr_sample_shape,
            s_enhance=2,
            t_enhance=2,
            batch_size=4,
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]
    batcher = DualBatchQueue(
        samplers=sampler_pairs,
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        max_workers=1,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, *lr_sample_shape, len(FEATURES))
        assert b.high_res.shape == (4, *hr_sample_shape, len(FEATURES))
    batcher.stop()


def test_pair_batch_queue_with_lr_only_features():
    """Smoke test for paired batch queue with an extra lr_only_feature."""
    lr_sample_shape = (4, 4, 5)
    hr_sample_shape = (8, 8, 10)
    lr_only_features = ['dummy_lr_feat']
    lr_features = [*lr_only_features, *FEATURES]
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=lr_features,
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=lr_features,
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=FEATURES,
        ),
    ]
    sampler_pairs = [
        DualSampler(
            Sup3rDataset(low_res=lr.data, high_res=hr.data),
            hr_sample_shape,
            s_enhance=2,
            t_enhance=2,
            batch_size=4,
            feature_sets={
                'lr_features': lr_features,
                'hr_out_features': FEATURES,
            },
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]
    batcher = DualBatchQueue(
        samplers=sampler_pairs,
        s_enhance=2,
        t_enhance=2,
        n_batches=3,
        batch_size=4,
        queue_cap=10,
        max_workers=1,
    )
    batcher.start()
    assert len(batcher) == 3
    for b in batcher:
        assert b.low_res.shape == (4, *lr_sample_shape, len(lr_features))
        assert b.high_res.shape == (4, *hr_sample_shape, len(FEATURES))
    batcher.stop()


def test_bad_enhancement_factors():
    """Failure when enhancement factors given to BatchQueue do not match those
    given to the contained DualSamplers, and when those given to DualSampler
    are not consistent with the low / high res shapes."""
    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=FEATURES,
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=FEATURES,
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=FEATURES,
        ),
    ]
    for s_enhance, t_enhance in zip([2, 4], [2, 6]):
        with pytest.raises(AssertionError):
            sampler_pairs = [
                DualSampler(
                    Sup3rDataset(low_res=lr.data, high_res=hr.data),
                    hr_sample_shape,
                    s_enhance=s_enhance,
                    t_enhance=t_enhance,
                    batch_size=4,
                )
                for lr, hr in zip(lr_containers, hr_containers)
            ]
            _ = DualBatchQueue(
                samplers=sampler_pairs,
                s_enhance=4,
                t_enhance=6,
                n_batches=3,
                batch_size=4,
                queue_cap=10,
                max_workers=1,
            )


def test_bad_sample_shapes():
    """Failure when sample shapes are not consistent across a collection of
    samplers."""

    samplers = [
        DummySampler(
            sample_shape=(4, 4, 5),
            data_shape=(10, 10, 20),
            batch_size=4,
            features=FEATURES,
        ),
        DummySampler(
            sample_shape=(3, 3, 5),
            data_shape=(12, 12, 15),
            batch_size=4,
            features=FEATURES,
        ),
    ]

    with pytest.raises(AssertionError):
        _ = SingleBatchQueue(
            samplers=samplers,
            s_enhance=4,
            t_enhance=6,
            n_batches=3,
            batch_size=4,
            queue_cap=10,
            max_workers=1,
        )
