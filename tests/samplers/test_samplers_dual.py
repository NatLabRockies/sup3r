"""Dual sampler regression tests."""

import numpy as np

from sup3r.preprocessing import DualSampler
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import DummyData
from sup3r.utilities.utilities import RANDOM_GENERATOR

LR_FEATURES = ['u_100m', 'v_100m', 'temperature_2m']


def test_dual_sampler_eager_vs_lazy():
    """Eager dual sampling should match lazy sampling for the same indices."""
    lr = DummyData(
        data_shape=(20, 20, 100), features=LR_FEATURES
    ).data.high_res
    hr = DummyData(
        data_shape=(40, 40, 100), features=[*LR_FEATURES, 'topography']
    ).data.high_res
    data = Sup3rDataset(low_res=lr, high_res=hr)
    kwargs = {
        'data': data,
        'sample_shape': (20, 20, 8),
        'batch_size': 4,
        's_enhance': 2,
        't_enhance': 1,
        'feature_sets': {
            'lr_features': LR_FEATURES,
            'hr_out_features': ['u_100m', 'v_100m'],
            'hr_exo_features': ['topography'],
        },
    }

    state = RANDOM_GENERATOR.bit_generator.state
    eager_sampler = DualSampler(mode='eager', **kwargs)
    RANDOM_GENERATOR.bit_generator.state = state
    lazy_sampler = DualSampler(mode='lazy', **kwargs)

    eager_batch = next(eager_sampler)
    RANDOM_GENERATOR.bit_generator.state = state
    lazy_batch = next(lazy_sampler)

    assert np.allclose(eager_batch[0], lazy_batch[0])
    assert np.allclose(eager_batch[1], lazy_batch[1])
