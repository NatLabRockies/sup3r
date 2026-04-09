"""Test correct handling of feature sets by samplers"""

import pytest

from sup3r.preprocessing import DualSampler, Sampler
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import DummyData


@pytest.mark.parametrize(
    ['features', 'lr_features', 'hr_exo_features', 'hr_out_features'],
    [
        (['V_100m'], ['V_100m'], [], ['U_100m']),
        (['U_100m'], ['V_100m'], ['V_100m'], []),
    ],
)
def test_feature_errors(
    features, lr_features, hr_exo_features, hr_out_features
):
    """Each of these feature combinations should raise an error due to no
    features left in hr output or bad ordering"""
    with pytest.raises((RuntimeError, AssertionError)):
        _ = Sampler(
            DummyData(data_shape=(20, 20, 10), features=features),
            sample_shape=(5, 5, 4),
            feature_sets={
                'lr_features': lr_features,
                'hr_exo_features': hr_exo_features,
                'hr_out_features': hr_out_features,
            },
        )


@pytest.mark.parametrize(
    ['lr_features', 'hr_exo_features', 'hr_out_features'],
    [
        (['V_100m', 'topography'], ['topography'], ['V_100m_obs']),
        (
            ['V_100m', 'topography'],
            ['topography', 'V_100m_obs'],
            ['V_100m_obs'],
        ),
    ],
)
def test_sampler_feature_sets(lr_features, hr_exo_features, hr_out_features):
    """Each of these feature combinations should pass without raising an
    error."""
    feats = set(lr_features) | set(hr_exo_features) | set(hr_out_features)
    sampler = Sampler(
        DummyData(data_shape=(20, 20, 10), features=sorted(feats)),
        sample_shape=(5, 5, 4),
        feature_sets={
            'lr_features': lr_features,
            'hr_exo_features': hr_exo_features,
            'hr_out_features': hr_out_features,
        },
    )

    _ = sampler.lr_features
    _ = sampler.hr_out_features
    _ = sampler.hr_exo_features
    _ = sampler.obs_features


@pytest.mark.parametrize(
    ['lr_features', 'hr_features', 'hr_exo_features'],
    [
        (['U_100m'], ['U_100m', 'V_100m'], ['V_100m']),
        (['U_100m'], ['U_100m', 'V_100m'], ('V_100m',)),
        (['U_100m'], ['V_100m', 'BVF2_200m'], ['BVF2_200m']),
        (['U_100m'], ('V_100m', 'BVF2_200m'), ['BVF2_200m']),
        (['U_100m'], ['V_100m', 'BVF2_200m'], []),
    ],
)
def test_mixed_lr_hr_features(lr_features, hr_features, hr_exo_features):
    """Each of these feature combinations should work fine with the
    DualSampler."""
    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=[f.lower() for f in lr_features],
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=[f.lower() for f in lr_features],
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=[f.lower() for f in hr_features],
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=[f.lower() for f in hr_features],
        ),
    ]
    sampler_pairs = [
        DualSampler(
            Sup3rDataset(low_res=lr.data, high_res=hr.data),
            hr_sample_shape,
            s_enhance=2,
            t_enhance=2,
            feature_sets={'hr_exo_features': hr_exo_features},
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]

    for pair in sampler_pairs:
        _ = pair.lr_features
        _ = pair.hr_out_features
        _ = pair.hr_exo_features


@pytest.mark.parametrize(
    ['lr_features', 'hr_exo_features', 'hr_out_features'],
    [
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'pressure',
                'kx',
                'dewpoint_temperature',
                'topography',
            ],
            ['topography'],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'pressure',
                'kx',
                'dewpoint_temperature',
                'topography',
                'srl',
            ],
            ['topography', 'srl'],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'pressure',
                'kx',
                'dewpoint_temperature',
            ],
            [],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'topography',
                'srl',
            ],
            ['topography', 'srl'],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
    ],
)
def test_dual_feature_sets(lr_features, hr_exo_features, hr_out_features):
    """Each of these feature combinations should work fine with the dual
    sampler"""

    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=[f.lower() for f in lr_features],
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=[f.lower() for f in lr_features],
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=[f.lower() for f in lr_features],
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=[f.lower() for f in lr_features],
        ),
    ]
    sampler_pairs = [
        DualSampler(
            Sup3rDataset(low_res=lr.data, high_res=hr.data),
            hr_sample_shape,
            s_enhance=2,
            t_enhance=2,
            feature_sets={
                'lr_features': lr_features,
                'hr_exo_features': hr_exo_features,
                'hr_out_features': hr_out_features,
            },
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]

    for pair in sampler_pairs:
        _ = pair.lr_features
        _ = pair.hr_out_features
        _ = pair.hr_exo_features


@pytest.mark.parametrize(
    ['lr_features', 'hr_exo_features', 'hr_out_features'],
    [
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'pressure',
                'kx',
                'dewpoint_temperature',
                'topography',
            ],
            ['topography', 'u_100m_obs', 'u_200m_obs'],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'pressure',
                'kx',
                'dewpoint_temperature',
                'topography',
                'srl',
            ],
            ['topography', 'srl', 'u_100m_obs', 'u_200m_obs'],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'pressure',
                'kx',
                'dewpoint_temperature',
            ],
            [],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
        (
            [
                'u_10m',
                'u_100m',
                'u_200m',
                'u_80m',
                'u_120m',
                'u_140m',
                'topography',
                'srl',
            ],
            ['topography', 'srl', 'u_100m_obs', 'u_200m_obs'],
            ['u_10m', 'u_100m', 'u_200m', 'u_80m', 'u_120m', 'u_140m'],
        ),
    ],
)
def test_dual_feature_sets_with_obs(
    lr_features, hr_exo_features, hr_out_features
):
    """Each of these feature combinations should work fine with the dual
    sampler when obs features are included"""

    hr_sample_shape = (8, 8, 10)
    lr_containers = [
        DummyData(
            data_shape=(10, 10, 20),
            features=[f.lower() for f in lr_features],
        ),
        DummyData(
            data_shape=(12, 12, 15),
            features=[f.lower() for f in lr_features],
        ),
    ]
    hr_containers = [
        DummyData(
            data_shape=(20, 20, 40),
            features=[f.lower() for f in lr_features],
        ),
        DummyData(
            data_shape=(24, 24, 30),
            features=[f.lower() for f in lr_features],
        ),
    ]
    sampler_pairs = [
        DualSampler(
            Sup3rDataset(low_res=lr.data, high_res=hr.data),
            hr_sample_shape,
            s_enhance=2,
            t_enhance=2,
            proxy_obs_kwargs={'onshore_obs_frac': {'spatial': 0.1}},
            feature_sets={
                'lr_features': lr_features,
                'hr_exo_features': hr_exo_features,
                'hr_out_features': hr_out_features,
            },
        )
        for lr, hr in zip(lr_containers, hr_containers)
    ]

    for pair in sampler_pairs:
        _ = pair.lr_features
        _ = pair.hr_out_features
        _ = pair.hr_exo_features
        _ = pair.obs_features
