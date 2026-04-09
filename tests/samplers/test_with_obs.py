"""Test sampler behavior with proxy observations."""

import numpy as np
import pytest

from sup3r.preprocessing import DualSampler, Sampler
from sup3r.preprocessing.base import Sup3rDataset
from sup3r.utilities.pytest.helpers import DummyData

LR_FEATURES = ['u_100m', 'v_100m', 'temperature_2m']
HR_OUT_FEATURES = ['u_100m', 'v_100m']
OBS_FEATURES = ['u_100m_obs', 'v_100m_obs']


def _make_sampler(
    sampler_cls,
    hr_shape,
    sample_shape,
    batch_size,
    proxy_obs_kwargs,
    hr_features=None,
):
    """Create either Sampler or DualSampler with proxy obs feature sets."""
    hr_features = hr_features or LR_FEATURES
    feature_sets = {
        'lr_features': LR_FEATURES,
        'hr_out_features': HR_OUT_FEATURES,
        'hr_exo_features': [
            *[f for f in hr_features if f == 'topography'],
            *OBS_FEATURES,
        ],
    }

    if sampler_cls is Sampler:
        data = DummyData(data_shape=hr_shape, features=hr_features)
        return Sampler(
            data,
            sample_shape=sample_shape,
            batch_size=batch_size,
            proxy_obs_kwargs=proxy_obs_kwargs,
            feature_sets=feature_sets,
        )

    lr_shape = (hr_shape[0] // 2, hr_shape[1] // 2, hr_shape[2])
    lr = DummyData(data_shape=lr_shape, features=LR_FEATURES).data.high_res
    hr = DummyData(data_shape=hr_shape, features=hr_features).data.high_res
    data = Sup3rDataset(low_res=lr, high_res=hr)
    return DualSampler(
        data,
        sample_shape=sample_shape,
        batch_size=batch_size,
        s_enhance=2,
        t_enhance=1,
        proxy_obs_kwargs=proxy_obs_kwargs,
        feature_sets=feature_sets,
    )


def _get_hr_batch(sampler):
    """Extract the high-res batch for Sampler and DualSampler outputs."""
    batch = next(sampler)
    return batch[-1] if isinstance(batch, tuple) else batch


@pytest.mark.parametrize('sampler_cls', [Sampler, DualSampler])
@pytest.mark.parametrize(
    'sample_shape, obs_fracs, expected',
    [
        ((30, 30, 1), {'spatial': 0.4, 'temporal': 1.0}, 0.4),
        ((30, 30, 12), {'spatial': 0.4, 'temporal': 0.5}, 0.2),
    ],
)
def test_proxy_obs_appended_and_fraction(
    sampler_cls, sample_shape, obs_fracs, expected
):
    """Proxy obs channels are appended and sampled at configured fraction."""
    sampler = _make_sampler(
        sampler_cls=sampler_cls,
        hr_shape=(60, 60, 500),
        sample_shape=sample_shape,
        batch_size=20,
        proxy_obs_kwargs={'onshore_obs_frac': obs_fracs},
    )

    batch = _get_hr_batch(sampler)
    obs = batch[..., -2:]

    expected_channels = 5 if sampler_cls is Sampler else 4
    assert batch.shape[-1] == expected_channels
    assert obs.shape[-1] == 2

    observed_frac = np.isfinite(obs[..., 0]).mean()
    assert np.isclose(observed_frac, expected, atol=0.05)


@pytest.mark.parametrize('sampler_cls', [Sampler, DualSampler])
def test_proxy_obs_fraction_bounds_with_ranges(sampler_cls):
    """Observed fraction stays within expected range for sampled fractions."""
    s_range = [0.1, 0.3]
    t_range = [0.2, 0.6]
    sampler = _make_sampler(
        sampler_cls=sampler_cls,
        hr_shape=(80, 80, 500),
        sample_shape=(40, 40, 20),
        batch_size=8,
        proxy_obs_kwargs={
            'onshore_obs_frac': {'spatial': s_range, 'temporal': t_range}
        },
    )

    batch = _get_hr_batch(sampler)
    obs = batch[..., -2:]
    observed_by_sample = np.isfinite(obs[..., 0]).mean(axis=(1, 2, 3))

    lower = s_range[0] * t_range[0]
    upper = s_range[1] * t_range[1]
    assert np.all(observed_by_sample >= (lower - 0.02))
    assert np.all(observed_by_sample <= (upper + 0.02))


@pytest.mark.parametrize('sampler_cls', [Sampler, DualSampler])
def test_proxy_obs_onshore_offshore_topography_fractions(sampler_cls):
    """Onshore and offshore obs fractions are applied by topography mask."""
    sampler = _make_sampler(
        sampler_cls=sampler_cls,
        hr_shape=(80, 80, 500),
        sample_shape=(40, 40, 12),
        batch_size=8,
        proxy_obs_kwargs={
            'onshore_obs_frac': {'spatial': 0.8, 'temporal': 1.0},
            'offshore_obs_frac': {'spatial': 0.1, 'temporal': 1.0},
        },
        hr_features=[*LR_FEATURES, 'topography'],
    )

    topo_var = sampler.data.high_res['topography']
    topo = np.ones(topo_var.shape, dtype=np.float32)
    topo[:, : topo.shape[1] // 2, :] = -1.0
    sampler.data.high_res['topography'] = (topo_var.dims, topo)

    batch = _get_hr_batch(sampler)
    topo_idx = sampler.hr_source_features.index('topography')
    topo_sample = batch[..., topo_idx]
    obs = batch[..., -2:]

    onshore = topo_sample > 0
    offshore = ~onshore

    onshore_frac = np.isfinite(obs[..., 0][onshore]).mean()
    offshore_frac = np.isfinite(obs[..., 0][offshore]).mean()

    assert np.isclose(onshore_frac, 0.8, atol=0.12)
    assert np.isclose(offshore_frac, 0.1, atol=0.08)
    assert onshore_frac > offshore_frac
