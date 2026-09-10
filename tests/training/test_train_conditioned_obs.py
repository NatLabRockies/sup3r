"""Test the training of super resolution GANs with exogenous observation
data."""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGanFixedObs
from sup3r.preprocessing import (
    BatchHandler,
    DataHandler,
)
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_W = ['u_10m', 'v_10m']
TARGET_W = (39.01, -105.15)


@pytest.mark.parametrize('gen_config', ['gen_config_with_concat_masked'])
def test_fixed_wind_obs(gen_config, request):
    """Test a special model which fixes observations mid network with
    ``Sup3rConcatObs`` layer."""

    gen_config = request.getfixturevalue(gen_config)()
    kwargs = {
        'file_paths': pytest.FP_WTK,
        'features': FEATURES_W,
        'target': TARGET_W,
        'shape': SHAPE,
    }

    train_handler = DataHandler(**kwargs, time_slice=slice(None, 3000, 10))

    val_handler = DataHandler(**kwargs, time_slice=slice(3000, None, 10))
    batcher = BatchHandler(
        [train_handler],
        [val_handler],
        batch_size=2,
        n_batches=1,
        s_enhance=2,
        t_enhance=1,
        sample_shape=(20, 20, 1),
    )

    Sup3rGanFixedObs.seed()

    model = Sup3rGanFixedObs(
        gen_config,
        pytest.S_FP_DISC,
        onshore_obs_frac={'spatial': 0.1},
        loss_obs_weight=0.1,
        learning_rate=1e-4,
    )
    model.meta['hr_out_features'] = ['u_10m', 'v_10m']
    test_mask = model._get_full_obs_mask(np.zeros((1, 20, 20, 1, 1))).numpy()
    frac = 1 - test_mask.sum() / test_mask.size
    assert np.abs(0.1 - frac) < test_mask.size / (2 * np.sqrt(test_mask.size))
    assert model.obs_features == ['u_10m_obs', 'v_10m_obs']
    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '16km', 'temporal': '3600min'},
            'n_epoch': 3,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': None,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batcher, **model_kwargs)

        loaded = model.load(os.path.join(td, 'test_2'))
        loaded.train(batcher, **model_kwargs)

    x = RANDOM_GENERATOR.uniform(0, 1, (4, 30, 30, len(FEATURES_W)))
    u10m_obs = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 1))
    v10m_obs = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 1))
    mask = RANDOM_GENERATOR.choice([True, False], (60, 60, 1), p=[0.9, 0.1])
    u10m_obs[:, mask] = np.nan
    v10m_obs[:, mask] = np.nan

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'u_10m_obs': {
            'steps': [{'model': 0, 'combine_type': 'layer', 'data': u10m_obs}]
        },
        'v_10m_obs': {
            'steps': [{'model': 0, 'combine_type': 'layer', 'data': v10m_obs}]
        },
    }
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == len(FEATURES_W)
