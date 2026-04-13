"""Test the training of super resolution GANs with exogenous observation
data."""

import itertools
import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGan
from sup3r.preprocessing import (
    BatchHandler,
    Container,
    DataHandler,
    DualBatchHandler,
    DualRasterizer,
)
from sup3r.preprocessing.samplers import DualSampler
from sup3r.utilities.pytest.helpers import BatchHandlerTesterFactory
from sup3r.utilities.utilities import RANDOM_GENERATOR

DualBatchHandlerWithObsTester = BatchHandlerTesterFactory(
    DualBatchHandler, DualSampler
)

SHAPE = (20, 20)
FEATURES_W = ['u_10m', 'v_10m']
TARGET_W = (39.01, -105.15)


@pytest.mark.parametrize(
    'gen_config, sample_shape, t_enhance, fp_disc',
    [
        ('gen_config_with_obs_2d', (20, 20, 1), 1, pytest.S_FP_DISC),
        ('gen_config_with_obs_3d', (20, 20, 10), 2, pytest.ST_FP_DISC),
    ],
)
def test_train_proxy_obs(
    gen_config, sample_shape, t_enhance, fp_disc, request
):
    """Test a special model which conditions model output on observations
    with a ``Sup3rConcatObs`` layer."""

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
        t_enhance=t_enhance,
        sample_shape=sample_shape,
        proxy_obs_kwargs={'onshore_obs_frac': {'spatial': 0.1}},
        feature_sets={
            'lr_features': FEATURES_W,
            'hr_exo_features': [f'{feat}_obs' for feat in FEATURES_W],
            'hr_out_features': FEATURES_W,
        },
    )

    Sup3rGan.seed()

    model = Sup3rGan(
        gen_config,
        fp_disc,
        learning_rate=1e-4,
        loss={
            'GeothermalObsLoss': {
                'gen_features': FEATURES_W,
                'true_features': [f'{feat}_obs' for feat in FEATURES_W],
            }
        },
    )
    model.meta['hr_out_features'] = FEATURES_W
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

    assert model.obs_features == [f'{feat}_obs' for feat in FEATURES_W]

    if t_enhance == 1:
        x = RANDOM_GENERATOR.uniform(0, 1, (4, 30, 30, len(FEATURES_W)))
        u10m_obs = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 1))
        v10m_obs = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 1))
    else:
        x = RANDOM_GENERATOR.uniform(0, 1, (4, 30, 30, 10, len(FEATURES_W)))
        u10m_obs = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 20, 1))
        v10m_obs = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 20, 1))
    mask = RANDOM_GENERATOR.choice(
        [True, False], u10m_obs.shape[1:], p=[0.9, 0.1]
    )
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
    assert y.shape[-1] == len(FEATURES_W)
    if y.ndim == 5:
        assert y.shape[3] == x.shape[3] * t_enhance


@pytest.mark.parametrize(
    'gen_config, sample_shape, t_enhance, fp_disc',
    [
        ('gen_config_with_obs_2d', (20, 20, 1), 1, pytest.S_FP_DISC),
        ('gen_config_with_obs_3d', (20, 20, 10), 2, pytest.ST_FP_DISC),
    ],
)
def test_train_real_obs(gen_config, sample_shape, t_enhance, fp_disc, request):
    """Test model training with only sparse high resolution ground truth data.
    This should skip any calculations involving the discriminator - since
    train_disc=False"""

    gen_config = request.getfixturevalue(gen_config)()
    kwargs = {
        'features': FEATURES_W,
        'target': TARGET_W,
        'shape': (20, 20),
    }
    hr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        time_slice=slice(None, None, 1),
    )

    lr_handler = DataHandler(
        pytest.FP_ERA,
        features=FEATURES_W,
        time_slice=slice(None, None, t_enhance),
    )

    dual_rasterizer = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=2,
        t_enhance=t_enhance,
        run_qa=False,
    )
    obs_data = dual_rasterizer.high_res.copy()
    for feat in FEATURES_W:
        tmp = np.full(obs_data[feat].shape, np.nan)
        lat_ids = RANDOM_GENERATOR.choice(
            obs_data[feat].shape[0], size=5, replace=False
        )
        lon_ids = RANDOM_GENERATOR.choice(
            obs_data[feat].shape[1], size=5, replace=False
        )
        for ilat, ilon in itertools.product(lat_ids, lon_ids):
            tmp[ilat, ilon, :] = obs_data[feat][ilat, ilon]
        obs_data[f'{feat}_obs'] = (obs_data[feat].dims, tmp)

    dual_with_obs = Container(
        data={
            'low_res': dual_rasterizer.low_res,
            'high_res': obs_data,
        }
    )

    batch_handler = DualBatchHandlerWithObsTester(
        train_containers=[dual_with_obs],
        val_containers=[],
        sample_shape=sample_shape,
        batch_size=3,
        s_enhance=2,
        t_enhance=t_enhance,
        n_batches=2,
        feature_sets={
            'lr_features': FEATURES_W,
            'hr_exo_features': [f'{feat}_obs' for feat in FEATURES_W],
            'hr_out_features': [f'{feat}_obs' for feat in FEATURES_W],
        },
        mode='lazy',
    )

    for batch in batch_handler:
        assert not np.isnan(batch.high_res).all()
        assert np.isnan(batch.high_res).any()

    Sup3rGan.seed()
    model = Sup3rGan(
        gen_config,
        fp_disc,
        learning_rate=1e-4,
        loss={
            'GeothermalObsLoss': {
                'gen_features': [f'{feat}_obs' for feat in FEATURES_W],
                'true_features': [f'{feat}_obs' for feat in FEATURES_W],
            }
        },
    )

    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': 5,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tloss = model.history['train_geothermal_obs_loss'].values
        assert not np.isnan(tloss).any()


@pytest.mark.parametrize('lr_only_features', [[], ['temperature_2m']])
def test_train_proxy_obs_with_topo(lr_only_features, request):
    """Test training with topo and obs. Make sure exo features are
    properly concatenated."""

    gen_config = 'gen_config_with_obs_3d_topo'
    gen_config = request.getfixturevalue(gen_config)()
    kwargs = {
        'file_paths': pytest.FP_WTK,
        'features': [*FEATURES_W, 'topography'],
        'target': TARGET_W,
        'shape': SHAPE,
    }

    train_handler = DataHandler(**kwargs, time_slice=slice(None, 3000, 10))
    val_handler = DataHandler(**kwargs, time_slice=slice(3000, None, 10))

    # Add dummy lr only features
    if lr_only_features:
        for feat in lr_only_features:
            train_handler[feat] = train_handler[FEATURES_W[0]].copy()
            val_handler[feat] = val_handler[FEATURES_W[0]].copy()

    batcher = BatchHandler(
        [train_handler],
        [val_handler],
        batch_size=2,
        n_batches=1,
        s_enhance=2,
        t_enhance=2,
        sample_shape=(20, 20, 10),
        proxy_obs_kwargs={'onshore_obs_frac': {'spatial': 0.1}},
        feature_sets={
            'lr_features': [*lr_only_features, *FEATURES_W],
            'hr_exo_features': [
                'topography',
                *[f'{feat}_obs' for feat in FEATURES_W],
            ],
            'hr_out_features': FEATURES_W,
        },
    )

    Sup3rGan.seed()

    model = Sup3rGan(
        gen_config,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        loss={
            'GeothermalObsLoss': {
                'gen_features': FEATURES_W,
                'true_features': [f'{feat}_obs' for feat in FEATURES_W],
            }
        },
    )
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

        loss = model.history['train_geothermal_obs_loss'].values
        assert not np.isnan(loss).any()
        gloss = model.history['train_loss_gen'].values
        assert not np.isnan(gloss).any()


@pytest.mark.parametrize('lr_only_features', [[], ['temperature_2m']])
def test_train_real_obs_with_topo(lr_only_features, request):
    """Test model training with only sparse high resolution ground truth data.
    This should skip any calculations involving the discriminator - since
    train_disc=False"""

    gen_config = request.getfixturevalue('gen_config_with_obs_3d_topo')()
    kwargs = {
        'features': [*FEATURES_W, 'topography'],
        'target': TARGET_W,
        'shape': SHAPE,
    }

    hr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        time_slice=slice(None, None, 1),
    )

    lr_handler = DataHandler(
        pytest.FP_ERA,
        features=FEATURES_W,
        time_slice=slice(None, None, 2),
    )

    # Add dummy lr only features
    if lr_only_features:
        for feat in lr_only_features:
            lr_handler[feat] = lr_handler[FEATURES_W[0]].copy()

    dual_rasterizer = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=2,
        t_enhance=2,
        run_qa=False,
    )
    obs_data = dual_rasterizer.high_res.copy()
    for feat in FEATURES_W:
        tmp = np.full(obs_data[feat].shape, np.nan)
        lat_ids = RANDOM_GENERATOR.choice(
            obs_data[feat].shape[0], size=5, replace=False
        )
        lon_ids = RANDOM_GENERATOR.choice(
            obs_data[feat].shape[1], size=5, replace=False
        )
        for ilat, ilon in itertools.product(lat_ids, lon_ids):
            tmp[ilat, ilon, :] = obs_data[feat][ilat, ilon]
        obs_data[f'{feat}_obs'] = (obs_data[feat].dims, tmp)

    dual_with_obs = Container(
        data={
            'low_res': dual_rasterizer.low_res,
            'high_res': obs_data,
        }
    )

    batch_handler = DualBatchHandlerWithObsTester(
        train_containers=[dual_with_obs],
        val_containers=[],
        sample_shape=(20, 20, 10),
        batch_size=2,
        s_enhance=2,
        t_enhance=2,
        n_batches=1,
        feature_sets={
            'lr_features': [*lr_only_features, *FEATURES_W],
            'hr_exo_features': [
                'topography',
                *[f'{feat}_obs' for feat in FEATURES_W],
            ],
            'hr_out_features': FEATURES_W,
        },
        mode='lazy',
    )

    Sup3rGan.seed()
    model = Sup3rGan(
        gen_config,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        loss={
            'MeanAbsoluteError': {},
            'GeothermalObsLoss': {
                'gen_features': FEATURES_W,
                'true_features': [f'{feat}_obs' for feat in FEATURES_W],
            },
        },
    )

    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': 5,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tloss = model.history['train_loss_gen'].values
        assert not np.isnan(tloss).any()

        model_kwargs['train_disc'] = True
        model.train(batch_handler, **model_kwargs)

        tloss_disc = model.history['train_loss_disc'].values
        assert not np.isnan(tloss_disc).any()
