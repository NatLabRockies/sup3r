"""Test the training of GANs with dual data handler"""

import itertools
import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGanWithObs
from sup3r.preprocessing import (
    Container,
    DataHandler,
    DualBatchHandler,
    DualRasterizer,
)
from sup3r.preprocessing.samplers import DualSampler
from sup3r.utilities.pytest.helpers import BatchHandlerTesterFactory

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']


DualBatchHandlerWithObsTester = BatchHandlerTesterFactory(
    DualBatchHandler, DualSampler
)


@pytest.mark.parametrize(
    [
        'fp_gen',
        'fp_disc',
        's_enhance',
        't_enhance',
        'sample_shape',
        'mode',
    ],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'lazy'),
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'eager'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'lazy'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'eager'),
    ],
)
def test_train_h5_nc(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, mode, n_epoch=2
):
    """Test model training with a dual data handler / batch handler with h5 and
    era as hr / lr datasets. Tests both spatiotemporal and spatial models."""

    lr = 1e-5
    kwargs = {
        'features': FEATURES,
        'target': TARGET_COORD,
        'shape': (20, 20),
    }
    hr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        time_slice=slice(None, None, 1),
    )

    lr_handler = DataHandler(
        pytest.FP_ERA,
        features=FEATURES,
        time_slice=slice(None, None, t_enhance),
    )

    dual_rasterizer = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=s_enhance,
        t_enhance=t_enhance,
    )
    obs_data = dual_rasterizer.high_res.copy()
    for feat in FEATURES:
        tmp = np.full(obs_data[feat].shape, np.nan)
        lat_ids = list(range(0, 20, 4))
        lon_ids = list(range(0, 20, 4))
        for ilat, ilon in itertools.product(lat_ids, lon_ids):
            tmp[ilat, ilon, :] = obs_data[feat][ilat, ilon]
        obs_data[feat] = (obs_data[feat].dims, tmp)

    dual_with_obs = Container(
        data={
            'low_res': dual_rasterizer.low_res,
            'high_res': dual_rasterizer.high_res,
            'obs': obs_data,
        }
    )

    batch_handler = DualBatchHandlerWithObsTester(
        train_containers=[dual_with_obs],
        val_containers=[],
        sample_shape=sample_shape,
        batch_size=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=3,
        mode=mode,
    )

    for batch in batch_handler:
        assert hasattr(batch, 'obs')
        assert not np.isnan(batch.obs).all()
        assert np.isnan(batch.obs).any()

    Sup3rGanWithObs.seed()
    model = Sup3rGanWithObs(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tlossg = model.history['train_loss_gen'].values
        tlosso = model.history['train_loss_obs'].values
        assert np.sum(np.diff(tlossg)) < 0
        assert np.sum(np.diff(tlosso)) < 0


@pytest.mark.parametrize(
    [
        'fp_gen',
        'fp_disc',
        's_enhance',
        't_enhance',
        'sample_shape',
        'mode',
    ],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'lazy'),
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16), 'eager'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'lazy'),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (20, 20, 1), 'eager'),
    ],
)
def test_train_coarse_h5(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, mode, n_epoch=2
):
    """Test model training with a dual data handler / batch handler with
    additional sparse observation data used in extra content loss term. Tests
    both spatiotemporal and spatial models."""

    lr = 1e-5
    kwargs = {
        'features': FEATURES,
        'target': TARGET_COORD,
        'shape': (20, 20),
    }
    hr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        time_slice=slice(None, None, 1),
    )

    lr_handler = DataHandler(
        pytest.FP_WTK,
        **kwargs,
        hr_spatial_coarsen=s_enhance,
        time_slice=slice(None, None, t_enhance),
    )

    dual_rasterizer = DualRasterizer(
        data={'low_res': lr_handler.data, 'high_res': hr_handler.data},
        s_enhance=s_enhance,
        t_enhance=t_enhance,
    )
    obs_data = dual_rasterizer.high_res.copy()
    for feat in FEATURES:
        tmp = np.full(obs_data[feat].shape, np.nan)
        lat_ids = list(range(0, 20, 4))
        lon_ids = list(range(0, 20, 4))
        for ilat, ilon in itertools.product(lat_ids, lon_ids):
            tmp[ilat, ilon, :] = obs_data[feat][ilat, ilon]
        obs_data[feat] = (obs_data[feat].dims, tmp)

    dual_with_obs = Container(
        data={
            'low_res': dual_rasterizer.low_res,
            'high_res': dual_rasterizer.high_res,
            'obs': obs_data,
        }
    )

    batch_handler = DualBatchHandlerWithObsTester(
        train_containers=[dual_with_obs],
        val_containers=[],
        sample_shape=sample_shape,
        batch_size=3,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=3,
        mode=mode,
    )

    for batch in batch_handler:
        assert hasattr(batch, 'obs')
        assert not np.isnan(batch.obs).all()
        assert np.isnan(batch.obs).any()

    Sup3rGanWithObs.seed()
    model = Sup3rGanWithObs(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    with tempfile.TemporaryDirectory() as td:
        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tlossg = model.history['train_loss_gen'].values
        tlosso = model.history['train_loss_obs'].values
        assert np.sum(np.diff(tlossg)) < 0
        assert np.sum(np.diff(tlosso)) < 0
