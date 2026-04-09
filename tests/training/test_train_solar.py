"""Test the basic training of super resolution GAN for solar climate change
applications"""

import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

from sup3r import CONFIG_DIR
from sup3r.models import SolarCC, Sup3rGan
from sup3r.preprocessing import BatchHandlerCC, DataHandlerH5SolarCC
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_S = ['clearsky_ratio', 'ghi', 'clearsky_ghi']
TARGET_S = (39.01, -105.13)

# added to get accurate pytest-cov report on tf.function
tf.config.run_functions_eagerly(True)


@pytest.mark.parametrize('hr_steps', (24, 72))
def test_solar_cc_model(hr_steps):
    """Test the solar climate change nsrdb super res model.

    NOTE: that the full 10x model is too big to train on the 20x20 test data.
    """

    kwargs = {
        'file_paths': pytest.FP_NSRDB,
        'features': FEATURES_S,
        'target': TARGET_S,
        'shape': SHAPE,
        'time_roll': -7,
    }
    train_handler = DataHandlerH5SolarCC(
        **kwargs, time_slice=slice(720, None, 2)
    )
    val_handler = DataHandlerH5SolarCC(
        **kwargs, time_slice=slice(None, 720, 2)
    )

    batcher = BatchHandlerCC(
        [train_handler],
        [val_handler],
        batch_size=2,
        n_batches=2,
        s_enhance=1,
        t_enhance=8,
        sample_shape=(20, 20, hr_steps),
        feature_sets={
            'lr_features': FEATURES_S,
            'hr_out_features': ['clearsky_ratio'],
        },
    )

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_1x_8x_1f.json')
    fp_disc = pytest.ST_FP_DISC

    Sup3rGan.seed()
    model = SolarCC(
        fp_gen, fp_disc, learning_rate=1e-4, loss='MeanAbsoluteError',
        t_enhance=8
    )

    with tempfile.TemporaryDirectory() as td:
        model.train(
            batcher,
            input_resolution={'spatial': '4km', 'temporal': '1440min'},
            n_epoch=1,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['clearsky_ratio']
        assert model.meta['class'] == 'SolarCC'

        out_dir = os.path.join(td, 'cc_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        assert model.meta['class'] == 'SolarCC'
        assert loaded.meta['class'] == 'SolarCC'

    x = RANDOM_GENERATOR.uniform(0, 1, (1, 30, 30, hr_steps // 8, 1))
    z = RANDOM_GENERATOR.uniform(0, 1, (1, 30, 30, hr_steps // 8, 1))
    mae = MeanAbsoluteError()(x, z)
    assert np.allclose(model.calc_loss_gen_content(x, z)[0], mae)
    assert np.allclose(loaded.calc_loss_gen_content(x, z)[0], mae)

    y = model.generate(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1]
    assert y.shape[2] == x.shape[2]
    assert y.shape[3] == x.shape[3] * 8
    assert y.shape[4] == x.shape[4]


def test_solar_cc_model_spatial():
    """Test the solar climate change nsrdb super res model with spatial
    enhancement only.
    """

    kwargs = {
        'file_paths': pytest.FP_NSRDB,
        'features': FEATURES_S,
        'target': TARGET_S,
        'shape': SHAPE,
        'time_roll': -7,
    }
    train_handler = DataHandlerH5SolarCC(
        **kwargs, time_slice=slice(720, None, 2)
    )
    val_handler = DataHandlerH5SolarCC(
        **kwargs, time_slice=slice(None, 720, 2)
    )

    batcher = BatchHandlerCC(
        [train_handler],
        [val_handler],
        batch_size=2,
        n_batches=2,
        s_enhance=5,
        t_enhance=1,
        sample_shape=(20, 20),
        feature_sets={
            'lr_features': FEATURES_S,
            'hr_out_features': ['clearsky_ratio'],
        },
    )

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_5x_1x_1f.json')
    fp_disc = pytest.S_FP_DISC

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    with tempfile.TemporaryDirectory() as td:
        model.train(
            batcher,
            input_resolution={'spatial': '25km', 'temporal': '15min'},
            n_epoch=1,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['clearsky_ratio']
        assert model.meta['class'] == 'Sup3rGan'

    x = RANDOM_GENERATOR.uniform(0, 1, (4, 10, 10, 1))
    y = model.generate(x)
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 5
    assert y.shape[2] == x.shape[2] * 5
    assert y.shape[3] == x.shape[3]


def test_solar_custom_loss():
    """Test custom solar loss with only disc and content over daylight hours"""
    handler = DataHandlerH5SolarCC(
        pytest.FP_NSRDB,
        FEATURES_S,
        target=TARGET_S,
        shape=SHAPE,
        time_slice=slice(None, None, 2),
        time_roll=-7,
    )

    batcher = BatchHandlerCC(
        [handler],
        [],
        batch_size=1,
        n_batches=1,
        s_enhance=1,
        t_enhance=8,
        sample_shape=(5, 5, 24),
        feature_sets={
            'lr_features': FEATURES_S,
            'hr_out_features': ['clearsky_ratio'],
        },
    )

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_1x_8x_1f.json')
    fp_disc = pytest.ST_FP_DISC

    Sup3rGan.seed()
    model = SolarCC(
        fp_gen, fp_disc, learning_rate=1e-4, loss='MeanAbsoluteError'
    )

    with tempfile.TemporaryDirectory() as td:
        model.train(
            batcher,
            input_resolution={'spatial': '4km', 'temporal': '40min'},
            n_epoch=1,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        shape = (1, 4, 4, 72, 1)
        hi_res_gen = RANDOM_GENERATOR.uniform(0, 1, shape).astype(np.float32)
        hi_res_true = RANDOM_GENERATOR.uniform(0, 1, shape).astype(np.float32)

        # hi res true and gen shapes need to match
        with pytest.raises(RuntimeError):
            loss1, _ = model.calc_loss(
                RANDOM_GENERATOR.uniform(0, 1, (1, 5, 5, 24, 1)).astype(
                    np.float32
                ),
                RANDOM_GENERATOR.uniform(0, 1, (1, 10, 10, 24, 1)).astype(
                    np.float32
                ),
            )

        # time steps need to be multiple of 24
        with pytest.raises(AssertionError):
            loss1, _ = model.calc_loss(
                RANDOM_GENERATOR.uniform(0, 1, (1, 5, 5, 20, 1)).astype(
                    np.float32
                ),
                RANDOM_GENERATOR.uniform(0, 1, (1, 5, 5, 20, 1)).astype(
                    np.float32
                ),
            )

        loss1, _ = model.calc_loss(
            hi_res_true, hi_res_gen, weight_gen_advers=0.0
        )

        t_len = hi_res_true.shape[3]
        n_days = int(t_len // 24)
        day_slices = [
            slice(
                SolarCC.STARTING_HOUR + x,
                SolarCC.STARTING_HOUR + x + SolarCC.DAYLIGHT_HOURS,
            )
            for x in range(0, 24 * n_days, 24)
        ]

        for tslice in day_slices:
            hi_res_gen[:, :, :, tslice, :] = hi_res_true[:, :, :, tslice, :]

        loss2, _ = model.calc_loss(
            hi_res_true, hi_res_gen, weight_gen_advers=0.0
        )

        assert loss1 > loss2
