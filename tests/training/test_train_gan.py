"""Test the basic training of super resolution GAN"""

import json
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework.errors_impl import InvalidArgumentError

from sup3r.models import Sup3rGan
from sup3r.preprocessing import BatchHandler, DataHandler

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']


def _get_handlers():
    """Initialize training and validation handlers used across tests."""

    kwargs = {
        'file_paths': pytest.FP_WTK,
        'features': FEATURES,
        'target': TARGET_COORD,
        'shape': (20, 20),
    }
    train_handler = DataHandler(
        **kwargs,
        time_slice=slice(1000, None, 1),
    )

    val_handler = DataHandler(
        **kwargs,
        time_slice=slice(None, 1000, 1),
    )

    return train_handler, val_handler


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16)),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_train_disc(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=8
):
    """Test that the discriminator is trained whenever loss is outside given
    bounds"""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        bh_kwargs = {
            'train_containers': [train_handler],
            'val_containers': [val_handler],
            'sample_shape': sample_shape,
            'batch_size': 15,
            's_enhance': s_enhance,
            't_enhance': t_enhance,
            'n_batches': 1,
            'means': None,
            'stds': None,
            'mode': 'eager',
        }
        batch_handler = BatchHandler(**bh_kwargs)

        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0.0,
            'train_gen': True,
            'train_disc': True,
            'disc_loss_bounds': [-np.inf, 0.0],
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        assert all(model.history['disc_train_frac'] == 1)

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        batch_handler = BatchHandler(**bh_kwargs)

        loaded.train(batch_handler, **model_kwargs)
        assert all(loaded.history['disc_train_frac'] == 1)

        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        batch_handler = BatchHandler(**bh_kwargs)

        loaded.train(batch_handler, **model_kwargs)
        assert all(loaded.history['disc_train_frac'] == 1)


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16)),
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_train(fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=3):
    """Test basic model training with only gen content loss. Tests both
    spatiotemporal and spatial models."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen,
        fp_disc,
        learning_rate=lr,
        loss={'MeanAbsoluteError': {}, 'MeanSquaredError': {}},
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        # stats will be calculated since they are given as None
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=3,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=3,
            means=None,
            stds=None,
        )

        assert batch_handler.means is not None
        assert batch_handler.stds is not None

        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        assert 'config_generator' in model.meta
        assert 'config_discriminator' in model.meta
        assert len(model.history) == n_epoch
        assert all(model.history['gen_train_frac'] == 1)
        assert all(model.history['disc_train_frac'] == 0)
        tlossg = model.history['train_loss_gen'].values
        vlossg = model.history['val_loss_gen'].values
        assert not np.isnan(tlossg).any()
        assert not np.isnan(vlossg).any()
        assert 'test_0' in os.listdir(td)
        assert 'test_1' in os.listdir(td)
        assert 'model_gen.pkl' in os.listdir(td + '/test_1')
        assert 'model_disc.pkl' in os.listdir(td + '/test_1')
        assert 'train_mean_absolute_error' in model.history
        assert 'train_mean_squared_error' in model.history
        assert 'val_mean_absolute_error' in model.history
        assert 'val_mean_squared_error' in model.history

        # test save/load functionality
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        loaded = model.load(out_dir)

        with open(os.path.join(out_dir, 'model_params.json')) as f:
            model_params = json.load(f)

        assert np.allclose(model_params['optimizer']['learning_rate'], lr)
        assert np.allclose(model_params['optimizer_disc']['learning_rate'], lr)
        assert 'OptmGen/learning_rate' in model.history
        assert 'OptmDisc/learning_rate' in model.history

        msg = (
            'Could not find OptmGen states in columns: '
            f'{sorted(model.history.columns)}'
        )
        check = [
            col.startswith('OptmGen') for col in model.history.columns
        ]
        assert any(check), msg

        assert 'config_generator' in loaded.meta
        assert 'config_discriminator' in loaded.meta
        assert model.meta['class'] == 'Sup3rGan'

        # make an un-trained dummy model
        dummy = Sup3rGan(
            fp_gen,
            fp_disc,
            learning_rate=lr,
            loss={'MeanAbsoluteError': {}, 'MeanSquaredError': {}},
        )
        dummy.set_model_params(
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            batch_handler=batch_handler,
        )

        for batch in batch_handler:
            out_og = model._tf_generate(batch.low_res)
            out_dummy = dummy._tf_generate(batch.low_res)
            out_loaded = loaded._tf_generate(batch.low_res)

            # make sure the loaded model generates the same data as the saved
            # model but different than the dummy

            tf.assert_equal(out_og, out_loaded)
            with pytest.raises(InvalidArgumentError):
                tf.assert_equal(out_og, out_dummy)

            # make sure the trained model has less loss than dummy
            loss_og = model.calc_loss(batch.high_res, out_og)[0]
            loss_dummy = dummy.calc_loss(batch.high_res, out_dummy)[0]
            assert loss_og.numpy() < loss_dummy.numpy()

        # test that a new shape can be passed through the generator
        if model.is_5d:
            test_data = np.ones(
                (3, 10, 10, 4, len(FEATURES)), dtype=np.float32
            )
            y_test = model._tf_generate(test_data)
            assert y_test.shape[3] == test_data.shape[3] * t_enhance

        else:
            test_data = np.ones((3, 10, 10, len(FEATURES)), dtype=np.float32)
            y_test = model._tf_generate(test_data)

        assert y_test.shape[0] == test_data.shape[0]
        assert y_test.shape[1] == test_data.shape[1] * s_enhance
        assert y_test.shape[2] == test_data.shape[2] * s_enhance
        assert y_test.shape[-1] == test_data.shape[-1]

        batch_handler.stop()


@pytest.mark.parametrize(
    'loss_func',
    [
        {'SlicedWassersteinLoss': {}},
        {
            'GeothermalObsLoss': {
                'gen_features': ['u_100m'],
                'true_features': ['u_100m'],
            }
        },
    ],
)
def test_train_with_custom_loss(loss_func, n_epoch=8):
    """Test basic model training with custom losses."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        pytest.ST_FP_GEN,
        pytest.ST_FP_DISC,
        learning_rate=lr,
        loss=loss_func,
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        # stats will be calculated since they are given as None
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=(12, 12, 16),
            batch_size=15,
            s_enhance=3,
            t_enhance=4,
            n_batches=10,
            means=None,
            stds=None,
        )

        model_kwargs = {
            'input_resolution': {'spatial': '30km', 'temporal': '60min'},
            'n_epoch': n_epoch,
            'weight_gen_advers': 0,
            'train_gen': True,
            'train_disc': False,
            'checkpoint_int': 1,
            'out_dir': os.path.join(td, 'test_{epoch}'),
        }

        model.train(batch_handler, **model_kwargs)

        tlossg = model.history['train_loss_gen'].values
        vlossg = model.history['val_loss_gen'].values
        assert not np.isnan(tlossg).any()
        assert not np.isnan(vlossg).any()

        batch_handler.stop()


def test_train_st_weight_update(n_epoch=2):
    """Test basic spatiotemporal model training with discriminators and
    adversarial loss updating."""

    Sup3rGan.seed()
    model = Sup3rGan(
        pytest.ST_FP_GEN,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        learning_rate_disc=4e-4,
    )

    train_handler, val_handler = _get_handlers()

    batch_handler = BatchHandler(
        [train_handler],
        [val_handler],
        batch_size=2,
        s_enhance=3,
        t_enhance=4,
        n_batches=2,
        sample_shape=(12, 12, 16),
    )

    adaptive_update_bounds = (0.9, 0.99)
    with tempfile.TemporaryDirectory() as td:
        model.train(
            batch_handler,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=1e-6,
            train_gen=True,
            train_disc=True,
            checkpoint_int=10,
            out_dir=os.path.join(td, 'test_{epoch}'),
            adaptive_update_bounds=adaptive_update_bounds,
            adaptive_update_fraction=0.05,
        )

        # check that weight is changed
        check_lower = any(
            frac < adaptive_update_bounds[0]
            for frac in model.history['disc_train_frac'][:-1]
        )
        check_higher = any(
            frac > adaptive_update_bounds[1]
            for frac in model.history['disc_train_frac'][:-1]
        )
        assert check_lower or check_higher
        for e in range(0, n_epoch - 1):
            weight_old = model.history['weight_gen_advers'][e]
            weight_new = model.history['weight_gen_advers'][e + 1]
            if model.history['disc_train_frac'][e] < adaptive_update_bounds[0]:
                assert weight_new > weight_old
            if model.history['disc_train_frac'][e] > adaptive_update_bounds[1]:
                assert weight_new < weight_old

        batch_handler.stop()


def test_optimizer_update():
    """Test updating optimizer method."""

    Sup3rGan.seed()
    model = Sup3rGan(
        pytest.ST_FP_GEN,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        learning_rate_disc=4e-4,
    )

    assert model.optimizer.learning_rate == 1e-4
    assert model.optimizer_disc.learning_rate == 4e-4

    model.update_optimizer(option='generator', learning_rate=2.0)

    assert model.optimizer.learning_rate == 2
    assert model.optimizer_disc.learning_rate == 4e-4

    model.update_optimizer(option='discriminator', learning_rate=0.4)

    assert model.optimizer.learning_rate == 2
    assert model.optimizer_disc.learning_rate == 0.4

    model.update_optimizer(option='all', learning_rate=0.1)

    assert model.optimizer.learning_rate == 0.1
    assert model.optimizer_disc.learning_rate == 0.1


def test_run_gradient_descent_multi_gpu_dispatch(monkeypatch):
    """Test that multi_gpu execution uses the strategy-backed path."""

    class FakeStrategy:
        num_replicas_in_sync = 2

    model = Sup3rGan(
        pytest.S_FP_GEN,
        pytest.S_FP_DISC,
        learning_rate=1e-4,
    )
    model._strategy = FakeStrategy()
    model._multi_gpu = True

    called = {}

    def fake_get_train_fns(train_gen=True, train_disc=False):
        called['train_flags'] = (train_gen, train_disc)
        return 'grad_fn', 'apply_fn'

    def fake_run_mirrored_grad(
        low_res,
        hi_res_true,
        grad_fn,
        apply_fn,
        **calc_loss_kwargs,
    ):
        called['mirrored'] = {
            'grad_fn': grad_fn,
            'apply_fn': apply_fn,
            'kwargs': calc_loss_kwargs,
            'low_res_shape': low_res.shape,
            'hi_res_shape': hi_res_true.shape,
        }
        return {'loss_gen': tf.constant(0.0)}

    monkeypatch.setattr(model, '_get_train_fns', fake_get_train_fns)
    monkeypatch.setattr(model, '_run_mirrored_grad', fake_run_mirrored_grad)
    monkeypatch.setattr(
        model,
        '_run_serial_grad',
        lambda *args, **kwargs: pytest.fail('serial path should not run'),
    )

    low_res = np.ones((2, 4, 4, len(FEATURES)), dtype=np.float32)
    hi_res = np.ones((2, 8, 8, len(FEATURES)), dtype=np.float32)
    loss_details = model.run_gradient_descent(
        low_res,
        hi_res,
        train_gen=True,
        train_disc=False,
        multi_gpu=True,
        weight_gen_advers=0.0,
    )

    assert called['train_flags'] == (True, False)
    assert called['mirrored']['grad_fn'] == 'grad_fn'
    assert called['mirrored']['apply_fn'] == 'apply_fn'
    assert called['mirrored']['kwargs']['weight_gen_advers'] == 0.0
    assert called['mirrored']['low_res_shape'] == low_res.shape
    assert called['mirrored']['hi_res_shape'] == hi_res.shape
    assert float(loss_details['loss_gen']) == 0.0


def test_input_res_check():
    """Make sure error is raised for invalid input resolution"""

    Sup3rGan.seed()
    model = Sup3rGan(
        pytest.ST_FP_GEN,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        learning_rate_disc=4e-4,
    )

    with pytest.raises(RuntimeError):
        model.set_model_params(
            input_resolution={'spatial': '22km', 'temporal': '9min'},
            s_enhance=3,
            t_enhance=4,
        )


def test_enhancement_check():
    """Make sure error is raised for invalid enhancement factor inputs"""

    Sup3rGan.seed()
    model = Sup3rGan(
        pytest.ST_FP_GEN,
        pytest.ST_FP_DISC,
        learning_rate=1e-4,
        learning_rate_disc=4e-4,
    )

    with pytest.raises(RuntimeError):
        model.set_model_params(
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            s_enhance=7,
            t_enhance=3,
        )
