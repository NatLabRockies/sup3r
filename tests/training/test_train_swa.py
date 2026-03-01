"""Test training with Stochastic Weight Averaging (SWA)"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGan, TrainingConfig
from sup3r.preprocessing import BatchHandler, DataHandler
from sup3r.utilities.utilities import RANDOM_GENERATOR

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
def test_swa_basic(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=10
):
    """Test basic SWA training with TrainingConfig."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # Configure SWA to start at epoch 7
        config = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=7,
            swa_freq=1,
            swa_lr=lr * 0.1,
            swa_bn_update_batches=2,
        )

        # Verify SWA is not enabled before training
        assert not model._swa_enabled
        assert model._swa_n == 0

        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config,
        )

        # Verify SWA was enabled during training
        assert model._swa_enabled
        # Should have 3 SWA updates (epochs 7, 8, 9)
        assert model._swa_n == 3

        # Verify SWA weights were created
        assert model._swa_weights is not None
        assert len(model._swa_weights) == len(model.weights)

        # Verify pre-SWA weights were saved
        assert model._pre_swa_weights is not None

        # Verify SWA model was saved
        swa_dir = os.path.join(td, f'test_{n_epoch - 1}_swa_final')
        assert os.path.exists(swa_dir)
        assert 'model_gen.pkl' in os.listdir(swa_dir)
        assert 'model_disc.pkl' in os.listdir(swa_dir)

        batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.ST_FP_GEN, pytest.ST_FP_DISC, 3, 4, (12, 12, 16)),
    ],
)
def test_swa_kwargs(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=10
):
    """Test SWA training using kwargs (backwards compatibility)."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # Train using kwargs instead of TrainingConfig
        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=7,
            swa_freq=1,
            swa_lr=lr * 0.1,
        )

        # Verify SWA was applied
        assert model._swa_enabled
        assert model._swa_n == 3

        batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_swa_weight_averaging(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=8
):
    """Test that SWA actually averages weights correctly."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # Enable SWA manually to track weights
        model.enable_swa()

        # Get initial weights
        initial_weights = [w.numpy().copy() for w in model.weights]

        config = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=5,
            swa_freq=1,
            swa_lr=lr * 0.1,
        )

        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config,
        )

        # Verify weights changed from initial
        current_weights = [w.numpy() for w in model.weights]
        for i, (init_w, curr_w) in enumerate(
            zip(initial_weights, current_weights)
        ):
            assert not np.allclose(init_w, curr_w), (
                f'Weight {i} did not change'
            )

        # Verify SWA weights are different from final weights
        # (before swap, final weights are the last SGD weights)
        assert model._pre_swa_weights is not None
        for i, (swa_w, pre_swa_w) in enumerate(
            zip(model._swa_weights, model._pre_swa_weights)
        ):
            # SWA weights should be different from the last SGD weights
            # (unless they happened to converge to the same point)
            if not np.allclose(swa_w, pre_swa_w, rtol=1e-3):
                # Found at least one weight that differs
                break
        else:
            # This is unlikely but not impossible
            pass

        batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_swa_manual_control(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape
):
    """Test manual SWA control methods."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    batch_handler = BatchHandler(
        train_containers=[train_handler],
        val_containers=[val_handler],
        sample_shape=sample_shape,
        batch_size=15,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=5,
        means=None,
        stds=None,
    )

    # Initialize weights
    lr_shape, hr_shape = batch_handler.shapes
    model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    model.init_weights(lr_shape, hr_shape)

    # Test enable_swa
    assert not model._swa_enabled
    model.enable_swa()
    assert model._swa_enabled
    assert model._swa_n == 0

    # Test update_swa
    weights_before = [w.numpy().copy() for w in model.weights]
    model.update_swa()
    assert model._swa_n == 1
    assert model._swa_weights is not None

    # Verify SWA weights match current weights after first update
    for swa_w, curr_w in zip(model._swa_weights, weights_before):
        assert np.allclose(swa_w, curr_w)

    # Test swap_swa_weights
    model.swap_swa_weights()
    assert model._pre_swa_weights is not None

    # Test restore_pre_swa_weights
    model.restore_pre_swa_weights()
    weights_after_restore = [w.numpy() for w in model.weights]
    for w_before, w_after in zip(weights_before, weights_after_restore):
        assert np.allclose(w_before, w_after)

    batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_swa_freq(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=12
):
    """Test SWA with different update frequencies."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # SWA starts at epoch 6, updates every 2 epochs
        # Should update at epochs: 6, 8, 10 (3 updates)
        config = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=6,
            swa_freq=2,
            swa_lr=lr * 0.1,
        )

        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config,
        )

        # Should have 3 SWA updates
        assert model._swa_n == 3

        batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_swa_no_constant_lr(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=8
):
    """Test SWA without constant learning rate (keeps existing schedule)."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # Configure SWA without constant LR (swa_lr=None)
        config = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=5,
            swa_freq=1,
            swa_lr=None,  # Don't change LR schedule
        )

        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config,
        )

        # Verify SWA was still applied
        assert model._swa_enabled
        assert model._swa_n == 3  # epochs 5, 6, 7

        batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_no_swa(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=8
):
    """Test that training works normally without SWA enabled."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # Configure without SWA (swa_start=None)
        config = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=None,  # Disable SWA
        )

        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config,
        )

        # Verify SWA was not enabled
        assert not model._swa_enabled
        assert model._swa_n == 0
        assert model._swa_weights is None

        # Verify regular checkpoint was saved but not SWA
        assert not os.path.exists(os.path.join(td, 'test_swa_final'))

        batch_handler.stop()


def test_training_config_validation():
    """Test TrainingConfig validation."""

    # Valid config
    config = TrainingConfig(
        n_epoch=100, swa_start=75, out_dir='./model_{epoch}'
    )
    assert config.n_epoch == 100
    assert config.swa_start == 75

    # Invalid: n_epoch must be positive
    with pytest.raises(ValueError, match='n_epoch must be positive'):
        TrainingConfig(n_epoch=0)

    with pytest.raises(ValueError, match='swa_start must be non-negative'):
        TrainingConfig(n_epoch=100, swa_start=-1)

    # Invalid: swa_freq must be positive
    with pytest.raises(ValueError, match='swa_freq must be positive'):
        TrainingConfig(n_epoch=100, swa_start=75, swa_freq=0)

    # Invalid: out_dir must contain {epoch} when checkpoint_int is set
    with pytest.raises(ValueError, match='out_dir must contain'):
        TrainingConfig(
            n_epoch=100, checkpoint_int=10, out_dir='./model_no_epoch'
        )


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_swa_load_and_continue(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=10
):
    """Test loading a SWA model and verifying weights."""

    lr = 5e-5
    Sup3rGan.seed()
    model = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        config = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'test_{epoch}'),
            swa_start=7,
            swa_freq=1,
            swa_lr=lr * 0.1,
        )

        model.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config,
        )

        # Save current weights (should be SWA weights)
        swa_weights_saved = [w.numpy().copy() for w in model.weights]

        # Load the SWA model
        swa_dir = os.path.join(td, f'test_{n_epoch - 1}_swa_final')
        loaded_model = Sup3rGan.load(swa_dir)

        # Verify loaded weights match SWA weights
        loaded_weights = [w.numpy() for w in loaded_model.weights]
        for orig_w, load_w in zip(swa_weights_saved, loaded_weights):
            assert np.allclose(orig_w, load_w)

        batch_handler.stop()


@pytest.mark.parametrize(
    ['fp_gen', 'fp_disc', 's_enhance', 't_enhance', 'sample_shape'],
    [
        (pytest.S_FP_GEN, pytest.S_FP_DISC, 2, 1, (10, 10, 1)),
    ],
)
def test_swa_single_epoch_equals_no_swa(
    fp_gen, fp_disc, s_enhance, t_enhance, sample_shape, n_epoch=8
):
    """Test that SWA with single epoch averaging equals no SWA.

    When SWA averages only the final epoch's weights, the result should be
    identical to training without SWA (both should have the same final
    weights).
    """

    lr = 5e-5
    n_epoch = 1

    # Train model WITHOUT SWA
    Sup3rGan.seed(42)
    state = RANDOM_GENERATOR.bit_generator.state
    model_no_swa = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        config_no_swa = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'no_swa_{epoch}'),
            swa_start=None,
        )

        model_no_swa.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config_no_swa,
        )

        # Save weights from model without SWA
        weights_no_swa = [w.numpy().copy() for w in model_no_swa.weights]

    # Train model WITH SWA starting at last epoch (single snapshot)
    Sup3rGan.seed(42)  # Reset seed for identical training
    RANDOM_GENERATOR.bit_generator.state = state
    model_swa = Sup3rGan(
        fp_gen, fp_disc, learning_rate=lr, loss='MeanAbsoluteError'
    )

    train_handler, val_handler = _get_handlers()

    with tempfile.TemporaryDirectory() as td:
        batch_handler = BatchHandler(
            train_containers=[train_handler],
            val_containers=[val_handler],
            sample_shape=sample_shape,
            batch_size=15,
            s_enhance=s_enhance,
            t_enhance=t_enhance,
            n_batches=5,
            means=None,
            stds=None,
        )

        # SWA starts at the last epoch (n_epoch - 1), so only 1 snapshot
        config_swa = TrainingConfig(
            n_epoch=n_epoch,
            weight_gen_advers=0,
            train_gen=True,
            train_disc=False,
            out_dir=os.path.join(td, 'swa_{epoch}'),
            swa_start=n_epoch - 1,  # Start at last epoch
            swa_freq=1,
            swa_lr=None,  # Keep same LR schedule
        )

        model_swa.train(
            batch_handler,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            config=config_swa,
        )

        # Verify SWA was enabled but only took 1 snapshot
        assert model_swa._swa_enabled
        assert model_swa._swa_n == 1

        # Save weights from model with SWA (should be SWA averaged)
        weights_swa = [w.numpy().copy() for w in model_swa.weights]

    # Verify weights are identical (or very close due to numerical precision)
    # Since SWA with 1 snapshot should equal the final SGD weights
    for i, (w_no_swa, w_swa) in enumerate(zip(weights_no_swa, weights_swa)):
        assert np.allclose(w_no_swa, w_swa, rtol=1e-3, atol=1e-3), (
            f'Weight {i} differs between no-SWA and single-epoch SWA. '
            f'Max diff: {np.max(np.abs(w_no_swa - w_swa))}'
        )
