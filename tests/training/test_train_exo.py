"""Test the training of super resolution GANs with exo data."""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGan
from sup3r.preprocessing import (
    BatchHandler,
    DataHandler,
)
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_W = ['temperature_100m', 'u_100m', 'v_100m', 'topography']
TARGET_W = (39.01, -105.15)


@pytest.mark.parametrize(
    ('CustomLayer', 'lr_features', 'hr_out_features', 'mode'),
    [
        ('Sup3rAdder', FEATURES_W, FEATURES_W[1:-1], 'lazy'),
        ('Sup3rConcat', FEATURES_W, FEATURES_W[1:-1], 'lazy'),
        ('Sup3rAdder', FEATURES_W[1:], FEATURES_W[1:-1], 'lazy'),
        ('Sup3rConcat', FEATURES_W[1:], FEATURES_W[1:-1], 'lazy'),
        ('Sup3rConcat', FEATURES_W[1:], FEATURES_W[1:-1], 'eager'),
    ],
)
def test_wind_hi_res_topo(
    CustomLayer, lr_features, hr_out_features, mode, gen_config_with_topo
):
    """Test a special wind model for non cc with the custom Sup3rAdder or
    Sup3rConcat layer that adds/concatenates hi-res topography in the middle of
    the network."""
    kwargs = {
        'file_paths': pytest.FP_WTK,
        'features': lr_features,
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
        feature_sets={
            'lr_features': lr_features,
            'hr_out_features': hr_out_features,
            'hr_exo_features': ['topography'],
        },
        mode=mode,
    )

    if mode == 'eager':
        assert batcher.loaded

    Sup3rGan.seed()
    model = Sup3rGan(
        gen_config_with_topo(CustomLayer), pytest.S_FP_DISC, learning_rate=1e-4
    )

    with tempfile.TemporaryDirectory() as td:
        model.train(
            batcher,
            input_resolution={'spatial': '16km', 'temporal': '3600min'},
            n_epoch=1,
            weight_gen_advers=0.0,
            train_gen=True,
            train_disc=False,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )

        assert model.lr_features == [f.lower() for f in lr_features]
        assert model.hr_out_features == [f.lower() for f in hr_out_features]
        assert model.hr_exo_features == ['topography']
        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == [
            f.lower() for f in hr_out_features
        ]
        assert model.meta['class'] == 'Sup3rGan'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = RANDOM_GENERATOR.uniform(0, 1, (4, 30, 30, len(lr_features)))
    hi_res_topo = RANDOM_GENERATOR.uniform(0, 1, (4, 60, 60, 1))

    with pytest.raises(RuntimeError):
        y = model.generate(x, exogenous_data=None)

    exo_tmp = {
        'topography': {
            'steps': [
                {'model': 0, 'combine_type': 'layer', 'data': hi_res_topo}
            ]
        }
    }
    y = model.generate(x, exogenous_data=exo_tmp)

    assert y.dtype == np.float32
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == x.shape[1] * 2
    assert y.shape[2] == x.shape[2] * 2
    assert y.shape[3] == len(hr_out_features)
