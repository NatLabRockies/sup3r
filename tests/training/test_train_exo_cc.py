"""Test the basic training of super resolution GAN for solar climate change
applications"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGan
from sup3r.preprocessing.batch_handlers.factory import BatchHandlerCC
from sup3r.preprocessing.data_handlers.base import DataHandlerH5WindCC
from sup3r.preprocessing.utilities import lowered
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_W = ['temperature_100m', 'u_100m', 'v_100m', 'topography']
TARGET_W = (39.01, -105.15)


@pytest.mark.parametrize(
    ('CustomLayer', 'lr_features', 'hr_out_features'),
    [
        ('Sup3rAdder', FEATURES_W, FEATURES_W[1:-1]),
        ('Sup3rConcat', FEATURES_W, FEATURES_W[1:-1]),
        ('Sup3rAdder', FEATURES_W[1:], FEATURES_W[1:-1]),
        ('Sup3rConcat', FEATURES_W[1:], FEATURES_W[1:-1]),
    ],
)
def test_wind_hi_res_topo(
    CustomLayer, lr_features, hr_out_features, gen_config_with_topo
):
    """Test a special wind cc model with the custom Sup3rAdder or Sup3rConcat
    layer that adds/concatenates hi-res topography in the middle of the
    network. The first two parameter sets include an lr only feature."""

    handler = DataHandlerH5WindCC(
        pytest.FP_WTK,
        features=lr_features,
        target=TARGET_W,
        shape=SHAPE,
        time_slice=slice(None, None, 2),
        time_roll=-7,
    )

    batcher = BatchHandlerCC(
        [handler],
        batch_size=2,
        n_batches=2,
        s_enhance=2,
        sample_shape=(20, 20),
        feature_sets={
            'lr_features': lr_features,
            'hr_out_features': hr_out_features,
            'hr_exo_features': ['topography'],
        },
        mode='eager',
    )

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

        assert model.lr_features == lowered(lr_features)
        assert model.hr_out_features == lowered(hr_out_features)
        assert model.hr_exo_features == ['topography']
        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == lowered(hr_out_features)
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
