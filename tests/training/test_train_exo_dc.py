"""Test the basic training of super resolution GAN for solar climate change
applications"""

import os
import tempfile

import numpy as np
import pytest

from sup3r.models import Sup3rGanDC
from sup3r.preprocessing import DataHandler
from sup3r.utilities.pytest.helpers import BatchHandlerTesterDC
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_W = ['temperature_100m', 'u_100m', 'v_100m', 'topography']
TARGET_W = (39.01, -105.15)


@pytest.mark.parametrize('CustomLayer', ['Sup3rAdder', 'Sup3rConcat'])
def test_wind_dc_hi_res_topo(CustomLayer):
    """Test a special data centric wind model with the custom Sup3rAdder or
    Sup3rConcat layer that adds/concatenates hi-res topography in the middle of
    the network."""

    kwargs = {
        'file_paths': pytest.FP_WTK,
        'features': ('u_100m', 'v_100m', 'topography'),
        'target': TARGET_W,
        'shape': SHAPE,
    }
    handler = DataHandler(**kwargs, time_slice=slice(100, None, 2))
    val_handler = DataHandler(**kwargs, time_slice=slice(None, 100, 2))

    # number of bins conflicts with data shape and sample shape
    with pytest.raises(AssertionError):
        batcher = BatchHandlerTesterDC(
            train_containers=[handler],
            val_containers=[val_handler],
            batch_size=2,
            n_space_bins=4,
            n_time_bins=4,
            n_batches=1,
            s_enhance=2,
            sample_shape=(20, 20, 8),
            feature_sets={
                'lr_features': ['u_100m', 'v_100m', 'topography'],
                'hr_out_features': ['u_100m', 'v_100m'],
                'hr_exo_features': ['topography']},
            )

    batcher = BatchHandlerTesterDC(
        train_containers=[handler],
        val_containers=[val_handler],
        batch_size=2,
        n_space_bins=4,
        n_time_bins=4,
        n_batches=1,
        s_enhance=2,
        sample_shape=(10, 10, 8),
        feature_sets={
            'lr_features': ['u_100m', 'v_100m', 'topography'],
            'hr_out_features': ['u_100m', 'v_100m'],
            'hr_exo_features': ['topography']},
    )

    gen_model = [
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [2, 2], [2, 2], [2, 2], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv3D',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping3D', 'cropping': 1},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv3D',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping3D', 'cropping': 2},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv3D',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping3D', 'cropping': 2},
        {'class': 'SpatioTemporalExpansion', 'spatial_mult': 2},
        {'class': 'Activation', 'activation': 'relu'},
        {'class': CustomLayer, 'name': 'topography'},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv3D',
            'filters': 2,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping3D', 'cropping': 2},
    ]

    Sup3rGanDC.seed()
    model = Sup3rGanDC(gen_model, pytest.ST_FP_DISC, learning_rate=1e-4)

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

        assert 'test_0' in os.listdir(td)
        assert model.meta['hr_out_features'] == ['u_100m', 'v_100m']
        assert model.meta['class'] == 'Sup3rGanDC'
        assert 'topography' in batcher.hr_exo_features
        assert 'topography' not in model.hr_out_features

    x = RANDOM_GENERATOR.uniform(0, 1, (1, 30, 30, 4, 3))
    hi_res_topo = RANDOM_GENERATOR.uniform(0, 1, (1, 60, 60, 4, 1))

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
    assert y.shape[3] == x.shape[3]
    assert y.shape[4] == x.shape[4] - 1
