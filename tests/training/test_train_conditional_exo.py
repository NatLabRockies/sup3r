"""Test the training of conditional moment estimation models with exogenous
inputs."""

import os
import tempfile

import pytest

from sup3r import CONFIG_DIR
from sup3r.models import Sup3rCondMom
from sup3r.preprocessing import (
    BatchHandlerMom1,
    BatchHandlerMom1SF,
    BatchHandlerMom2,
    BatchHandlerMom2Sep,
    BatchHandlerMom2SepSF,
    BatchHandlerMom2SF,
    DataHandler,
)

SHAPE = (20, 20)
TARGET_COORD = (39.01, -105.15)


def make_s_gen_model(custom_layer):
    """Make simple conditional moment model with flexible custom layer."""
    return [
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 64,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
        {'class': 'SpatialExpansion', 'spatial_mult': 2},
        {'class': 'Activation', 'activation': 'relu'},
        {'class': custom_layer, 'name': 'topography'},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {
            'class': 'Conv2DTranspose',
            'filters': 2,
            'kernel_size': 3,
            'strides': 1,
            'activation': 'relu',
        },
        {'class': 'Cropping2D', 'cropping': 4},
    ]


@pytest.mark.parametrize('batch_class', [BatchHandlerMom1, BatchHandlerMom1SF])
def test_wind_non_cc_hi_res_st_topo_mom1(
    batch_class,
    n_epoch=1,
    n_batches=2,
    batch_size=2,
):
    """Test spatiotemporal first conditional moment for wind model for non cc
    Sup3rConcat layer that concatenates hi-res topography in the middle of
    the network. Test for direct first moment or subfilter velocity."""

    handler = DataHandler(
        pytest.FP_WTK,
        ['u_100m', 'v_100m', 'topography'],
        target=TARGET_COORD,
        shape=SHAPE,
        time_slice=slice(None, None, 1),
    )

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc', 'gen_wind_3x_4x_2f.json')

    Sup3rCondMom.seed()
    model_mom1 = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    batcher = batch_class(
        [handler],
        batch_size=batch_size,
        s_enhance=3,
        t_enhance=4,
        sample_shape=(12, 12, 24),
        lower_models={1: model_mom1},
        n_batches=n_batches,
        feature_sets={'hr_exo_features': ['topography'],
                      'hr_out_features': ['u_100m', 'v_100m']},
    )

    with tempfile.TemporaryDirectory() as td:
        model_mom1.train(
            batcher,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )


@pytest.mark.parametrize(
    'batch_class',
    [
        BatchHandlerMom2,
        BatchHandlerMom2Sep,
        BatchHandlerMom2SF,
        BatchHandlerMom2SepSF,
    ],
)
def test_wind_non_cc_hi_res_st_topo_mom2(
    batch_class,
    n_epoch=1,
    n_batches=2,
    batch_size=2,
):
    """Test spatiotemporal second conditional moment for wind model for non cc
    Sup3rConcat layer that concatenates hi-res topography in the middle of
    the network. Test for direct second moment or subfilter velocity.
    Test for separate or learning coupled with first moment."""

    handler = DataHandler(
        pytest.FP_WTK,
        ['u_100m', 'v_100m', 'topography'],
        target=TARGET_COORD,
        shape=SHAPE,
        time_slice=slice(None, None, 1),
    )

    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc', 'gen_wind_3x_4x_2f.json')

    Sup3rCondMom.seed()
    model_mom1 = Sup3rCondMom(fp_gen, learning_rate=1e-4)
    model_mom2 = Sup3rCondMom(fp_gen, learning_rate=1e-4)

    batcher = batch_class(
        [handler],
        batch_size=batch_size,
        s_enhance=3,
        t_enhance=4,
        lower_models={1: model_mom1},
        n_batches=n_batches,
        sample_shape=(12, 12, 24),
        feature_sets={
            'lr_features': ['u_100m', 'v_100m', 'topography'],
            'hr_out_features': ['u_100m', 'v_100m'],
            'hr_exo_features': ['topography']},
        mode='eager'
    )

    with tempfile.TemporaryDirectory() as td:
        model_mom2.train(
            batcher,
            input_resolution={'spatial': '12km', 'temporal': '60min'},
            n_epoch=n_epoch,
            checkpoint_int=None,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
