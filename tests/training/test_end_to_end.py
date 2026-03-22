"""Test data loading, extraction, batch building, and training workflows."""

import os
from tempfile import TemporaryDirectory

import pytest

from sup3r.models import Sup3rGan
from sup3r.preprocessing import BatchHandler, DataHandler

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']
target = (39.01, -105.15)
shape = (20, 20)
kwargs = {
    'target': target,
    'shape': shape,
    'max_delta': 20,
    'time_slice': slice(None, None, 1),
}


def test_end_to_end():
    """Test data loading, extraction to h5 files with chunks, batch building,
    and training with validation end to end workflow."""

    derive_features = ['u_100m', 'v_100m']

    with TemporaryDirectory() as td:
        train_cache_pattern = os.path.join(td, 'train_{feature}.h5')
        val_cache_pattern = os.path.join(td, 'val_{feature}.h5')
        # get training data
        train_dh = DataHandler(
            pytest.FPS_WTK[0],
            features=derive_features,
            **kwargs,
            cache_kwargs={
                'cache_pattern': train_cache_pattern,
                'max_workers': 1,
                'chunks': {'u_100m': (50, 20, 20), 'v_100m': (50, 20, 20)},
            },
        )
        # get val data
        val_dh = DataHandler(
            pytest.FPS_WTK[1],
            features=derive_features,
            **kwargs,
            cache_kwargs={
                'cache_pattern': val_cache_pattern,
                'max_workers': 1,
                'chunks': {'u_100m': (50, 20, 20), 'v_100m': (50, 20, 20)},
            },
        )

        means = os.path.join(td, 'means.json')
        stds = os.path.join(td, 'stds.json')

        batcher = BatchHandler(
            train_containers=[train_dh],
            val_containers=[val_dh],
            n_batches=2,
            batch_size=4,
            sample_shape=(12, 12, 16),
            s_enhance=3,
            t_enhance=4,
            means=means,
            stds=stds,
        )

        Sup3rGan.seed()
        model = Sup3rGan(
            pytest.ST_FP_GEN,
            pytest.ST_FP_DISC,
            learning_rate=2e-5,
            loss='MeanAbsoluteError',
        )
        model.train(
            batcher,
            input_resolution={'spatial': '30km', 'temporal': '60min'},
            n_epoch=2,
            weight_gen_advers=0.01,
            train_gen=True,
            train_disc=True,
            checkpoint_int=10,
            out_dir=os.path.join(td, 'test_{epoch}'),
        )
