"""Test basic generator calls with conditional moment estimation models."""

import os

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

TARGET_COORD = (39.01, -105.15)
FEATURES = ['u_100m', 'v_100m']


@pytest.mark.parametrize(
    'bh_class',
    [
        BatchHandlerMom1,
        BatchHandlerMom1SF,
        BatchHandlerMom2,
        BatchHandlerMom2Sep,
        BatchHandlerMom2SepSF,
        BatchHandlerMom2SF,
    ],
)
def test_out_conditional(
    bh_class,
    full_shape=(20, 20),
    sample_shape=(12, 12, 24),
    batch_size=4,
    n_batches=4,
    s_enhance=3,
    t_enhance=4,
    end_t_padding=False,
):
    """Test basic spatiotemporal model outputing for
    first conditional moment."""
    handler = DataHandler(
        pytest.FP_WTK,
        features=FEATURES,
        target=TARGET_COORD,
        shape=full_shape,
        time_slice=slice(None, None, 1),
    )

    fp_gen = os.path.join(
        CONFIG_DIR, 'spatiotemporal', 'gen_3x_4x_2f.json'
    )
    model = Sup3rCondMom(fp_gen)

    batch_handler = bh_class(
        [handler],
        batch_size=batch_size,
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        n_batches=n_batches,
        lower_models={1: model},
        sample_shape=sample_shape,
        end_t_padding=end_t_padding,
        mode='eager'
    )

    # Check sizes
    for batch in batch_handler:
        assert batch.high_res.shape == (
            batch_size,
            sample_shape[0],
            sample_shape[1],
            sample_shape[2],
            2,
        )
        assert batch.output.shape == (
            batch_size,
            sample_shape[0],
            sample_shape[1],
            sample_shape[2],
            2,
        )
        assert batch.low_res.shape == (
            batch_size,
            sample_shape[0] // s_enhance,
            sample_shape[1] // s_enhance,
            sample_shape[2] // t_enhance,
            2,
        )
        out = model._tf_generate(batch.low_res)
        assert out.shape == (
            batch_size,
            sample_shape[0],
            sample_shape[1],
            sample_shape[2],
            2,
        )
    batch_handler.stop()
