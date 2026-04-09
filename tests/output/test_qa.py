"""pytests for sup3r QA module"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from rex import Resource

from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.qa.qa import Sup3rQa
from sup3r.qa.utilities import (
    continuous_dist,
    direct_dist,
    frequency_spectrum,
    gradient_dist,
    time_derivative_dist,
    tke_frequency_spectrum,
    tke_wavenumber_spectrum,
    wavenumber_spectrum,
)
from sup3r.utilities.pytest.helpers import make_fake_nc_file
from sup3r.utilities.utilities import RANDOM_GENERATOR

TRAIN_FEATURES = ['u_100m', 'v_100m', 'pressure_0m']
MODEL_OUT_FEATURES = ['u_100m', 'v_100m']
FOUT_FEATURES = ['windspeed_100m', 'winddirection_100m']
TARGET = (19.3, -123.5)
SHAPE = (8, 8)
TEMPORAL_SLICE = slice(None, None, 1)
FWP_CHUNK_SHAPE = (8, 8, 8)
S_ENHANCE = 3
T_ENHANCE = 4


@pytest.fixture(scope='module')
def input_files(tmpdir_factory):
    """Dummy netcdf input files for qa testing"""

    input_file = str(tmpdir_factory.mktemp('data').join('qa_input.nc'))
    make_fake_nc_file(input_file, shape=(100, 100, 8), features=TRAIN_FEATURES)
    return input_file


@pytest.mark.parametrize('ext', ['nc', 'h5'])
def test_qa(input_files, ext):
    """Test QA module for fwp output to NETCDF and H5 files."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(TRAIN_FEATURES))))
    model.set_model_params(
        lr_features=TRAIN_FEATURES,
        hr_out_features=MODEL_OUT_FEATURES,
        s_enhance=3,
        t_enhance=4,
        input_resolution={'spatial': '12km', 'temporal': '60min'},
    )
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        input_handler_kwargs = {
            'target': TARGET,
            'shape': SHAPE,
            'time_slice': TEMPORAL_SLICE,
        }

        out_files = os.path.join(td, 'out_{file_id}.' + ext)
        strategy = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=FWP_CHUNK_SHAPE,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs.copy(),
            out_pattern=out_files,
            max_nodes=1,
            invert_uv=False
        )

        forward_pass = ForwardPass(strategy)
        forward_pass.run(strategy, node_index=0)

        assert len(strategy.out_files) == 1

        args = [input_files, strategy.out_files[0]]
        qa_fp = os.path.join(td, 'qa.h5')
        kwargs = {
            's_enhance': S_ENHANCE,
            't_enhance': T_ENHANCE,
            'temporal_coarsening_method': 'subsample',
            'qa_fp': qa_fp,
            'save_sources': True,
            'input_handler_kwargs': input_handler_kwargs,
        }
        with Sup3rQa(*args, **kwargs) as qa:
            data = qa.get_dset_out(qa.features[0])
            data = qa.coarsen_data(0, qa.features[0], data)

            assert isinstance(qa.meta, pd.DataFrame)
            assert isinstance(qa.time_index, pd.DatetimeIndex)
            for i in range(3):
                assert data.shape[i] == qa.input_handler.data.shape[i]

            qa.run()

            assert os.path.exists(qa_fp)

            with Resource(qa_fp) as qa_out:
                for dset in qa.features:
                    idf = qa.input_handler.features.index(dset.lower())
                    qa_true = qa_out[dset + '_true'].flatten()
                    qa_syn = qa_out[dset + '_synthetic'].flatten()
                    qa_diff = qa_out[dset + '_error'].flatten()

                    wtk_source = qa.input_handler.data[dset][...]
                    wtk_source = np.asarray(wtk_source).transpose(2, 0, 1)

                    wtk_source = wtk_source.flatten()

                    fwp_data = (
                        qa.output_handler[dset].values
                        if ext == 'nc'
                        else qa.output_handler[dset][...]
                    )

                    if ext == 'h5':
                        shape = (
                            qa.input_handler.shape[2] * T_ENHANCE,
                            qa.input_handler.shape[0] * S_ENHANCE,
                            qa.input_handler.shape[1] * S_ENHANCE,
                        )
                        fwp_data = fwp_data.reshape(shape)

                    fwp_data = np.transpose(fwp_data, axes=(1, 2, 0))
                    fwp_data = qa.coarsen_data(idf, dset, fwp_data)
                    fwp_data = np.transpose(fwp_data, axes=(2, 0, 1))
                    fwp_data = fwp_data.flatten()

                    test_diff = fwp_data - wtk_source

                    assert np.allclose(qa_true, wtk_source, atol=0.01)
                    assert np.allclose(qa_syn, fwp_data, atol=0.01)
                    assert np.allclose(test_diff, qa_diff, atol=0.01)


def test_continuous_dist():
    """Test distribution interpolation function"""

    a = np.linspace(-6, 6, 10)
    counts, centers = continuous_dist(a, bins=20, range=[-10, 10])
    assert not all(np.isnan(counts))
    assert centers[0] < -9.0
    assert centers[-1] > 9.0


@pytest.mark.parametrize(
    'func', [direct_dist, gradient_dist, time_derivative_dist]
)
def test_dist_smoke(func):
    """Test QA dist functions for basic operations."""

    a = RANDOM_GENERATOR.random((10, 10))
    _ = func(a)


@pytest.mark.parametrize(
    'func', [tke_frequency_spectrum, tke_wavenumber_spectrum]
)
def test_uv_spectrum_smoke(func):
    """Test QA uv spectrum functions for basic operations."""

    u = RANDOM_GENERATOR.random((10, 10))
    v = RANDOM_GENERATOR.random((10, 10))
    _ = func(u, v)


@pytest.mark.parametrize('func', [frequency_spectrum, wavenumber_spectrum])
def test_spectrum_smoke(func):
    """Test QA spectrum functions for basic operations."""

    ke = RANDOM_GENERATOR.random((10, 10))
    _ = func(ke)
