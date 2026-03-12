"""Test the training of super resolution GANs with exogenous observation
data."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from rex import Outputs

from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.utilities.pytest.helpers import make_fake_dset
from sup3r.utilities.utilities import RANDOM_GENERATOR

SHAPE = (20, 20)
FEATURES_W = ['u_10m', 'v_10m']
TARGET_W = (39.01, -105.15)

target = (19.3, -123.5)
shape = (8, 8)
sample_shape = (8, 8, 6)
time_slice = slice(None, None, 1)
list_chunk_size = 10
fwp_chunk_shape = (4, 4, 150)
s_enhance = 3
t_enhance = 4


@pytest.fixture(scope='module')
def input_file(tmpdir_factory):
    """Dummy input for :class:`ForwardPass`"""

    input_file = str(tmpdir_factory.mktemp('data').join('fwp_input.nc'))
    dset = make_fake_dset(
        shape=(100, 100, 8),
        features=['u_10m', 'v_10m'],
    )
    dset = dset.coarsen(west_east=2, south_north=2).mean()
    dset.to_netcdf(input_file)
    return input_file


@pytest.fixture(scope='module')
def nc_obs_file(tmpdir_factory):
    """Dummy observation data saved to netcdf file"""
    obs_file = str(tmpdir_factory.mktemp('data').join('fwp_obs.nc'))
    dset = make_fake_dset(
        shape=(100, 100, 20),
        features=['u_10m', 'v_10m'],
    )

    mask = RANDOM_GENERATOR.choice(
        [True, False], dset['u_10m'].shape, p=[0.9, 0.1]
    )
    u_10m = dset['u_10m'].values
    v_10m = dset['v_10m'].values
    u_10m[mask] = np.nan
    v_10m[mask] = np.nan
    dset['u_10m'] = (dset['u_10m'].dims, u_10m)
    dset['v_10m'] = (dset['v_10m'].dims, v_10m)
    dset.to_netcdf(obs_file)

    return obs_file


@pytest.fixture(scope='module')
def h5_obs_file(tmpdir_factory):
    """Dummy observation data, flattened and sparsified and saved to h5"""
    obs_file = str(tmpdir_factory.mktemp('data').join('fwp_obs.h5'))
    dset = make_fake_dset(
        shape=(100, 100, 20),
        features=['u_10m', 'v_10m'],
    )

    mask = RANDOM_GENERATOR.choice(
        [True, False], dset.latitude.values.shape, p=[0.95, 0.05]
    )
    lats = dset.latitude.values[~mask].flatten()
    lons = dset.longitude.values[~mask].flatten()
    flat_shape = (len(dset.time), len(lats))
    u_10m = dset['u_10m'].values[:, ~mask].reshape(flat_shape)
    v_10m = dset['v_10m'].values[:, ~mask].reshape(flat_shape)

    meta = pd.DataFrame({'latitude': lats, 'longitude': lons})

    shapes = {'u_10m': flat_shape, 'v_10m': flat_shape}
    attrs = {'u_10m': None, 'v_10m': None}
    chunks = {'u_10m': None, 'v_10m': None}
    dtypes = {'u_10m': 'float32', 'v_10m': 'float32'}

    Outputs.init_h5(
        obs_file,
        ['u_10m', 'v_10m'],
        shapes,
        attrs,
        chunks,
        dtypes,
        meta=meta,
        time_index=pd.DatetimeIndex(dset.time),
    )
    with Outputs(obs_file, 'a') as out:
        out['u_10m'] = u_10m
        out['v_10m'] = v_10m

    return obs_file


@pytest.mark.parametrize('obs_file', ['nc_obs_file', 'h5_obs_file'])
def test_fwp_with_obs(input_file, obs_file, gen_config_with_obs_2d, request):
    """Test a special model trained to condition output on input
    observations."""

    obs_file = request.getfixturevalue(obs_file)
    Sup3rGan.seed()

    model = Sup3rGan(
        gen_config_with_obs_2d(), pytest.S_FP_DISC, learning_rate=1e-4
    )
    model.meta['input_resolution'] = {'spatial': '16km', 'temporal': '3600min'}
    model.meta['lr_features'] = ['u_10m', 'v_10m']
    model.meta['hr_exo_features'] = ['u_10m_obs', 'v_10m_obs']
    model.meta['hr_out_features'] = ['u_10m', 'v_10m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 1

    with tempfile.TemporaryDirectory() as td:
        exo_tmp = {
            'u_10m_obs': {
                'steps': [
                    {
                        'model': 0,
                        'combine_type': 'layer',
                        'data': np.ones((6, 20, 20, 1)),
                    }
                ]
            },
            'v_10m_obs': {
                'steps': [
                    {
                        'model': 0,
                        'combine_type': 'layer',
                        'data': np.ones((6, 20, 20, 1)),
                    }
                ]
            },
        }
        _ = model.generate(np.ones((6, 10, 10, 2)), exogenous_data=exo_tmp)
        model_dir = os.path.join(td, 'test')
        model.save(model_dir)

        exo_handler_kwargs = {
            'u_10m_obs': {
                'file_paths': input_file,
                'source_files': obs_file,
                'cache_dir': td,
            },
            'v_10m_obs': {
                'file_paths': input_file,
                'source_files': obs_file,
                'cache_dir': td,
            },
        }

        model_kwargs = {'model_dir': model_dir}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        handler = ForwardPassStrategy(
            input_file,
            model_kwargs=model_kwargs,
            model_class='Sup3rGan',
            fwp_chunk_shape=fwp_chunk_shape,
            input_handler_kwargs=input_handler_kwargs,
            spatial_pad=0,
            temporal_pad=0,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
            pass_workers=2,
        )

        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)
