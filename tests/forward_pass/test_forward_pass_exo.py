""":class:`ForwardPass` tests with exogenous features"""

import json
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
from rex import ResourceX

from sup3r import CONFIG_DIR, __version__
from sup3r.models import (
    LinearInterp,
    SolarMultiStepGan,
    Sup3rGan,
    SurfaceSpatialMetModel,
)
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.preprocessing import Dimension, ExoDataHandler
from sup3r.utilities.pytest.helpers import make_fake_nc_file
from sup3r.utilities.utilities import RANDOM_GENERATOR, xr_open_mfdataset

target = (39.5, -105)
shape = (8, 8)
sample_shape = (8, 8, 6)
time_slice = slice(None, None, 1)
list_chunk_size = 10
fwp_chunk_shape = (4, 4, 150)
s_enhance = 3
t_enhance = 4


@pytest.fixture(scope='module')
def input_files(tmpdir_factory):
    """Dummy netcdf input files for :class:`ForwardPass`"""

    input_file = str(tmpdir_factory.mktemp('data').join('fwp_input.nc'))
    make_fake_nc_file(
        input_file,
        shape=(100, 100, 8),
        features=['pressure_0m', 'u_100m', 'v_100m'],
    )
    return input_file


def test_fwp_multi_step_model_topo_exoskip(input_files):
    """Test the forward pass with a multi step model class using exogenous data
    for the first two steps and not the last"""

    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {
        'spatial': '48km',
        'temporal': '60min',
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {
        'spatial': '24km',
        'temporal': '60min',
    }
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['u_100m', 'v_100m']
    st_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    st_model.meta['input_resolution'] = {
        'spatial': '12km',
        'temporal': '60min',
    }
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        fwp_chunk_shape = (4, 4, 8)
        s_enhance = 12
        t_enhance = 4

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            }
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
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
        t_steps = len(xr_open_mfdataset(input_files)[Dimension.TIME])

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * t_steps,
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1],
            )
            assert all(
                f in fh.attrs for f in ('windspeed_100m', 'winddirection_100m')
            )

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'model_meta' in fh.global_attrs
            model_meta = json.loads(fh.global_attrs['model_meta'])
            assert len(model_meta) == 3  # three step model
            assert model_meta[0]['lr_features'] == [
                'u_100m',
                'v_100m',
                'topography',
            ]


def test_fwp_multi_step_spatial_model_topo_noskip(input_files):
    """Test the forward pass with a multi step spatial only model class using
    exogenous data for all model steps"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {
        'spatial': '16km',
        'temporal': '60min',
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {'spatial': '8km', 'temporal': '60min'}
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    with tempfile.TemporaryDirectory() as td:
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        fwp_chunk_shape = (4, 4, 8)
        s_enhancements = [2, 2, 1]
        s_enhance = np.prod(s_enhancements)

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            }
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
        )

        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)
        t_steps = len(xr_open_mfdataset(input_files)[Dimension.TIME])

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_steps,
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1],
            )
            assert all(
                f in fh.attrs for f in ('windspeed_100m', 'winddirection_100m')
            )

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'model_meta' in fh.global_attrs
            model_meta = json.loads(fh.global_attrs['model_meta'])
            assert len(model_meta) == 2  # two step model
            assert model_meta[0]['lr_features'] == [
                'u_100m',
                'v_100m',
                'topography',
            ]


def test_fwp_multi_step_model_topo_noskip(input_files):
    """Test the forward pass with a multi step model class using exogenous data
    for all model steps"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {
        'spatial': '48km',
        'temporal': '60min',
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {
        'spatial': '24km',
        'temporal': '60min',
    }
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    st_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    st_model.meta['input_resolution'] = {
        'spatial': '12km',
        'temporal': '60min',
    }
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)))

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        fwp_chunk_shape = (4, 4, 8)
        s_enhancements = [2, 2, 3]
        s_enhance = np.prod(s_enhancements)
        t_enhance = 4

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            }
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
        )

        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)
        t_steps = len(xr_open_mfdataset(input_files)[Dimension.TIME])

        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * t_steps,
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1],
            )
            assert all(
                f in fh.attrs for f in ('windspeed_100m', 'winddirection_100m')
            )

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'model_meta' in fh.global_attrs
            model_meta = json.loads(fh.global_attrs['model_meta'])
            assert len(model_meta) == 3  # three step model
            assert model_meta[0]['lr_features'] == [
                'u_100m',
                'v_100m',
                'topography',
            ]


def test_fwp_single_step_sfc_model(input_files, plot=False):
    """Test the forward pass with a single SurfaceSpatialMetModel model
    which requires low and high-resolution topography input from the
    exogenous_data feature."""

    model = SurfaceSpatialMetModel(
        lr_features=['pressure_0m'],
        s_enhance=2,
        input_resolution={'spatial': '8km', 'temporal': '60min'},
    )

    with tempfile.TemporaryDirectory() as td:
        sfc_out_dir = os.path.join(td, 'sfc')
        model.save(sfc_out_dir)

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
                'steps': [
                    {'model': 0, 'combine_type': 'input'},
                    {'model': 0, 'combine_type': 'output'},
                ],
            }
        }

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=sfc_out_dir,
            model_class='SurfaceSpatialMetModel',
            fwp_chunk_shape=(8, 8, 8),
            spatial_pad=3,
            temporal_pad=3,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            pass_workers=2,
            max_nodes=1,
        )
        forward_pass = ForwardPass(handler)

        if plot:
            for ifeature, feature in enumerate(model.hr_out_features):
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(111)
                vmin = np.min(forward_pass.input_data[..., ifeature])
                vmax = np.max(forward_pass.input_data[..., ifeature])
                nc = ax1.imshow(
                    forward_pass.input_data[..., 0, ifeature],
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(nc, ax=ax1, shrink=0.6, label=f'{feature}')
                plt.savefig(f'./input_{feature}.png')
                plt.close()

        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_single_step_wind_hi_res_topo(input_files, plot=False):
    """Test the forward pass with a single spatiotemporal Sup3rGan model
    requiring high-resolution topography input from the exogenous_data
    feature."""
    Sup3rGan.seed()
    gen_model = [
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 64, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
        {
            'class': 'SpatioTemporalExpansion',
            'temporal_mult': 2,
            'temporal_method': 'nearest',
        },
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 64, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
        {'class': 'SpatioTemporalExpansion', 'spatial_mult': 2},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 64, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
        {'alpha': 0.2, 'class': 'LeakyReLU'},
        {'class': 'Sup3rConcat', 'name': 'topography'},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 2, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
    ]

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model = Sup3rGan(gen_model, fp_disc, learning_rate=1e-4)
    model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    model.meta['hr_exo_features'] = ['topography']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 2
    model.meta['input_resolution'] = {'spatial': '8km', 'temporal': '60min'}
    exo_tmp = {
        'topography': {
            'steps': [
                {
                    'model': 0,
                    'combine_type': 'layer',
                    'data': RANDOM_GENERATOR.random((4, 20, 20, 12, 1)),
                }
            ]
        }
    }
    _ = model.generate(
        RANDOM_GENERATOR.random((4, 10, 10, 6, 3)), exogenous_data=exo_tmp
    )

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        model.save(st_out_dir)

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            }
        }

        model_kwargs = {'model_dir': st_out_dir}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='Sup3rGan',
            fwp_chunk_shape=(8, 8, 8),
            spatial_pad=2,
            temporal_pad=2,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
        )
        forward_pass = ForwardPass(handler)

        if plot:
            for ifeature, feature in enumerate(model.hr_out_features):
                fig = plt.figure(figsize=(15, 5))
                ax1 = fig.add_subplot(111)
                vmin = np.min(forward_pass.input_data[..., ifeature])
                vmax = np.max(forward_pass.input_data[..., ifeature])
                nc = ax1.imshow(
                    forward_pass.input_data[..., 0, ifeature],
                    vmin=vmin,
                    vmax=vmax,
                )
                fig.colorbar(nc, ax=ax1, shrink=0.6, label=f'{feature}')
                plt.savefig(f'./input_{feature}.png')
                plt.close()

        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_multi_step_wind_hi_res_topo(input_files, gen_config_with_topo):
    """Test the forward pass with multiple Sup3rGan models requiring
    high-resolution topograph input from the exogenous_data feature."""
    Sup3rGan.seed()
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(
        gen_config_with_topo('Sup3rConcat'), fp_disc, learning_rate=1e-4
    )
    s1_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s1_model.meta['hr_exo_features'] = ['topography']
    s1_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {
        'spatial': '48km',
        'temporal': '60min',
    }

    exo_tmp = {
        'topography': {
            'steps': [
                {
                    'model': 0,
                    'combine_type': 'layer',
                    'data': RANDOM_GENERATOR.random((4, 20, 20, 1)),
                }
            ]
        }
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 3)), exogenous_data=exo_tmp)

    s2_model = Sup3rGan(
        gen_config_with_topo('Sup3rConcat'), fp_disc, learning_rate=1e-4
    )
    s2_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s2_model.meta['hr_exo_features'] = ['topography']
    s2_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {
        'spatial': '24km',
        'temporal': '60min',
    }
    _ = s2_model.generate(np.ones((4, 10, 10, 3)), exogenous_data=exo_tmp)

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    st_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    st_model.meta['input_resolution'] = {
        'spatial': '12km',
        'temporal': '60min',
    }
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)))

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            }
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
        )
        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_wind_hi_res_topo_plus_linear(input_files, gen_config_with_topo):
    """Test the forward pass with a Sup3rGan model requiring high-res topo
    input from exo data for spatial enhancement and a linear interpolation
    model for temporal enhancement."""

    Sup3rGan.seed()

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s_model = Sup3rGan(
        gen_config_with_topo('Sup3rConcat'), fp_disc, learning_rate=1e-4
    )
    s_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s_model.meta['hr_exo_features'] = ['topography']
    s_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s_model.meta['s_enhance'] = 2
    s_model.meta['t_enhance'] = 1
    s_model.meta['input_resolution'] = {'spatial': '12km', 'temporal': '60min'}
    exo_tmp = {
        'topography': {
            'steps': [
                {
                    'combine_type': 'layer',
                    'data': RANDOM_GENERATOR.random((4, 20, 20, 1)),
                }
            ]
        }
    }
    _ = s_model.generate(np.ones((4, 10, 10, 3)), exogenous_data=exo_tmp)

    t_model = LinearInterp(
        lr_features=['u_100m', 'v_100m'], s_enhance=1, t_enhance=4
    )
    t_model.meta['input_resolution'] = {'spatial': '4km', 'temporal': '60min'}

    with tempfile.TemporaryDirectory() as td:
        s_out_dir = os.path.join(td, 's_gan')
        t_out_dir = os.path.join(td, 't_interp')
        s_model.save(s_out_dir)
        t_model.save(t_out_dir)

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            }
        }

        model_kwargs = {'model_dirs': [s_out_dir, t_out_dir]}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=(8, 8, 8),
            spatial_pad=2,
            temporal_pad=2,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            allowed_const=0,
            max_nodes=1,
        )
        forward_pass = ForwardPass(handler)
        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)


def test_fwp_multi_step_model_multi_exo(input_files):
    """Test the forward pass with a multi step model class using 2 exogenous
    data features"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s1_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s1_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {
        'spatial': '48km',
        'temporal': '60min',
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 3)))

    s2_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s2_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography']
    s2_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {
        'spatial': '24km',
        'temporal': '60min',
    }
    _ = s2_model.generate(np.ones((4, 10, 10, 3)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.meta['input_resolution'] = {
        'spatial': '12km',
        'temporal': '60min',
    }
    st_model.meta['lr_features'] = ['u_100m', 'v_100m', 'sza']
    st_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 4
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)))

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        fwp_chunk_shape = (4, 4, 8)
        s_enhancements = [2, 2, 3]
        s_enhance = np.prod(s_enhancements)
        t_enhance = 4

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            },
            'sza': {
                'file_paths': input_files,
                'cache_dir': td,
            },
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]}

        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
        )

        forward_pass = ForwardPass(handler)

        forward_pass.run(handler, node_index=0)
        t_steps = len(xr_open_mfdataset(input_files)[Dimension.TIME])
        with ResourceX(handler.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * t_steps,
                s_enhance**2 * fwp_chunk_shape[0] * fwp_chunk_shape[1],
            )
            assert all(
                f in fh.attrs for f in ('windspeed_100m', 'winddirection_100m')
            )

            assert fh.global_attrs['package'] == 'sup3r'
            assert fh.global_attrs['version'] == __version__
            assert 'full_version_record' in fh.global_attrs
            version_record = json.loads(fh.global_attrs['full_version_record'])
            assert version_record['tensorflow'] == tf.__version__
            assert 'model_meta' in fh.global_attrs
            model_meta = json.loads(fh.global_attrs['model_meta'])
            assert len(model_meta) == 3  # three step model
            assert model_meta[0]['lr_features'] == [
                'u_100m',
                'v_100m',
                'topography',
            ]

    shutil.rmtree('./exo_cache', ignore_errors=True)


def test_fwp_multi_step_exo_hi_res_topo_and_sza(
    input_files, gen_config_with_topo
):
    """Test the forward pass with multiple ExoGan models requiring
    high-resolution topography and sza input from the exogenous_data
    feature."""
    Sup3rGan.seed()

    gen_t_model = [
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 1, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
        {'alpha': 0.2, 'class': 'LeakyReLU'},
        {
            'class': 'SpatioTemporalExpansion',
            'temporal_mult': 2,
            'temporal_method': 'nearest',
        },
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 1, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 36, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
        {'class': 'SpatioTemporalExpansion', 'spatial_mult': 3},
        {'alpha': 0.2, 'class': 'LeakyReLU'},
        {'class': 'Sup3rConcat', 'name': 'sza'},
        {
            'class': 'FlexiblePadding',
            'paddings': [[0, 0], [3, 3], [3, 3], [3, 3], [0, 0]],
            'mode': 'REFLECT',
        },
        {'class': 'Conv3D', 'filters': 2, 'kernel_size': 3, 'strides': 1},
        {'class': 'Cropping3D', 'cropping': 2},
    ]

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s1_model = Sup3rGan(
        gen_config_with_topo('Sup3rConcat'), fp_disc, learning_rate=1e-4
    )
    s1_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography', 'sza']
    s1_model.meta['hr_exo_features'] = ['topography']
    s1_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s1_model.meta['s_enhance'] = 2
    s1_model.meta['t_enhance'] = 1
    s1_model.meta['input_resolution'] = {
        'spatial': '48km',
        'temporal': '60min',
    }
    exo_tmp = {
        'topography': {
            'steps': [
                {
                    'model': 0,
                    'combine_type': 'layer',
                    'data': np.ones((4, 20, 20, 1)),
                }
            ]
        },
        'sza': {
            'steps': [
                {
                    'model': 0,
                    'combine_type': 'layer',
                    'data': np.ones((4, 20, 20, 1)),
                }
            ]
        },
    }
    _ = s1_model.generate(np.ones((4, 10, 10, 4)), exogenous_data=exo_tmp)

    s2_model = Sup3rGan(
        gen_config_with_topo('Sup3rConcat'), fp_disc, learning_rate=1e-4
    )
    s2_model.meta['lr_features'] = ['u_100m', 'v_100m', 'topography', 'sza']
    s2_model.meta['hr_exo_features'] = ['topography']
    s2_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    s2_model.meta['s_enhance'] = 2
    s2_model.meta['t_enhance'] = 1
    s2_model.meta['input_resolution'] = {
        'spatial': '24km',
        'temporal': '60min',
    }
    _ = s2_model.generate(np.ones((4, 10, 10, 4)), exogenous_data=exo_tmp)

    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(gen_t_model, fp_disc, learning_rate=1e-4)
    st_model.meta['lr_features'] = ['u_100m', 'v_100m', 'sza']
    st_model.meta['hr_exo_features'] = ['sza']
    st_model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    st_model.meta['s_enhance'] = 3
    st_model.meta['t_enhance'] = 2
    st_model.meta['input_resolution'] = {
        'spatial': '12km',
        'temporal': '60min',
    }
    exo_tmp = {
        'sza': {
            'steps': [
                {
                    'model': 0,
                    'combine_type': 'layer',
                    'data': np.ones((4, 30, 30, 12, 1)),
                }
            ]
        }
    }
    _ = st_model.generate(np.ones((4, 10, 10, 6, 3)), exogenous_data=exo_tmp)

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        s1_out_dir = os.path.join(td, 's1_gan')
        s2_out_dir = os.path.join(td, 's2_gan')
        st_model.save(st_out_dir)
        s1_model.save(s1_out_dir)
        s2_model.save(s2_out_dir)

        exo_handler_kwargs = {
            'topography': {
                'file_paths': input_files,
                'source_files': pytest.FP_WTK,
                'cache_dir': td,
            },
            'sza': {
                'file_paths': input_files,
                'cache_dir': td,
            },
        }

        model_kwargs = {'model_dirs': [s1_out_dir, s2_out_dir, st_out_dir]}
        out_files = os.path.join(td, 'out_{file_id}.h5')
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        handler = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=(4, 4, 8),
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            exo_handler_kwargs=exo_handler_kwargs,
            max_nodes=1,
        )
        forward_pass = ForwardPass(handler)

        assert ExoDataHandler.get_exo_steps(
            'topography', forward_pass.model.models
        ) == [
            {'model': 0, 'combine_type': 'input'},
            {'model': 0, 'combine_type': 'layer'},
            {'model': 1, 'combine_type': 'input'},
            {'model': 1, 'combine_type': 'layer'},
        ]

        assert ExoDataHandler.get_exo_steps(
            'sza', forward_pass.model.models
        ) == [
            {'model': 0, 'combine_type': 'input'},
            {'model': 1, 'combine_type': 'input'},
            {'model': 2, 'combine_type': 'input'},
            {'model': 2, 'combine_type': 'layer'},
        ]

        forward_pass.run(handler, node_index=0)

        for fp in handler.out_files:
            assert os.path.exists(fp)

    shutil.rmtree('./exo_cache', ignore_errors=True)


def test_solar_multistep_exo(gen_config_with_topo):
    """Test the special solar multistep model with exo features."""
    features1 = ['clearsky_ratio']
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model1 = Sup3rGan(fp_gen, fp_disc)
    _ = model1.generate(np.ones((4, 10, 10, len(features1))))
    model1.set_norm_stats({'clearsky_ratio': 0.7}, {'clearsky_ratio': 0.04})
    model1.meta['input_resolution'] = {'spatial': '8km', 'temporal': '40min'}
    model1.set_model_params(lr_features=features1, hr_out_features=features1)

    features2 = ['U_200m', 'V_200m', 'topography']

    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model2 = Sup3rGan(gen_config_with_topo('Sup3rConcat'), fp_disc)

    exo_tmp = {
        'topography': {
            'steps': [
                {
                    'model': 0,
                    'combine_type': 'layer',
                    'data': RANDOM_GENERATOR.random((4, 20, 20, 1)),
                }
            ]
        }
    }

    _ = model2.generate(
        np.ones((4, 10, 10, len(features2))), exogenous_data=exo_tmp
    )
    model2.set_norm_stats(
        {'U_200m': 4.2, 'V_200m': 5.6, 'topography': 100.2},
        {'U_200m': 1.1, 'V_200m': 1.3, 'topography': 50.3},
    )
    model2.meta['input_resolution'] = {'spatial': '4km', 'temporal': '40min'}
    model2.set_model_params(
        lr_features=features2,
        hr_out_features=features2[:-1],
        hr_exo_features=features2[-1:],
    )

    features_in_3 = ['clearsky_ratio', 'U_200m', 'V_200m']
    features_out_3 = ['clearsky_ratio']
    fp_gen = os.path.join(CONFIG_DIR, 'sup3rcc/gen_solar_1x_8x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model3 = Sup3rGan(fp_gen, fp_disc)
    _ = model3.generate(np.ones((4, 10, 10, 3, len(features_in_3))))
    model3.set_norm_stats(
        {'U_200m': 4.2, 'V_200m': 5.6, 'clearsky_ratio': 0.7},
        {'U_200m': 1.1, 'V_200m': 1.3, 'clearsky_ratio': 0.04},
    )
    model3.meta['input_resolution'] = {'spatial': '2km', 'temporal': '40min'}
    model3.set_model_params(
        lr_features=features_in_3, hr_out_features=features_out_3
    )

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        fp3 = os.path.join(td, 'model3')
        model1.save(fp1)
        model2.save(fp2)
        model3.save(fp3)

        with pytest.raises(AssertionError):
            SolarMultiStepGan.load(fp2, fp1, fp3)

        ms_model = SolarMultiStepGan.load(fp1, fp2, fp3)

        x = np.ones((3, 10, 10, len(features1 + features2)))
        exo_tmp = {
            'topography': {
                'steps': [
                    {
                        'model': 0,
                        'combine_type': 'input',
                        'data': RANDOM_GENERATOR.random((3, 10, 10, 1)),
                    },
                    {
                        'model': 0,
                        'combine_type': 'layer',
                        'data': RANDOM_GENERATOR.random((3, 20, 20, 1)),
                    },
                ]
            }
        }
        steps = ExoDataHandler.get_exo_steps('topography', ms_model.models)
        assert steps == [
            {'model': 0, 'combine_type': 'input'},
            {'model': 0, 'combine_type': 'layer'},
        ]
        out = ms_model.generate(x, exogenous_data=exo_tmp)
        assert out.shape == (1, 20, 20, 24, 1)
