"""pytests for forward pass module"""

import json
import os
import tempfile
from glob import glob

import numpy as np
import pytest
import tensorflow as tf
from rex import ResourceX

from sup3r import CONFIG_DIR, __version__
from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.preprocessing import DataHandler, Dimension
from sup3r.utilities.pytest.helpers import (
    make_fake_nc_file,
)
from sup3r.utilities.utilities import xr_open_mfdataset

FEATURES = ['u_100m', 'v_100m']
target = (19.3, -123.5)
shape = (8, 8)
time_slice = slice(None, None, 1)
fwp_chunk_shape = (4, 4, 150)
s_enhance = 3
t_enhance = 4


@pytest.fixture(scope='module')
def input_files(tmpdir_factory):
    """Dummy netcdf input files for :class:`ForwardPass`"""

    input_file = str(tmpdir_factory.mktemp('data').join('fwp_input.nc'))
    make_fake_nc_file(input_file, shape=(100, 100, 50), features=FEATURES)
    return input_file


def test_fwp_nc_cc():
    """Test forward pass handler output for netcdf write with cc data. Also
    tests default fwp_chunk_shape"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    features = ['u_100m', 'v_100m']
    target = (13.67, 125.0)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['lr_features'] = features
    model.meta['hr_out_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        out_files = os.path.join(td, 'out_{file_id}.nc')
        # 1st forward pass
        strat = ForwardPassStrategy(
            pytest.FPS_GCM,
            fwp_chunk_shape=(*fwp_chunk_shape[:-1], None),
            spatial_pad=1,
            model_kwargs={'model_dir': out_dir},
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
            },
            out_pattern=out_files,
            input_handler_name='DataHandlerNCforCC',
            pass_workers=None,
            invert_uv=False,
        )
        forward_pass = ForwardPass(strat)
        forward_pass.run(strat, node_index=0)

        with xr_open_mfdataset(strat.out_files[0]) as fh:
            assert fh[FEATURES[0]].transpose(
                Dimension.TIME, *Dimension.dims_2d()
            ).shape == (
                t_enhance * len(strat.input_handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1],
            )
            assert fh[FEATURES[1]].transpose(
                Dimension.TIME, *Dimension.dims_2d()
            ).shape == (
                t_enhance * len(strat.input_handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1],
            )


def test_fwp_nc_cc_with_cache():
    """Test forward pass handler with input caching"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)

    features = ['u_100m', 'v_100m']
    target = (13.67, 125.0)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.meta['lr_features'] = features
    model.meta['hr_out_features'] = features
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        cache_pattern = os.path.join(td, 'cache_{feature}.nc')
        out_files = os.path.join(td, 'out_{file_id}.nc')
        # 1st forward pass
        strat = ForwardPassStrategy(
            pytest.FPS_GCM,
            fwp_chunk_shape=(*fwp_chunk_shape[:-1], None),
            spatial_pad=1,
            model_kwargs={'model_dir': out_dir},
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
                'cache_kwargs': {'cache_pattern': cache_pattern},
            },
            out_pattern=out_files,
            input_handler_name='DataHandlerNCforCC',
            pass_workers=None,
        )
        forward_pass = ForwardPass(strat)
        forward_pass.run(strat, node_index=0)

        cache_files = [cache_pattern.format(feature=f) for f in features]
        assert sorted(glob(cache_pattern.format(feature='*'))) == sorted(
            cache_files
        )


def test_fwp_spatial_only(input_files):
    """Test forward pass handler output for spatial only model."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    model.meta['s_enhance'] = 2
    model.meta['t_enhance'] = 1
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 's_gan')
        model.save(out_dir)
        out_files = os.path.join(td, 'out_{file_id}.nc')
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_name='Rasterizer',
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
            },
            out_pattern=out_files,
            pass_workers=1,
            output_workers=1,
            invert_uv=False,
        )
        forward_pass = ForwardPass(strat)
        assert strat.output_workers == 1
        assert strat.pass_workers == 1
        forward_pass.run(strat, node_index=0)

        with xr_open_mfdataset(strat.out_files[0]) as fh:
            assert fh[FEATURES[0]].transpose(
                Dimension.TIME, *Dimension.dims_2d()
            ).shape == (
                len(strat.input_handler.time_index),
                2 * fwp_chunk_shape[0],
                2 * fwp_chunk_shape[1],
            )
            assert fh[FEATURES[1]].transpose(
                Dimension.TIME, *Dimension.dims_2d()
            ).shape == (
                len(strat.input_handler.time_index),
                2 * fwp_chunk_shape[0],
                2 * fwp_chunk_shape[1],
            )


def test_fwp_nc(input_files):
    """Test forward pass handler output for netcdf write."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        out_files = os.path.join(td, 'out_{file_id}.nc')
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
            },
            out_pattern=out_files,
            pass_workers=1,
            invert_uv=False,
        )
        forward_pass = ForwardPass(strat)
        assert forward_pass.strategy.pass_workers == 1
        forward_pass.run(strat, node_index=0)

        with xr_open_mfdataset(strat.out_files[0]) as fh:
            assert fh[FEATURES[0]].transpose(
                Dimension.TIME, *Dimension.dims_2d()
            ).shape == (
                t_enhance * len(strat.input_handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1],
            )
            assert fh[FEATURES[1]].transpose(
                Dimension.TIME, *Dimension.dims_2d()
            ).shape == (
                t_enhance * len(strat.input_handler.time_index),
                s_enhance * fwp_chunk_shape[0],
                s_enhance * fwp_chunk_shape[1],
            )


def test_fwp_with_cache_reload(input_files):
    """Test forward pass handler output with cache loading"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        out_files = os.path.join(td, 'out_{file_id}.nc')
        cache_pattern = os.path.join(td, 'cache_{feature}.nc')
        kwargs = {
            'model_kwargs': {'model_dir': out_dir},
            'fwp_chunk_shape': fwp_chunk_shape,
            'spatial_pad': 1,
            'temporal_pad': 1,
            'input_handler_kwargs': {
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
                'cache_kwargs': {
                    'cache_pattern': cache_pattern,
                    'max_workers': 1,
                },
            },
            'input_handler_name': 'DataHandler',
            'out_pattern': out_files,
            'pass_workers': 1,
        }
        strat = ForwardPassStrategy(input_files, **kwargs)
        forward_pass = ForwardPass(strat)
        forward_pass.run(strat, node_index=0)

        cache_files = [cache_pattern.format(feature=f) for f in FEATURES]
        assert sorted(glob(cache_pattern.format(feature='*'))) == sorted(
            cache_files
        )

        strat = ForwardPassStrategy(input_files, **kwargs)
        forward_pass = ForwardPass(strat)
        forward_pass.run(strat, node_index=0)


def test_fwp_time_slice(input_files):
    """Test forward pass handler output to h5 file. Includes temporal
    slicing."""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, 2)))
    model.meta['lr_features'] = ['u_100m', 'v_100m']
    model.meta['hr_out_features'] = ['u_100m', 'v_100m']
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4
    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        out_files = os.path.join(td, 'out_{file_id}.h5')
        time_slice = slice(5, 17, 3)
        raw_time_index = np.arange(20)
        n_tsteps = len(raw_time_index[time_slice])
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
            },
            out_pattern=out_files,
            pass_workers=1,
        )
        forward_pass = ForwardPass(strat)
        forward_pass.run(strat, node_index=0)

        with ResourceX(strat.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * n_tsteps,
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
            assert isinstance(model_meta, dict)
            assert model_meta['lr_features'] == ['u_100m', 'v_100m']


def test_fwp_handler(input_files):
    """Test forward pass handler. Make sure it is
    returning the correct data shape"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:-1]
    model.meta['s_enhance'] = s_enhance
    model.meta['t_enhance'] = t_enhance
    _ = model.generate(np.ones((4, 10, 10, 12, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=1,
            temporal_pad=1,
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
            },
        )
        fwp = ForwardPass(strat)

        _, data = fwp.run_chunk(
            chunk=fwp.get_input_chunk(chunk_index=0),
            model_kwargs=strat.model_kwargs,
            model_class=strat.model_class,
            allowed_const=strat.allowed_const,
            output_workers=strat.output_workers,
            meta=fwp.meta,
        )

        raw_tsteps = len(xr_open_mfdataset(input_files)[Dimension.TIME])
        assert data.shape == (
            s_enhance * fwp_chunk_shape[0],
            s_enhance * fwp_chunk_shape[1],
            t_enhance * raw_tsteps,
            2,
        )


def test_fwp_chunking(input_files):
    """Test forward pass spatialtemporal chunking. Make sure chunking agrees
    closely with non chunked forward pass.
    """

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES
    model.meta['s_enhance'] = s_enhance
    model.meta['t_enhance'] = t_enhance
    _ = model.generate(np.ones((4, 10, 10, 12, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'test_1')
        model.save(out_dir)
        spatial_pad = 20
        temporal_pad = 20
        raw_tsteps = len(xr_open_mfdataset(input_files)[Dimension.TIME])
        fwp_shape = (5, 5, raw_tsteps // 2)
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=fwp_shape,
            spatial_pad=spatial_pad,
            temporal_pad=temporal_pad,
            input_handler_kwargs={
                'target': target,
                'shape': shape,
                'time_slice': time_slice,
            },
        )
        data_chunked = np.zeros((
            shape[0] * s_enhance,
            shape[1] * s_enhance,
            raw_tsteps * t_enhance,
            len(model.hr_out_features),
        ))
        handlerNC = DataHandler(
            input_files, FEATURES, target=target, shape=shape
        )
        pad_width = (
            (spatial_pad, spatial_pad),
            (spatial_pad, spatial_pad),
            (temporal_pad, temporal_pad),
            (0, 0),
        )
        hr_crop = (
            slice(s_enhance * spatial_pad, -s_enhance * spatial_pad),
            slice(s_enhance * spatial_pad, -s_enhance * spatial_pad),
            slice(t_enhance * temporal_pad, -t_enhance * temporal_pad),
            slice(None),
        )
        input_data = np.pad(
            handlerNC.data.as_array(), pad_width=pad_width, mode='constant'
        )
        data_nochunk = model.generate(np.expand_dims(input_data, axis=0))[0][
            hr_crop
        ]
        fwp = ForwardPass(strat)
        for i in range(strat.n_chunks):
            _, out = fwp.run_chunk(
                fwp.get_input_chunk(i, mode='constant'),
                model_kwargs=strat.model_kwargs,
                model_class=strat.model_class,
                allowed_const=strat.allowed_const,
                output_workers=strat.output_workers,
                meta=fwp.meta,
            )
            s_chunk_idx, t_chunk_idx = (
                fwp.strategy.fwp_slicer.get_chunk_indices(i)
            )
            ti_slice = fwp.strategy.ti_slices[t_chunk_idx]
            hr_slice = fwp.strategy.hr_slices[s_chunk_idx]

            t_hr_slice = slice(
                ti_slice.start * t_enhance, ti_slice.stop * t_enhance
            )
            data_chunked[hr_slice][..., t_hr_slice, :] = out

        err = data_chunked - data_nochunk
        assert np.mean(np.abs(err)) < 1e-6


def test_fwp_nochunking(input_files):
    """Test forward pass without chunking. Make sure using a single chunk
    (a.k.a nochunking) matches direct forward pass of full dataset.
    """

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:-1]
    model.meta['s_enhance'] = s_enhance
    model.meta['t_enhance'] = t_enhance
    _ = model.generate(np.ones((4, 10, 10, 12, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': out_dir},
            fwp_chunk_shape=(
                shape[0],
                shape[1],
                len(xr_open_mfdataset(input_files)[Dimension.TIME]),
            ),
            spatial_pad=0,
            temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
        )
        fwp = ForwardPass(strat)
        _, data_chunked = fwp.run_chunk(
            fwp.get_input_chunk(chunk_index=0),
            model_kwargs=strat.model_kwargs,
            model_class=strat.model_class,
            allowed_const=strat.allowed_const,
            output_workers=strat.output_workers,
            meta=fwp.meta,
        )

        handlerNC = DataHandler(
            input_files,
            FEATURES,
            target=target,
            shape=shape,
            time_slice=time_slice,
        )

        data_nochunk = model.generate(
            np.expand_dims(handlerNC.data.as_array(), axis=0)
        )[0]

        assert np.array_equal(data_chunked, data_nochunk)


def test_fwp_multi_step_model(input_files):
    """Test the forward pass with a multi step model class"""
    Sup3rGan.seed()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    s_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    s_model.set_model_params(
        lr_features=['u_100m', 'v_100m'],
        hr_out_features=['u_100m', 'v_100m'],
        s_enhance=2,
        t_enhance=1,
        input_resolution={'spatial': '6km', 'temporal': '40min'},
    )
    _ = s_model.generate(np.ones((4, 10, 10, 2)))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    st_model.set_model_params(
        lr_features=['u_100m', 'v_100m'],
        hr_out_features=['u_100m', 'v_100m'],
        s_enhance=3,
        t_enhance=4,
        input_resolution={'spatial': '3km', 'temporal': '40min'},
    )
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        st_out_dir = os.path.join(td, 'st_gan')
        s_out_dir = os.path.join(td, 's_gan')
        st_model.save(st_out_dir)
        s_model.save(s_out_dir)

        out_files = os.path.join(td, 'out_{file_id}.h5')

        fwp_chunk_shape = (4, 4, 8)
        s_enhance = 6
        t_enhance = 4

        model_kwargs = {'model_dirs': [s_out_dir, st_out_dir]}

        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        strat = ForwardPassStrategy(
            input_files,
            model_kwargs=model_kwargs,
            model_class='MultiStepGan',
            fwp_chunk_shape=fwp_chunk_shape,
            spatial_pad=0,
            temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            max_nodes=1,
        )
        fwp = ForwardPass(strat)

        _, _ = fwp.run_chunk(
            fwp.get_input_chunk(chunk_index=0),
            model_kwargs=strat.model_kwargs,
            model_class=strat.model_class,
            allowed_const=strat.allowed_const,
            output_workers=strat.output_workers,
            invert_uv=strat.invert_uv,
            meta=fwp.meta,
        )

        with ResourceX(strat.out_files[0]) as fh:
            assert fh.shape == (
                t_enhance * fwp_chunk_shape[2],
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
            assert model_meta[0]['lr_features'] == ['u_100m', 'v_100m']


def test_slicing_no_pad(input_files):
    """Test the slicing of input data via the ForwardPassStrategy +
    ForwardPassSlicer vs. the actual source data. Does not include any
    reflected padding at the edges."""

    Sup3rGan.seed()
    s_enhance = 3
    t_enhance = 4
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    features = ['u_100m', 'v_100m']
    st_model.meta['lr_features'] = features
    st_model.meta['hr_out_features'] = features
    st_model.meta['s_enhance'] = s_enhance
    st_model.meta['t_enhance'] = t_enhance
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        out_files = os.path.join(td, 'out_{file_id}.h5')
        st_out_dir = os.path.join(td, 'st_gan')
        st_model.save(st_out_dir)

        handler = DataHandler(
            input_files, features, target=target, shape=shape
        )

        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        # raises warning because fwp_chunk_shape is too small
        with pytest.warns(match='at least 4'):
            strategy = ForwardPassStrategy(
                input_files,
                model_kwargs={'model_dir': st_out_dir},
                model_class='Sup3rGan',
                fwp_chunk_shape=(3, 2, 4),
                spatial_pad=0,
                temporal_pad=0,
                input_handler_kwargs=input_handler_kwargs,
                out_pattern=out_files,
                max_nodes=1,
            )
        strategy = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(4, 4, 4),
            spatial_pad=0,
            temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            max_nodes=1,
        )

        fwp = ForwardPass(strategy)
        for i in strategy.node_chunks[0]:
            chunk = fwp.get_input_chunk(i)
            s_idx, t_idx = strategy.fwp_slicer.get_chunk_indices(i)
            s_slices = strategy.lr_slices[s_idx]
            s_pad_slices = strategy.lr_pad_slices[s_idx]
            s_crop_slices = strategy.fwp_slicer.s_lr_crop_slices[s_idx]
            t_crop_slice = strategy.fwp_slicer.t_lr_crop_slices[t_idx]
            lr_pad_data_slice = (
                s_pad_slices[0],
                s_pad_slices[1],
                fwp.strategy.ti_pad_slices[t_idx],
            )
            lr_crop_data_slice = (
                s_crop_slices[0],
                s_crop_slices[1],
                t_crop_slice,
            )
            lr_data_slice = (
                s_slices[0],
                s_slices[1],
                fwp.strategy.ti_slices[t_idx],
            )

            assert handler.data[lr_pad_data_slice].shape[:-2][0] > 3
            assert handler.data[lr_pad_data_slice].shape[:-2][1] > 3
            assert chunk.input_data.shape[:-2][0] > 3
            assert chunk.input_data.shape[:-2][1] > 3
            assert np.allclose(
                chunk.input_data, handler.data[lr_pad_data_slice]
            )
            assert np.allclose(
                chunk.input_data[lr_crop_data_slice],
                handler.data[lr_data_slice],
            )


@pytest.mark.parametrize('spatial_pad', [0, 1])
def test_slicing_auto_boundary_pad(input_files, spatial_pad):
    """Test that automatic boundary padding is applied when the fwp chunk shape
    and grid size result in a slice that is too small for the generator.

    Here the fwp chunk shape is (7, 7, 4) and the grid size is (8, 8) so with
    no spatial padding this results in some chunk slices that have length 1.
    With spatial padding equal to 1 some slices have length 3. In each of these
    case we need to pad the slices so the input to the generator has at least 4
    elements."""

    Sup3rGan.seed()
    s_enhance = 3
    t_enhance = 4
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    features = ['u_100m', 'v_100m']
    st_model.meta['lr_features'] = features
    st_model.meta['hr_out_features'] = features
    st_model.meta['s_enhance'] = s_enhance
    st_model.meta['t_enhance'] = t_enhance
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        out_files = os.path.join(td, 'out_{file_id}.h5')
        st_out_dir = os.path.join(td, 'st_gan')
        st_model.save(st_out_dir)

        handler = DataHandler(
            input_files, features, target=target, shape=shape
        )

        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }

        strategy = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(7, 7, 4),
            spatial_pad=spatial_pad,
            temporal_pad=0,
            input_handler_kwargs=input_handler_kwargs,
            out_pattern=out_files,
            max_nodes=1,
        )

        fwp = ForwardPass(strategy)
        for i in strategy.node_chunks[0]:
            chunk = fwp.get_input_chunk(i)
            s_idx, t_idx = strategy.fwp_slicer.get_chunk_indices(i)
            pad_width = strategy.fwp_slicer.get_pad_width(i)
            s_slices = strategy.lr_slices[s_idx]
            s_crop_slices = strategy.fwp_slicer.s_lr_crop_slices[s_idx]
            t_crop_slice = strategy.fwp_slicer.t_lr_crop_slices[t_idx]
            lr_crop_data_slice = (
                s_crop_slices[0],
                s_crop_slices[1],
                t_crop_slice,
            )
            lr_data_slice = (
                s_slices[0],
                s_slices[1],
                fwp.strategy.ti_slices[t_idx],
            )

            assert chunk.input_data.shape[0] > strategy.fwp_slicer.min_width[0]
            assert chunk.input_data.shape[1] > strategy.fwp_slicer.min_width[1]
            input_data = chunk.input_data.copy()
            if spatial_pad > 0:
                slices = [
                    slice(pw[0] or None, -pw[1] or None) for pw in pad_width
                ]
                input_data = input_data[slices[0], slices[1]]
            hdata = handler.data[lr_data_slice]
            assert np.allclose(input_data[lr_crop_data_slice], hdata)


def test_slicing_pad(input_files):
    """Test the slicing of input data via the ForwardPassStrategy +
    ForwardPassSlicer vs. the actual source data. Includes reflected padding
    at the edges."""

    Sup3rGan.seed()
    s_enhance = 3
    t_enhance = 4
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    st_model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    features = ['u_100m', 'v_100m']
    st_model.meta['lr_features'] = features
    st_model.meta['hr_out_features'] = features
    st_model.meta['s_enhance'] = s_enhance
    st_model.meta['t_enhance'] = t_enhance
    _ = st_model.generate(np.ones((4, 10, 10, 6, 2)))

    with tempfile.TemporaryDirectory() as td:
        out_files = os.path.join(td, 'out_{file_id}.h5')
        st_out_dir = os.path.join(td, 'st_gan')
        st_model.save(st_out_dir)
        handler = DataHandler(
            input_files, features, target=target, shape=shape
        )
        input_handler_kwargs = {
            'target': target,
            'shape': shape,
            'time_slice': time_slice,
        }
        strategy = ForwardPassStrategy(
            input_files,
            model_kwargs={'model_dir': st_out_dir},
            model_class='Sup3rGan',
            fwp_chunk_shape=(2, 1, 4),
            input_handler_kwargs=input_handler_kwargs,
            spatial_pad=2,
            temporal_pad=2,
            out_pattern=out_files,
            max_nodes=1,
        )

        chunk_lookup = strategy.fwp_slicer.chunk_lookup
        n_s1 = len(strategy.fwp_slicer.s1_lr_slices)
        n_s2 = len(strategy.fwp_slicer.s2_lr_slices)
        n_t = len(strategy.fwp_slicer.t_lr_slices)

        assert chunk_lookup[0, 0, 0] == 0
        assert chunk_lookup[0, 1, 0] == 1
        assert chunk_lookup[0, 2, 0] == 2
        assert chunk_lookup[1, 0, 0] == n_s2
        assert chunk_lookup[1, 1, 0] == n_s2 + 1
        assert chunk_lookup[2, 0, 0] == n_s2 * 2
        assert chunk_lookup[0, 0, 1] == n_s1 * n_s2
        assert chunk_lookup[0, 1, 1] == n_s1 * n_s2 + 1

        fwp = ForwardPass(strategy)
        for i in strategy.node_chunks[0]:
            chunk = fwp.get_input_chunk(i, mode='constant')
            s_idx, t_idx = strategy.fwp_slicer.get_chunk_indices(i)
            s_slices = strategy.lr_pad_slices[s_idx]
            lr_data_slice = (
                s_slices[0],
                s_slices[1],
                fwp.strategy.ti_pad_slices[t_idx],
            )

            # do a manual calculation of what the padding should be.
            # s1 and t axes should have padding of 2 and the borders and
            # padding of 1 when 1 index away from the borders (chunk shape is 1
            # in those axes). s2 should have padding of 2 at the
            # borders and 0 everywhere else.
            ids1, ids2, idt = np.where(chunk_lookup == i)
            ids1, ids2, idt = ids1[0], ids2[0], idt[0]

            start_s1_pad_lookup = {0: 2}
            start_s2_pad_lookup = {0: 2, 1: 1}
            start_t_pad_lookup = {0: 2}
            end_s1_pad_lookup = {n_s1 - 1: 2}
            end_s2_pad_lookup = {n_s2 - 1: 2, n_s2 - 2: 1}
            end_t_pad_lookup = {n_t - 1: 2}

            pad_s1_start = start_s1_pad_lookup.get(ids1, 0)
            pad_s2_start = start_s2_pad_lookup.get(ids2, 0)
            pad_t_start = start_t_pad_lookup.get(idt, 0)
            pad_s1_end = end_s1_pad_lookup.get(ids1, 0)
            pad_s2_end = end_s2_pad_lookup.get(ids2, 0)
            pad_t_end = end_t_pad_lookup.get(idt, 0)

            pad_width = (
                (pad_s1_start, pad_s1_end),
                (pad_s2_start, pad_s2_end),
                (pad_t_start, pad_t_end),
                (0, 0),
            )

            assert chunk.input_data.shape[0] > 3
            assert chunk.input_data.shape[1] > 3

            truth = handler.data[lr_data_slice]
            padded_truth = np.pad(truth, pad_width, mode='constant')

            assert chunk.input_data.shape == padded_truth.shape
            assert np.allclose(chunk.input_data, padded_truth)
