"""pytests for sup3r cli"""

import glob
import json
import os
import tempfile
import traceback

import h5py
import numpy as np
import pytest
from click.testing import CliRunner
from rex import ResourceX

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.bias.bias_calc_cli import from_config as bias_main
from sup3r.bias.utilities import lin_bc
from sup3r.models.base import Sup3rGan
from sup3r.pipeline.forward_pass_cli import from_config as fwp_main
from sup3r.pipeline.pipeline_cli import from_config as pipe_main
from sup3r.postprocessing.data_collect_cli import from_config as dc_main
from sup3r.preprocessing import DataHandler
from sup3r.solar.solar_cli import from_config as solar_main
from sup3r.utilities.pytest.helpers import (
    make_fake_cs_ratio_files,
    make_fake_h5_chunks,
    make_fake_nc_file,
)
from sup3r.utilities.utilities import (
    RANDOM_GENERATOR,
    pd_date_range,
    xr_open_mfdataset,
)

FEATURES = ['u_100m', 'v_100m', 'pressure_0m']
fwp_chunk_shape = (4, 4, 6)
data_shape = (100, 100, 10)
shape = (8, 8)

FP_CS = os.path.join(TEST_DATA_DIR, 'test_nsrdb_clearsky_2018.h5')
GAN_META = {'s_enhance': 4, 't_enhance': 24}
LR_LAT = np.linspace(40, 39, 5)
LR_LON = np.linspace(-105.5, -104.3, 5)
LR_LON, LR_LAT = np.meshgrid(LR_LON, LR_LAT)
LR_LON = np.expand_dims(LR_LON, axis=2)
LR_LAT = np.expand_dims(LR_LAT, axis=2)
LOW_RES_LAT_LON = np.concatenate((LR_LAT, LR_LON), axis=2)
LOW_RES_TIMES = pd_date_range(
    '20500101', '20500104', inclusive='left', freq='1d'
)
HIGH_RES_TIMES = pd_date_range(
    '20500101', '20500104', inclusive='left', freq='1h'
)


@pytest.fixture(scope='module')
def input_files(tmpdir_factory):
    """Dummy netcdf input files for fwp testing"""

    input_file = str(tmpdir_factory.mktemp('data').join('fwp_input.nc'))
    make_fake_nc_file(input_file, shape=data_shape, features=FEATURES)
    return input_file


@pytest.fixture(scope='module')
def runner():
    """Cli runner helper utility."""
    return CliRunner()


def test_pipeline_fwp_collect(runner, input_files):
    """Test pipeline with forward pass and data collection"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.meta['lr_features'] = FEATURES
    model.meta['hr_out_features'] = FEATURES[:2]
    model.meta['s_enhance'] = 3
    model.meta['t_enhance'] = 4

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        fp_out = os.path.join(td, 'fwp_combined.h5')
        n_nodes = len(input_files) // fwp_chunk_shape[2] + 1
        n_nodes *= shape[0] // fwp_chunk_shape[0]
        n_nodes *= shape[1] // fwp_chunk_shape[1]
        out_files = os.path.join(td, 'out_{file_id}.h5')
        fwp_config = {
            'input_handler_kwargs': {'target': (19.3, -123.5), 'shape': shape},
            'file_paths': input_files,
            'model_kwargs': {'model_dir': out_dir},
            'out_pattern': out_files,
            'fwp_chunk_shape': fwp_chunk_shape,
            'spatial_pad': 1,
            'temporal_pad': 1,
            'execution_control': {'option': 'local'},
        }

        features = ['windspeed_100m', 'winddirection_100m']
        out_files = os.path.join(td, 'out_*.h5')
        dc_config = {
            'file_paths': out_files,
            'out_file': fp_out,
            'features': features,
            'execution_control': {'option': 'local'},
        }

        fwp_config_path = os.path.join(td, 'config_fwp.json')
        dc_config_path = os.path.join(td, 'config_dc.json')
        pipe_config_path = os.path.join(td, 'config_pipe.json')

        pipe_config = {
            'pipeline': [
                {'forward-pass': fwp_config_path},
                {'data-collect': dc_config_path},
            ]
        }

        with open(fwp_config_path, 'w') as fh:
            json.dump(fwp_config, fh)
        with open(dc_config_path, 'w') as fh:
            json.dump(dc_config, fh)
        with open(pipe_config_path, 'w') as fh:
            json.dump(pipe_config, fh)

        result = runner.invoke(
            pipe_main, ['-c', pipe_config_path, '-v', '--monitor']
        )
        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert os.path.exists(fp_out)
        with ResourceX(fp_out) as fh:
            assert all(f in fh for f in features)
            full_ti = fh.time_index
            full_gids = fh.meta['gid']
            combined_ti = []
            for _, f in enumerate(glob.glob(out_files)):
                with ResourceX(f) as fh_i:
                    fi_ti = fh_i.time_index
                    fi_gids = fh_i.meta['gid']
                    assert all(gid in full_gids for gid in fi_gids)
                    s_indices = np.where(full_gids.isin(fi_gids))[0]
                    t_indices = np.where(full_ti.isin(fi_ti))[0]
                    t_indices = slice(t_indices[0], t_indices[-1] + 1)
                    chunk = fh['windspeed_100m'][t_indices, s_indices]
                    assert np.allclose(chunk, fh_i['windspeed_100m'])
                    chunk = fh['winddirection_100m'][t_indices, s_indices]
                    assert np.allclose(chunk, fh_i['winddirection_100m'])
                    combined_ti += list(fh_i.time_index)
            assert len(full_ti) == len(set(combined_ti))


def test_data_collection_cli(runner, collect_check):
    """Test cli call to data collection on forward pass output"""

    with tempfile.TemporaryDirectory() as td:
        fp_out = os.path.join(td, 'out_combined.h5')
        out = make_fake_h5_chunks(td)
        out_files = out[0]

        features = ['windspeed_100m', 'winddirection_100m']
        config = {
            'file_paths': out_files,
            'out_file': fp_out,
            'features': features,
            'execution_control': {'option': 'local'},
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(dc_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert os.path.exists(fp_out)

        collect_check(out, fp_out)


def test_fwd_pass_with_bc_cli(runner, input_files):
    """Test cli call to run forward pass with bias correction"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES[:2],
        s_enhance=3,
        t_enhance=4,
        input_resolution={'spatial': '12km', 'temporal': '60min'},
    )

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        n_chunks = np.prod([
            int(np.ceil(ds / fs))
            for ds, fs in zip([*shape, data_shape[2]], fwp_chunk_shape)
        ])
        out_files = os.path.join(td, 'out_{file_id}.nc')
        cache_pattern = os.path.join(td, 'cache_{feature}.nc')
        log_pattern = os.path.join(td, 'logs', 'log_{node_index}.log')

        input_handler_kwargs = {
            'target': (19.3, -123.5),
            'shape': shape,
            'cache_kwargs': {'cache_pattern': cache_pattern, 'max_workers': 1},
        }

        lat_lon = DataHandler(
            file_paths=input_files, features=[], **input_handler_kwargs
        ).lat_lon

        bias_fp = os.path.join(td, 'bc.h5')

        scalar = RANDOM_GENERATOR.uniform(0.5, 1, (8, 8, 12))
        adder = RANDOM_GENERATOR.uniform(0, 1, (8, 8, 12))

        with h5py.File(bias_fp, 'w') as f:
            f.create_dataset('u_100m_scalar', data=scalar)
            f.create_dataset('u_100m_adder', data=adder)
            f.create_dataset('v_100m_scalar', data=scalar)
            f.create_dataset('v_100m_adder', data=adder)
            f.create_dataset('latitude', data=lat_lon[..., 0])
            f.create_dataset('longitude', data=lat_lon[..., 1])

        bias_correct_kwargs = {
            'u_100m': {
                'feature_name': 'u_100m',
                'bias_fp': bias_fp,
                'smoothing': 0,
                'temporal_avg': False,
                'out_range': [-100, 100],
            },
            'v_100m': {
                'feature_name': 'v_100m',
                'smoothing': 0,
                'bias_fp': bias_fp,
                'temporal_avg': False,
                'out_range': [-100, 100],
            },
        }

        config = {
            'file_paths': input_files,
            'model_kwargs': {'model_dir': out_dir},
            'out_pattern': out_files,
            'log_pattern': log_pattern,
            'fwp_chunk_shape': fwp_chunk_shape,
            'input_handler_name': 'DataHandler',
            'input_handler_kwargs': input_handler_kwargs.copy(),
            'spatial_pad': 1,
            'temporal_pad': 1,
            'bias_correct_kwargs': bias_correct_kwargs.copy(),
            'bias_correct_method': 'monthly_local_linear_bc',
            'execution_control': {'option': 'local'},
            'max_nodes': 2,
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(fwp_main, ['-c', config_path, '-v'])

        assert result.exit_code == 0, traceback.print_exception(
            *result.exc_info
        )

        assert len(glob.glob(f'{td}/cache*')) == len(FEATURES)
        assert len(glob.glob(f'{td}/logs/log_*.log')) == config['max_nodes']
        assert len(glob.glob(f'{td}/out*')) == n_chunks


def test_fwd_pass_cli(runner, input_files):
    """Test cli call to run forward pass"""

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))
    model.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES[:2],
        s_enhance=3,
        t_enhance=4,
        input_resolution={'spatial': '12km', 'temporal': '60min'},
    )

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)
        n_chunks = np.prod([
            int(np.ceil(ds / fs))
            for ds, fs in zip([*shape, data_shape[2]], fwp_chunk_shape)
        ])
        out_files = os.path.join(td, 'out_{file_id}.nc')
        cache_pattern = os.path.join(td, 'cache_{feature}.nc')
        log_pattern = os.path.join(td, 'logs', 'log_{node_index}.log')
        input_handler_kwargs = {
            'target': (19.3, -123.5),
            'shape': shape,
            'cache_kwargs': {'cache_pattern': cache_pattern, 'max_workers': 1},
        }
        config = {
            'file_paths': input_files,
            'model_kwargs': {'model_dir': out_dir},
            'out_pattern': out_files,
            'log_pattern': log_pattern,
            'input_handler_kwargs': input_handler_kwargs,
            'input_handler_name': 'DataHandler',
            'fwp_chunk_shape': fwp_chunk_shape,
            'pass_workers': 1,
            'spatial_pad': 1,
            'temporal_pad': 1,
            'execution_control': {'option': 'local'},
            'max_nodes': 5,
        }

        config_path = os.path.join(td, 'config.json')
        with open(config_path, 'w') as fh:
            json.dump(config, fh)

        result = runner.invoke(fwp_main, ['-c', config_path, '-v'])

        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert len(glob.glob(f'{td}/cache*')) == len(FEATURES)
        assert len(glob.glob(f'{td}/logs/log_*.log')) == config['max_nodes']
        assert len(glob.glob(f'{td}/out*')) == n_chunks


def test_pipeline_fwp_qa(runner, input_files):
    """Test the sup3r pipeline with Forward Pass and QA modules
    via pipeline cli"""

    Sup3rGan.seed()
    model = Sup3rGan(pytest.ST_FP_GEN, pytest.ST_FP_DISC, learning_rate=1e-4)
    model.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES[:2],
        s_enhance=3,
        t_enhance=4,
        input_resolution={'spatial': '12km', 'temporal': '60min'},
    )
    input_resolution = {'spatial': '12km', 'temporal': '60min'}
    assert model.input_resolution == input_resolution
    assert model.output_resolution == {'spatial': '4km', 'temporal': '15min'}
    _ = model.generate(np.ones((4, 8, 8, 4, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        out_dir = os.path.join(td, 'st_gan')
        model.save(out_dir)

        input_handler_kwargs = {
            'target': (19.3, -123.5),
            'shape': shape,
        }

        fwp_config = {
            'file_paths': input_files,
            'model_kwargs': {'model_dir': out_dir},
            'out_pattern': os.path.join(td, 'out_{file_id}.h5'),
            'log_pattern': os.path.join(td, 'fwp_log.log'),
            'log_level': 'DEBUG',
            'input_handler_kwargs': input_handler_kwargs,
            'fwp_chunk_shape': (100, 100, 100),
            'max_workers': 1,
            'spatial_pad': 1,
            'temporal_pad': 1,
            'execution_control': {'option': 'local'},
        }

        qa_config = {
            'source_file_paths': input_files,
            'out_file_path': os.path.join(td, 'out_000000_000000.h5'),
            'qa_fp': os.path.join(td, 'qa.h5'),
            's_enhance': 3,
            't_enhance': 4,
            'temporal_coarsening_method': 'subsample',
            'input_handler_kwargs': input_handler_kwargs,
            'max_workers': 1,
            'execution_control': {'option': 'local'},
        }

        fwp_config_path = os.path.join(td, 'config_fwp.json')
        qa_config_path = os.path.join(td, 'config_qa.json')
        pipe_config_path = os.path.join(td, 'config_pipe.json')

        pipe_flog = os.path.join(td, 'pipeline.log')
        pipe_config = {
            'logging': {'log_level': 'DEBUG', 'log_file': pipe_flog},
            'pipeline': [
                {'forward-pass': fwp_config_path},
                {'qa': qa_config_path},
            ],
        }

        with open(fwp_config_path, 'w') as fh:
            json.dump(fwp_config, fh)
        with open(qa_config_path, 'w') as fh:
            json.dump(qa_config, fh)
        with open(pipe_config_path, 'w') as fh:
            json.dump(pipe_config, fh)

        result = runner.invoke(
            pipe_main, ['-c', pipe_config_path, '-v', '--monitor']
        )
        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert len(glob.glob(f'{td}/fwp_log*.log')) == 1
        assert len(glob.glob(f'{td}/out*.h5')) == 1
        assert len(glob.glob(f'{td}/qa.h5')) == 1
        status_fps = glob.glob(f'{td}/.gaps/*status*.json')
        assert len(status_fps) == 1
        status_fp = status_fps[0]
        with open(status_fp) as f:
            status = json.load(f)

        fwp_status = status['forward-pass']
        del fwp_status['pipeline_index']
        fwp_status = next(iter(fwp_status.values()))
        assert fwp_status['job_status'] == 'successful'
        assert fwp_status['time'] > 0

        assert len(status['qa']) == 2
        qa_status = status['qa']
        del qa_status['pipeline_index']
        qa_status = next(iter(qa_status.values()))
        assert qa_status['job_status'] == 'successful'
        assert qa_status['time'] > 0


@pytest.mark.parametrize(
    'bias_calc_class',
    [
        'LinearCorrection',
        'ScalarCorrection',
        'MonthlyLinearCorrection',
        'MonthlyScalarCorrection',
    ],
)
def test_cli_bias_calc(runner, bias_calc_class):
    """Test cli for bias correction"""

    with xr_open_mfdataset(pytest.FP_RSDS) as fh:
        MIN_LAT = np.min(fh.lat.values.astype(np.float32))
        MIN_LON = np.min(fh.lon.values.astype(np.float32)) - 360
        TARGET = (float(MIN_LAT), float(MIN_LON))
        SHAPE = (len(fh.lat.values), len(fh.lon.values))

    with tempfile.TemporaryDirectory() as td:
        fp_out = f'{td}/bc_file.h5'
        bc_config = {
            'bias_calc_class': bias_calc_class,
            'jobs': [
                {
                    'base_fps': [pytest.FP_NSRDB],
                    'bias_fps': [pytest.FP_RSDS],
                    'base_dset': 'ghi',
                    'bias_feature': 'rsds',
                    'target': TARGET,
                    'shape': SHAPE,
                    'max_workers': 2,
                    'fp_out': fp_out,
                }
            ],
            'execution_control': {
                'option': 'local',
            },
        }

        bc_config_path = os.path.join(td, 'config_bc.json')

        with open(bc_config_path, 'w') as fh:
            json.dump(bc_config, fh)

        result = runner.invoke(bias_main, ['-c', bc_config_path, '-v'])
        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)

        assert os.path.exists(fp_out)

        handler = DataHandler(
            pytest.FP_RSDS, features=['rsds'], target=TARGET, shape=SHAPE
        )
        og_data = handler['rsds'][...].copy()
        lin_bc(handler, bc_files=[fp_out])
        bc_data = handler['rsds'][...].copy()

        assert not np.array_equal(bc_data, og_data)


def test_cli_solar(runner):
    """Test cli for bias correction"""

    with tempfile.TemporaryDirectory() as td:
        fps, _ = make_fake_cs_ratio_files(
            td, LOW_RES_TIMES, LOW_RES_LAT_LON, model_meta=GAN_META
        )

        solar_config = {
            'fp_pattern': fps,
            'nsrdb_fp': FP_CS,
            'execution_control': {
                'option': 'local',
            },
        }

        solar_config_path = os.path.join(td, 'config_solar.json')

        with open(solar_config_path, 'w') as fh:
            json.dump(solar_config, fh)

        result = runner.invoke(solar_main, ['-c', solar_config_path, '-v'])
        if result.exit_code != 0:
            msg = 'Failed with error {}'.format(
                traceback.print_exception(*result.exc_info)
            )
            raise RuntimeError(msg)
