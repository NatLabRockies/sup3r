"""Global pytest fixtures."""

import os
import re

import numpy as np
import pytest
from rex import ResourceX, init_logger

from sup3r import CONFIG_DIR, TEST_DATA_DIR
from sup3r.utilities.utilities import RANDOM_GENERATOR

GLOBAL_STATE = RANDOM_GENERATOR.bit_generator.state


@pytest.hookimpl
def pytest_configure(config):  # pylint: disable=unused-argument # noqa: ARG001
    """Global pytest config."""
    init_logger('sup3r', log_level='DEBUG')

    pytest.FP_WTK = os.path.join(TEST_DATA_DIR, 'test_wtk_co_2012.h5')
    pytest.FP_NSRDB = os.path.join(TEST_DATA_DIR, 'test_nsrdb_co_2018.h5')
    pytest.FPS_WTK = [
        pytest.FP_WTK,
        os.path.join(TEST_DATA_DIR, 'test_wtk_co_2013.h5'),
    ]
    pytest.FP_WTK_SURF = os.path.join(
        TEST_DATA_DIR, 'test_wtk_surface_vars.h5'
    )
    pytest.FP_ERA = os.path.join(TEST_DATA_DIR, 'test_era5_co_2012.nc')
    pytest.FP_WRF = os.path.join(TEST_DATA_DIR, 'test_wrf_2014-10-01_00_00_00')
    pytest.ST_FP_GEN = os.path.join(
        CONFIG_DIR, 'spatiotemporal', 'gen_3x_4x_2f.json'
    )
    pytest.S_FP_GEN = os.path.join(CONFIG_DIR, 'spatial', 'gen_2x_2f.json')

    # Note that disc should not use "same" zeros padding but easier to use this
    # for testing on small sample sizes.
    pytest.ST_FP_DISC = os.path.join(TEST_DATA_DIR, 'config_disc_st_test.json')
    pytest.S_FP_DISC = os.path.join(TEST_DATA_DIR, 'config_disc_s_test.json')
    pytest.ST_FP_DISC_JSON5 = os.path.join(
        TEST_DATA_DIR, 'config_disc_st_test.json5'
    )

    pytest.ST_FP_DISC_PROD = os.path.join(
        CONFIG_DIR, 'spatiotemporal/disc.json'
    )

    pytest.FPS_GCM = [
        os.path.join(TEST_DATA_DIR, 'ua_test.nc'),
        os.path.join(TEST_DATA_DIR, 'va_test.nc'),
        os.path.join(TEST_DATA_DIR, 'orog_test.nc'),
        os.path.join(TEST_DATA_DIR, 'zg_test.nc'),
    ]
    pytest.FP_UAS = os.path.join(TEST_DATA_DIR, 'uas_test.nc')
    pytest.FP_RSDS = os.path.join(TEST_DATA_DIR, 'rsds_test.nc')


@pytest.fixture(autouse=True)
def set_random_state():
    """Set random generator state for reproducibility across tests with random
    sampling."""
    RANDOM_GENERATOR.bit_generator.state = GLOBAL_STATE


@pytest.fixture(autouse=True)
def train_on_cpu():
    """Train on cpu for tests."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


@pytest.fixture(scope='package')
def gen_config_with_concat_masked():
    """Get generator config with custom concat masked layer."""

    def func():
        return [
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
            {'class': 'Sup3rConcatObs', 'name': 'u_10m_obs'},
            {'class': 'Sup3rConcatObs', 'name': 'v_10m_obs'},
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

    return func


@pytest.fixture(scope='package')
def gen_config_with_topo():
    """Get generator config with custom topo layer."""

    def func(CustomLayer):
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
            {'class': CustomLayer, 'name': 'topography'},
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

    return func


@pytest.fixture(scope='package')
def collect_check():
    """Collection check used in cli test and collection test."""

    def func(dummy_output, fp_out):
        (
            out_files,
            data,
            ws_true,
            wd_true,
            _,
            _,
            t_slices_hr,
            _,
            s_slices_hr,
            _,
            low_res_times,
        ) = dummy_output

        with ResourceX(fp_out) as fh:
            full_ti = fh.time_index
            combined_ti = []
            for _, f in enumerate(out_files):
                t_idx, s_idx = re.match(
                    r'.*_([0-9]+)_([0-9]+)\.\w+$', f
                ).groups()
                s1_idx = int(s_idx[:3])
                s2_idx = int(s_idx[3:])
                t_hr = t_slices_hr[int(t_idx)]
                s1_hr = s_slices_hr[s1_idx]
                s2_hr = s_slices_hr[s2_idx]
                with ResourceX(f) as fh_i:
                    if s1_idx == s2_idx == 0:
                        combined_ti += list(fh_i.time_index)

                    ws_i = np.transpose(
                        data[s1_hr, s2_hr, t_hr, 0], axes=(2, 0, 1)
                    )
                    wd_i = np.transpose(
                        data[s1_hr, s2_hr, t_hr, 1], axes=(2, 0, 1)
                    )
                    ws_i = ws_i.reshape(48, 625)
                    wd_i = wd_i.reshape(48, 625)
                    assert np.allclose(ws_i, fh_i['windspeed_100m'], atol=0.01)
                    assert np.allclose(
                        wd_i, fh_i['winddirection_100m'], atol=0.1
                    )

                    for k, v in fh_i.global_attrs.items():
                        assert k in fh.global_attrs, k
                        assert fh.global_attrs[k] == v, k

            assert len(full_ti) == len(combined_ti)
            assert len(full_ti) == 2 * len(low_res_times)
            wd_true = np.transpose(wd_true[..., 0], axes=(2, 0, 1))
            ws_true = np.transpose(ws_true[..., 0], axes=(2, 0, 1))
            wd_true = wd_true.reshape(96, 2500)
            ws_true = ws_true.reshape(96, 2500)
            assert np.allclose(ws_true, fh['windspeed_100m'], atol=0.01)
            assert np.allclose(wd_true, fh['winddirection_100m'], atol=0.1)

    return func
