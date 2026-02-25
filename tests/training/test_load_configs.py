"""Test the sample super resolution GAN configs"""

import os

import numpy as np
import pytest
import tensorflow as tf

from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan

ST_CONFIG_DIR = os.path.join(CONFIG_DIR, 'spatiotemporal/')
GEN_CONFIGS = [fn for fn in os.listdir(ST_CONFIG_DIR) if fn.startswith('gen')]


@pytest.mark.parametrize('spatial_len', (5, 6, 7))
def test_load_spatial(spatial_len):
    """Test the loading of a sample the spatial gan model."""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_10x_2f.json')

    model = Sup3rGan(fp_gen, pytest.S_FP_DISC)

    coarse_shapes = [
        (32, spatial_len, spatial_len, 2),
        (16, 2 * spatial_len, 2 * spatial_len, 2),
    ]

    for coarse_shape in coarse_shapes:
        x = np.ones(coarse_shape)
        gen_out = model._tf_generate(x)

        assert len(gen_out.shape) == 4
        assert gen_out.shape[0] == coarse_shape[0]
        assert gen_out.shape[1] == 10 * coarse_shape[1]
        assert gen_out.shape[2] == 10 * coarse_shape[2]
        assert gen_out.shape[3] == coarse_shape[3]

        disc_out = model.discriminate(x)
        assert len(disc_out.shape) == 2
        assert disc_out.shape[0] == coarse_shape[0]
        assert disc_out.shape[1] == 1


def test_load_all_spatial_generators():
    """Test all generator configs in the spatial config dir"""
    s_config_dir = os.path.join(CONFIG_DIR, 'spatial/')

    gen_configs = [
        fn for fn in os.listdir(s_config_dir) if fn.startswith('gen')
    ]

    for fn in gen_configs:
        enhancements = [
            s for s in fn.replace('.json', '').split('_') if s.endswith('x')
        ]
        assert len(enhancements) == 1
        s_enhance = int(enhancements[0].strip('x'))

        n_features = [
            s for s in fn.replace('.json', '').split('_') if s.endswith('f')
        ]
        assert len(n_features) == 1
        n_features = int(n_features[0].strip('f'))

        fp_gen = os.path.join(s_config_dir, fn)
        model = Sup3rGan(fp_gen, pytest.S_FP_DISC)

        coarse_shape = (1, 5, 5, 2)
        x = np.ones(coarse_shape)

        for layer in model.generator:
            x = layer(x)

        assert len(x.shape) == 4
        assert x.shape[0] == coarse_shape[0]
        assert x.shape[1] == s_enhance * coarse_shape[1]
        assert x.shape[2] == s_enhance * coarse_shape[2]
        assert x.shape[3] == n_features


def test_load_spatiotemporal():
    """Test loading of a sample spatiotemporal gan model"""

    model = Sup3rGan(pytest.ST_FP_GEN, pytest.ST_FP_DISC)

    coarse_shape = (32, 5, 5, 4, 2)
    x = np.ones(coarse_shape)

    for layer in model.generator:
        x = layer(x)

    gen_out = tf.identity(x)

    assert len(gen_out.shape) == 5
    assert gen_out.shape[0] == coarse_shape[0]
    assert gen_out.shape[1] == 3 * coarse_shape[1]
    assert gen_out.shape[2] == 3 * coarse_shape[2]
    assert gen_out.shape[3] == 4 * coarse_shape[3]
    assert gen_out.shape[4] == coarse_shape[4]

    x = tf.identity(gen_out)
    for layer in model.discriminator:
        x = layer(x)
    disc_out = tf.identity(x)
    assert len(disc_out.shape) == 2
    assert disc_out.shape[0] == coarse_shape[0]
    assert disc_out.shape[1] == 1


def test_load_json5():
    """Test loading of a sample spatiotemporal gan model from JSON5 config"""

    model = Sup3rGan(pytest.ST_FP_GEN, pytest.ST_FP_DISC_JSON5)

    coarse_shape = (32, 5, 5, 4, 2)
    x = np.ones(coarse_shape)

    for layer in model.generator:
        x = layer(x)

    gen_out = tf.identity(x)

    assert len(gen_out.shape) == 5
    assert gen_out.shape[0] == coarse_shape[0]
    assert gen_out.shape[1] == 3 * coarse_shape[1]
    assert gen_out.shape[2] == 3 * coarse_shape[2]
    assert gen_out.shape[3] == 4 * coarse_shape[3]
    assert gen_out.shape[4] == coarse_shape[4]

    x = tf.identity(gen_out)
    for layer in model.discriminator:
        x = layer(x)
    disc_out = tf.identity(x)
    assert len(disc_out.shape) == 2
    assert disc_out.shape[0] == coarse_shape[0]
    assert disc_out.shape[1] == 1


@pytest.mark.parametrize('fn_gen', GEN_CONFIGS)
@pytest.mark.parametrize(
    'coarse_shape', ((1, 5, 5, 4, 2), (1, 7, 7, 9, 2), (3, 6, 6, 8, 2))
)
def test_load_all_st_generators(fn_gen, coarse_shape):
    """Test all generator configs in the spatiotemporal config dir"""
    fp_gen = os.path.join(ST_CONFIG_DIR, fn_gen)

    enhancements = [
        s for s in fn_gen.replace('.json', '').split('_') if s.endswith('x')
    ]
    assert len(enhancements) == 2
    s_enhance = int(enhancements[0].strip('x'))
    t_enhance = int(enhancements[1].strip('x'))

    n_features = [
        s for s in fn_gen.replace('.json', '').split('_') if s.endswith('f')
    ]
    assert len(n_features) == 1
    n_features = int(n_features[0].strip('f'))

    model = Sup3rGan(fp_gen, pytest.ST_FP_DISC)

    x = np.ones(coarse_shape)
    for layer in model.generator:
        x = layer(x)
    assert len(x.shape) == 5
    assert x.shape[0] == coarse_shape[0]
    assert x.shape[1] == s_enhance * coarse_shape[1]
    assert x.shape[2] == s_enhance * coarse_shape[2]
    assert x.shape[3] == t_enhance * coarse_shape[3]
    assert x.shape[4] == n_features
