"""Test forward passes through multi-step GAN models"""

import os
import tempfile

import numpy as np
import pytest

from sup3r import CONFIG_DIR
from sup3r.models import (
    LinearInterp,
    MultiStepGan,
    SolarMultiStepGan,
    Sup3rGan,
)

FEATURES = ['u_100m', 'v_100m']


@pytest.mark.parametrize('features', [FEATURES, FEATURES[:-1]])
def test_multi_step_model(features):
    """Test a basic forward pass through a multi step model with 2 steps.
    Include a test where the first step outputs more features than needed for
    the second step."""
    Sup3rGan.seed(0)

    fp_gen1 = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_gen2 = os.path.join(
        CONFIG_DIR, f'spatiotemporal/gen_3x_4x_{len(features)}f.json'
    )
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    model1 = Sup3rGan(fp_gen1, fp_disc)
    model2 = Sup3rGan(fp_gen2, fp_disc)

    model1.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '27km', 'temporal': '64min'},
        s_enhance=3,
        t_enhance=4,
    )
    model2.set_model_params(
        lr_features=features,
        hr_out_features=features,
        input_resolution={'spatial': '9km', 'temporal': '16min'},
        s_enhance=3,
        t_enhance=4,
    )

    _ = model1.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    _ = model2.generate(np.ones((4, 10, 10, 6, len(features))))

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        model1.save(fp1)
        model2.save(fp2)

        ms_model = MultiStepGan.load([fp1, fp2])

        x = np.ones((4, 5, 5, 6, len(FEATURES)))
        out = ms_model.generate(x)
        out1 = model1.generate(x)
        out2 = model2.generate(out1[..., : len(features)])

        assert out.shape == (4, 45, 45, 96, len(features))
        assert np.allclose(out, out2)


@pytest.mark.parametrize('norm_option', ['same_stats', 'diff_stats'])
def test_multi_step_norm(norm_option):
    """Test the multi step model with 3 GAN's with the same and different norm
    stats for each model"""
    Sup3rGan.seed(0)
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')

    model1 = Sup3rGan(fp_gen, fp_disc)
    model2 = Sup3rGan(fp_gen, fp_disc)
    model3 = Sup3rGan(fp_gen, fp_disc)

    if norm_option == 'diff_stats':
        # models have different norm stats
        model1.set_norm_stats(
            {'u_100m': 0.1, 'v_100m': 0.2}, {'u_100m': 0.04, 'v_100m': 0.02}
        )
        model2.set_norm_stats(
            {'u_100m': 0.1, 'v_100m': 0.2}, {'u_100m': 0.04, 'v_100m': 0.02}
        )
        model3.set_norm_stats(
            {'u_100m': 0.3, 'v_100m': 0.9}, {'u_100m': 0.02, 'v_100m': 0.07}
        )
    else:
        # all models have the same norm stats
        model1.set_norm_stats(
            {'u_100m': 0.1, 'v_100m': 0.8}, {'u_100m': 0.04, 'v_100m': 0.02}
        )
        model2.set_norm_stats(
            {'u_100m': 0.1, 'v_100m': 0.8}, {'u_100m': 0.04, 'v_100m': 0.02}
        )
        model3.set_norm_stats(
            {'u_100m': 0.1, 'v_100m': 0.8}, {'u_100m': 0.04, 'v_100m': 0.02}
        )

    model1.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '27km', 'temporal': '64min'},
        s_enhance=3,
        t_enhance=4,
    )
    model2.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '9km', 'temporal': '16min'},
        s_enhance=3,
        t_enhance=4,
    )
    model3.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '3km', 'temporal': '4min'},
        s_enhance=3,
        t_enhance=4,
    )

    _ = model1.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    _ = model2.generate(np.ones((4, 10, 10, 6, len(FEATURES))))
    _ = model3.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        fp3 = os.path.join(td, 'model3')
        model1.save(fp1)
        model2.save(fp2)
        model3.save(fp3)

        ms_model = MultiStepGan.load([fp1, fp2, fp3])

        x = np.ones((1, 4, 4, 4, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (1, 108, 108, 256, len(FEATURES))

        # make sure the multistep generate is the same as several manual
        # norm_in/unnorm_out forward passes
        out1 = model1.generate(x)
        out2 = model2.generate(out1)
        out3 = model3.generate(out2)
        err = np.abs(out - out3).mean() / np.abs(out.mean())
        assert err < 1e4


def test_spatial_then_temporal_gan():
    """Test the 2-step spatial-then-spatiotemporal GAN"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model1 = Sup3rGan(fp_gen, fp_disc)
    _ = model1.generate(np.ones((4, 10, 10, len(FEATURES))))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model2 = Sup3rGan(fp_gen, fp_disc)
    _ = model2.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    model1.set_norm_stats(
        {'u_100m': 0.1, 'v_100m': 0.2}, {'u_100m': 0.04, 'v_100m': 0.02}
    )
    model2.set_norm_stats(
        {'u_100m': 0.3, 'v_100m': 0.9}, {'u_100m': 0.02, 'v_100m': 0.07}
    )

    model1.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '12km', 'temporal': '40min'},
        s_enhance=2,
        t_enhance=1,
    )
    model2.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '6km', 'temporal': '40min'},
        s_enhance=3,
        t_enhance=4,
    )

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        model1.save(fp1)
        model2.save(fp2)

        ms_model = MultiStepGan.load([fp1, fp2])

        x = np.ones((4, 10, 10, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (1, 60, 60, 16, 2)


def test_temporal_then_spatial_gan():
    """Test the 2-step temporal-then-spatial GAN"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model1 = Sup3rGan(fp_gen, fp_disc)
    _ = model1.generate(np.ones((4, 10, 10, len(FEATURES))))

    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    model2 = Sup3rGan(fp_gen, fp_disc)
    _ = model2.generate(np.ones((4, 10, 10, 6, len(FEATURES))))

    model1.set_norm_stats(
        {'u_100m': 0.1, 'v_100m': 0.2}, {'u_100m': 0.04, 'v_100m': 0.02}
    )
    model2.set_norm_stats(
        {'u_100m': 0.3, 'v_100m': 0.9}, {'u_100m': 0.02, 'v_100m': 0.07}
    )

    model1.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '12km', 'temporal': '40min'},
        s_enhance=2,
        t_enhance=1,
    )
    model2.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '6km', 'temporal': '40min'},
        s_enhance=3,
        t_enhance=4,
    )

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        model1.save(fp1)
        model2.save(fp2)

        ms_model = MultiStepGan.load([fp2, fp1])

        x = np.ones((1, 10, 10, 4, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (16, 60, 60, 2)


def test_spatial_gan_then_linear_interp():
    """Test the 2-step spatial GAN then linear spatiotemporal interpolation"""
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model1 = Sup3rGan(fp_gen, fp_disc)
    _ = model1.generate(np.ones((4, 10, 10, len(FEATURES))))

    model2 = LinearInterp(lr_features=FEATURES, s_enhance=3, t_enhance=4)

    model1.set_norm_stats(
        {'u_100m': 0.1, 'v_100m': 0.2}, {'u_100m': 0.04, 'v_100m': 0.02}
    )
    model1.set_model_params(
        lr_features=FEATURES,
        hr_out_features=FEATURES,
        input_resolution={'spatial': '12km', 'temporal': '60min'},
        s_enhance=2,
        t_enhance=1,
    )

    with tempfile.TemporaryDirectory() as td:
        fp1 = os.path.join(td, 'model1')
        fp2 = os.path.join(td, 'model2')
        model1.save(fp1)
        model2.save(fp2)

        ms_model = MultiStepGan.load([fp1, fp2])

        x = np.ones((4, 10, 10, len(FEATURES)))
        out = ms_model.generate(x)
        assert out.shape == (1, 60, 60, 16, 2)


def test_solar_multistep():
    """Test the special solar multistep model that uses parallel solar+wind
    spatial enhancement models that join to a single solar spatiotemporal
    model."""
    features1 = ['clearsky_ratio']
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_1f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model1 = Sup3rGan(fp_gen, fp_disc)
    _ = model1.generate(np.ones((4, 10, 10, len(features1))))
    model1.set_norm_stats({'clearsky_ratio': 0.7}, {'clearsky_ratio': 0.04})
    model1.set_model_params(
        lr_features=features1,
        hr_out_features=features1,
        s_enhance=2,
        t_enhance=1,
        input_resolution={'spatial': '8km', 'temporal': '40min'},
    )

    features2 = ['U_200m', 'V_200m']
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model2 = Sup3rGan(fp_gen, fp_disc)
    _ = model2.generate(np.ones((4, 10, 10, len(features2))))
    model2.set_norm_stats(
        {'U_200m': 4.2, 'V_200m': 5.6}, {'U_200m': 1.1, 'V_200m': 1.3}
    )
    model2.set_model_params(
        lr_features=features2,
        hr_out_features=features2,
        s_enhance=2,
        t_enhance=1,
        input_resolution={'spatial': '4km', 'temporal': '40min'},
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
    model3.set_model_params(
        lr_features=features_in_3,
        hr_out_features=features_out_3,
        s_enhance=1,
        t_enhance=8,
        input_resolution={'spatial': '2km', 'temporal': '40min'},
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
        out = ms_model.generate(x)
        assert out.shape == (1, 20, 20, 24, 1)
