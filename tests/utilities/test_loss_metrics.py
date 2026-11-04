"""Tests for GAN loss functions"""

import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError

from sup3r import CONFIG_DIR
from sup3r.models import Sup3rGan
from sup3r.utilities.loss_metrics import (
    CoarseMseLoss,
    GeothermalMohoBCLoss,
    GeothermalNonnegativeHeatProductionLoss,
    GeothermalPositiveTemperatureGradientLoss,
    GeothermalSurfaceHeatFluxLoss,
    LowResLoss,
    MaterialDerivativeLoss,
    MmdLoss,
    ObsAssimilationLoss,
    SpatialExtremesLoss,
    SpatiotemporalFftLoss,
    TemporalExtremesLoss,
    _reshape_depth_feature_for_vertical_derivative,
    tf_derivative,
)
from sup3r.utilities.utilities import (
    RANDOM_GENERATOR,
    spatial_coarsening,
    temporal_coarsening,
)


def test_mmd_loss():
    """Test content loss using mse + mmd for content loss."""

    x = np.zeros((6, 10, 10, 8, 3))
    y = np.zeros((6, 10, 10, 8, 3))
    x[:, 7:9, 7:9, :, :] = 1
    y[:, 2:5, 2:5, :, :] = 1

    # distributions differing by only a small peak should give small mse and
    # larger mmd
    mse_fun = tf.keras.losses.MeanSquaredError()
    mmd_fun = MmdLoss()

    mse = mse_fun(x, y)
    mmd_plus_mse = (mmd_fun(x, y) + mse) / 2

    assert mmd_plus_mse > mse

    x = RANDOM_GENERATOR.random((6, 10, 10, 8, 3))
    x /= np.max(x)
    y = RANDOM_GENERATOR.random((6, 10, 10, 8, 3))
    y /= np.max(y)

    # scaling the same distribution should give high mse and smaller mmd
    mse = mse_fun(5 * x, x)
    mmd_plus_mse = (mmd_fun(5 * x, x) + mse) / 2

    assert mmd_plus_mse < mse


def test_coarse_mse_loss():
    """Test the coarse MSE loss on spatial average data"""
    x = RANDOM_GENERATOR.uniform(0, 1, (6, 10, 10, 8, 3))
    y = RANDOM_GENERATOR.uniform(0, 1, (6, 10, 10, 8, 3))

    mse_fun = tf.keras.losses.MeanSquaredError()
    cmse_fun = CoarseMseLoss()

    mse = mse_fun(x, y)
    coarse_mse = cmse_fun(x, y)

    assert isinstance(mse, tf.Tensor)
    assert isinstance(coarse_mse, tf.Tensor)
    assert mse.numpy().size == 1
    assert coarse_mse.numpy().size == 1
    assert mse.numpy() > 10 * coarse_mse.numpy()


def test_tex_loss():
    """Test custom TemporalExtremesLoss function that looks at min/max values
    in the timeseries."""
    loss_obj = TemporalExtremesLoss()

    x = np.zeros((1, 1, 1, 72, 1))
    y = np.zeros((1, 1, 1, 72, 1))

    # loss should be dominated by special min/max values
    x[..., 24, 0] = 20
    y[..., 25, 0] = 25
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5

    # loss should be dominated by special min/max values
    x[..., 24, 0] = -20
    y[..., 25, 0] = -25
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5


def test_spex_loss():
    """Test custom SpatialExtremesLoss function that looks at min/max values
    in the timeseries."""
    loss_obj = SpatialExtremesLoss()

    x = np.zeros((1, 10, 10, 2, 1))
    y = np.zeros((1, 10, 10, 2, 1))

    # loss should be dominated by special min/max values
    x[:, 5, 5, :, 0] = 20
    y[:, 5, 5, :, 0] = 25
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5

    # loss should be dominated by special min/max values
    x[:, 5, 5, :, 0] = -20
    y[:, 5, 5, :, 0] = -25
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5


def test_stex_loss():
    """Test custom SpatioTemporalExtremesLoss function that looks at min/max
    values in the timeseries."""

    def loss_obj(x, y):
        loss = (
            MeanAbsoluteError()(x, y)
            + SpatialExtremesLoss()(x, y)
            + TemporalExtremesLoss()(x, y)
        )
        return 1 / 3 * loss

    x = np.zeros((1, 10, 10, 5, 1))
    y = np.zeros((1, 10, 10, 5, 1))

    # loss should be dominated by special min/max values
    x[:, 5, 5, 2, 0] = 100
    y[:, 5, 5, 2, 0] = 150
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5

    # loss should be dominated by special min/max values
    x[:, 5, 5, 2, 0] = -100
    y[:, 5, 5, 2, 0] = -150
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.5


def test_st_fft_loss():
    """Test custom StExtremesFftLoss function that looks at min/max
    values in the timeseries and also encourages accuracy of the frequency
    spectrum"""

    def loss_obj(x, y):
        loss = (
            SpatiotemporalFftLoss()(x, y)
            + SpatialExtremesLoss()(x, y)
            + TemporalExtremesLoss()(x, y)
            + MeanAbsoluteError()(x, y)
        )
        return 1 / 4 * loss

    x = np.zeros((1, 10, 10, 5, 1))
    y = np.zeros((1, 10, 10, 5, 1))

    # loss should be dominated by special min/max values
    x[:, 5, 5, 2, 0] = 100
    y[:, 5, 5, 2, 0] = 150
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.0

    # loss should be dominated by special min/max values
    x[:, 5, 5, 2, 0] = -100
    y[:, 5, 5, 2, 0] = -150
    loss = loss_obj(x, y)
    assert loss.numpy() > 1.0


def test_lr_loss():
    """Test custom LowResLoss that re-coarsens synthetic and true high-res
    fields and calculates pointwise loss on the low-res fields"""

    # test w/o enhance
    t_meth = 'average'
    loss_obj = LowResLoss(
        s_enhance=1, t_enhance=1, t_method=t_meth, tf_loss='MeanSquaredError'
    )
    xarr = RANDOM_GENERATOR.uniform(-1, 1, (3, 10, 10, 48, 2))
    yarr = RANDOM_GENERATOR.uniform(-1, 1, (3, 10, 10, 48, 2))
    xtensor = tf.convert_to_tensor(xarr)
    ytensor = tf.convert_to_tensor(yarr)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xtensor, ytensor))

    # test 5D with s_enhance
    s_enhance = 5
    loss_obj = LowResLoss(
        s_enhance=s_enhance,
        t_enhance=1,
        t_method=t_meth,
        tf_loss='MeanSquaredError',
    )
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 5D with s/t enhance
    s_enhance = 5
    t_enhance = 12
    loss_obj = LowResLoss(
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        t_method=t_meth,
        tf_loss='MeanSquaredError',
    )
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    xarr_lr = temporal_coarsening(xarr_lr, t_enhance=t_enhance, method=t_meth)
    yarr_lr = temporal_coarsening(yarr_lr, t_enhance=t_enhance, method=t_meth)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 5D with subsample
    t_meth = 'subsample'
    loss_obj = LowResLoss(
        s_enhance=s_enhance,
        t_enhance=t_enhance,
        t_method=t_meth,
        tf_loss='MeanSquaredError',
    )
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    xarr_lr = temporal_coarsening(xarr_lr, t_enhance=t_enhance, method=t_meth)
    yarr_lr = temporal_coarsening(yarr_lr, t_enhance=t_enhance, method=t_meth)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 4D spatial only
    xarr = RANDOM_GENERATOR.uniform(-1, 1, (3, 10, 10, 2))
    yarr = RANDOM_GENERATOR.uniform(-1, 1, (3, 10, 10, 2))
    xtensor = tf.convert_to_tensor(xarr)
    ytensor = tf.convert_to_tensor(yarr)
    s_enhance = 5
    loss_obj = LowResLoss(
        s_enhance=s_enhance,
        t_enhance=1,
        t_method=t_meth,
        tf_loss='MeanSquaredError',
    )
    xarr_lr = spatial_coarsening(xarr, s_enhance=s_enhance, obs_axis=True)
    yarr_lr = spatial_coarsening(yarr, s_enhance=s_enhance, obs_axis=True)
    loss = loss_obj(xtensor, ytensor)
    assert np.allclose(loss, loss_obj._tf_loss(xarr_lr, yarr_lr))

    # test 4D spatial only with spatial extremes
    loss_obj = LowResLoss(
        s_enhance=s_enhance,
        t_enhance=1,
        t_method=t_meth,
        tf_loss='MeanSquaredError',
        ex_loss='SpatialExtremesLoss',
    )
    ex_loss = loss_obj(xtensor, ytensor)
    assert ex_loss > loss


def test_obs_assimilation_loss_no_obs_matches_background_mae():
    """Obs assimilation loss should reduce to background MAE without obs."""
    x_true_bg = np.arange(25, dtype=np.float32).reshape(1, 5, 5, 1)
    x_obs = np.full_like(x_true_bg, np.nan)
    x_true = np.concatenate([x_true_bg, x_obs], axis=-1)
    x_gen = x_true_bg + 2.0

    loss = ObsAssimilationLoss(
        gen_features=['f'],
        true_features=['f', 'f_obs'],
        blend_distance=2,
    )(x_true, x_gen)

    assert np.allclose(loss, MeanAbsoluteError()(x_true_bg, x_gen))


def test_obs_assimilation_loss_blends_obs_into_background():
    """Obs assimilation loss should apply Gaussian blending around obs."""
    x_true_bg = np.zeros((1, 5, 5, 1), dtype=np.float32)
    x_obs = np.full_like(x_true_bg, np.nan)
    x_obs[:, 2, 2, 0] = 1.0
    x_true = np.concatenate([x_true_bg, x_obs], axis=-1)
    x_gen = np.zeros_like(x_true_bg)

    loss = ObsAssimilationLoss(
        gen_features=['f'],
        true_features=['f', 'f_obs'],
        blend_distance=1,
    )(x_true, x_gen)

    edge_weight = np.exp(-2.0)
    corner_weight = np.exp(-4.0)
    expected = (1.0 + 4 * edge_weight + 4 * corner_weight) / 25.0

    assert np.allclose(loss, expected)


def test_md_loss():
    """Test the material derivative calculation in the material derivative
    content loss class."""

    x = RANDOM_GENERATOR.random((6, 10, 10, 8, 2))
    y = x.copy()

    md_loss = MaterialDerivativeLoss(gen_features=['u_100m', 'v_100m'])
    u_div = md_loss._compute_md(x, feature='u_100m')
    v_div = md_loss._compute_md(x, feature='v_100m')

    u_div_np = np.gradient(y[..., 0], axis=3)
    u_div_np += y[..., 0] * np.gradient(y[..., 0], axis=1)
    u_div_np += y[..., 1] * np.gradient(y[..., 0], axis=2)

    v_div_np = np.gradient(x[..., 1], axis=3)
    v_div_np += y[..., 0] * np.gradient(y[..., 1], axis=1)
    v_div_np += y[..., 1] * np.gradient(y[..., 1], axis=2)

    with pytest.raises(ValueError):
        tf_derivative(x, axis=0)

    with pytest.raises(ValueError):
        md_loss(x[..., 0], y[..., 0])

    assert np.allclose(u_div, u_div_np)
    assert np.allclose(v_div, v_div_np)


def test_multiterm_loss():
    """Test multi-term loss functionality."""

    x = RANDOM_GENERATOR.random((6, 10, 10, 8, 3))
    y = x.copy()

    md_loss = MaterialDerivativeLoss(
        gen_features=['u_100m', 'v_100m', 'temp_100m']
    )
    mae_loss = MeanAbsoluteError()
    fp_gen = os.path.join(CONFIG_DIR, 'spatial/gen_2x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatial/disc.json')
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    model.set_model_params(
        lr_features=['u_100m', 'v_100m', 'temp_100m'],
        hr_out_features=['u_100m', 'v_100m', 'temp_100m'],
        input_resolution={'spatial': '12km', 'temporal': '60min'},
        s_enhance=1,
        t_enhance=1,
    )
    multi_loss = model.get_loss_fun(
        {
            'MaterialDerivativeLoss': {
                'weight': 0.2,
                'gen_features': ['u_100m', 'v_100m', 'temp_100m'],
            },
            'MeanAbsoluteError': {'weight': 0.8},
        }
    )
    loss, _ = multi_loss(x, y)

    assert np.allclose(0.2 * md_loss(x, y) + 0.8 * mae_loss(x, y), loss)


@pytest.mark.parametrize('shape', [(2, 4, 5), (2, 4, 5, 1), (2, 4, 5, 3)])
def test_geothermal_surface_heat_flux_loss(shape):
    """Check surface flux units, signs, and independence of lateral slopes."""
    loss_obj = GeothermalSurfaceHeatFluxLoss()
    coordinates = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    surface_temperature = 10 + coordinates / 2
    deep_temperature = surface_temperature + 20
    conductivity = 2 + coordinates / 100
    heat_flux = conductivity * 20
    predictions = np.stack(
        [surface_temperature, deep_temperature, conductivity, heat_flux],
        axis=-1,
    )
    assert np.isclose(
        loss_obj(np.zeros_like(predictions), predictions), 0, atol=1e-8
    )

    for offset in (-25.0, 25.0):
        perturbed = predictions.copy()
        perturbed[..., -1] += offset
        assert np.isclose(
            loss_obj(np.zeros_like(perturbed), perturbed), offset**2
        )


def test_geothermal_surface_heat_flux_features_and_gradients():
    """Check feature selection, depth spacing, and trainable residuals."""
    loss_obj = GeothermalSurfaceHeatFluxLoss(
        depths=[3000, 0, 500],
        temperature_prefix='temperature',
        heat_flux_prefix='flux',
        thermal_conductivity_prefix='conductivity',
    )
    assert loss_obj.gen_features == [
        'temperature_0m',
        'temperature_500m',
        'conductivity_0m',
        'flux_0m',
    ]
    assert loss_obj.true_features == loss_obj.gen_features
    predictions = tf.Variable([[[[10.0, 20.0, 2.0, 50.0]]]])
    with tf.GradientTape() as tape:
        loss = loss_obj(tf.zeros_like(predictions), predictions)
    gradients = tape.gradient(loss, predictions)
    assert np.isclose(loss, 100)
    assert gradients is not None
    assert np.all(np.isfinite(gradients))
    assert np.all(gradients.numpy() != 0)


@pytest.mark.parametrize(
    'depths',
    [[], [0], [1000, 2000], [-1000, 0], [0, 0], [0, np.nan], [0, np.inf]],
)
def test_geothermal_surface_heat_flux_invalid_depths(depths):
    """Reject depths that cannot define a finite surface derivative."""
    with pytest.raises(AssertionError):
        GeothermalSurfaceHeatFluxLoss(depths=depths)


@pytest.mark.parametrize('shape', [(2, 4, 4), (2, 4, 5, 1, 1, 4)])
def test_geothermal_surface_heat_flux_invalid_rank(shape):
    """Require spatial or spatiotemporal feature tensors."""
    predictions = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError, match='4D or 5D'):
        GeothermalSurfaceHeatFluxLoss()(predictions, predictions)


@pytest.mark.parametrize('n_times', [None, 1, 3])
def test_reshape_depth_feature_for_vertical_derivative(n_times):
    """Preserve each observation's spatial grid and ordered time slices."""
    shape = (2, 3, 4, 5) if n_times is None else (2, 3, 4, n_times, 5)
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = values
    if n_times is not None:
        expected = np.stack(
            [
                values[observation, :, :, time, :]
                for observation in range(shape[0])
                for time in range(n_times)
            ]
        )
    reshape = tf.function(_reshape_depth_feature_for_vertical_derivative)
    actual = reshape(tf.constant(values))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('conductivity_slope', [-0.001, 0.0, 0.001])
def test_geothermal_nonnegative_production_loss(axis, conductivity_slope):
    """Check source sign, units, and variable conductivity on each axis."""
    spacings = (1000, 2000, 500)
    depths = [0, 500, 1000, 1500, 2000]
    loss_obj = GeothermalNonnegativeHeatProductionLoss(
        dx=spacings[1], dy=spacings[0], depths=depths
    )
    coordinates = np.indices((5, 5, 5), dtype=np.float32)[axis]
    coordinates *= spacings[axis]
    temperature = 10 + 0.02 * coordinates
    conductivity = 20 + conductivity_slope * coordinates
    predictions = np.stack([temperature, conductivity], axis=-1)
    predictions = predictions.reshape(1, 5, 5, -1)
    expected_production = -1000 * 0.02 * conductivity_slope
    production = loss_obj._compute_heat_production(tf.constant(predictions))
    np.testing.assert_allclose(production, expected_production, atol=1e-7)
    assert np.isclose(
        loss_obj(np.zeros_like(predictions), predictions),
        max(-expected_production, 0) ** 2,
        atol=1e-10,
    )


@pytest.mark.parametrize('n_times', [None, 1, 3])
def test_geothermal_nonnegative_production_divergence(n_times):
    """Sum signed 3D terms before penalizing, independently at each time."""
    depths = [0, 500, 1000, 1500, 2000]
    loss_obj = GeothermalNonnegativeHeatProductionLoss(
        dx=2000, dy=1000, depths=depths
    )
    north, east, depth = np.indices((5, 5, 5), dtype=np.float32)
    north *= 1000
    east *= 2000
    depth *= 500
    temperature = 10 + 0.01 * (north + east + depth)
    conductivity = 20 - 0.001 * north + 0.002 * east - 0.002 * depth
    predictions = np.stack([temperature, conductivity], axis=-1)
    predictions = predictions.reshape(1, 5, 5, -1)
    predictions = np.repeat(predictions, 2, axis=0)
    predictions[1, ..., ::2] *= 2
    if n_times is not None:
        predictions = np.repeat(predictions[..., None, :], n_times, axis=3)
        predictions[..., ::2] *= np.arange(1, n_times + 1)[
            None, None, None, :, None
        ]

    production = loss_obj._compute_heat_production(tf.constant(predictions))
    expected = np.full((2, 5, 5, 5), 0.01, dtype=np.float32)
    expected[1] *= 2
    if n_times is not None:
        expected = np.repeat(expected[..., None, :], n_times, axis=3)
        expected *= np.arange(1, n_times + 1)[None, None, None, :, None]
    np.testing.assert_allclose(production, expected, atol=1e-7)
    assert np.isclose(loss_obj(np.zeros_like(predictions), predictions), 0)

    predictions[..., ::2] *= -1
    assert np.isclose(
        loss_obj(np.zeros_like(predictions), predictions),
        np.mean(expected**2),
    )


def test_geothermal_nonnegative_production_curvature_and_gradients():
    """Check temperature curvature, feature ordering, and backpropagation."""
    loss_obj = GeothermalNonnegativeHeatProductionLoss(
        dx=2000,
        dy=1000,
        depths=[3000, 1000, 2000, 1500, 2500],
        temperature_prefix='temperature',
        thermal_conductivity_prefix='conductivity',
    )
    depths = np.arange(1000, 3500, 500)
    assert loss_obj.gen_features == [
        feature
        for depth in depths
        for feature in (f'temperature_{depth}m', f'conductivity_{depth}m')
    ]
    assert loss_obj.true_features == loss_obj.gen_features
    temperature = np.broadcast_to(10 + 1e-5 * depths**2, (2, 5, 5, 5))
    conductivity = np.full_like(temperature, 2)
    predictions = np.stack([temperature, conductivity], axis=-1)
    predictions = tf.Variable(
        predictions.reshape(2, 5, 5, -1), dtype=tf.float32
    )
    expected = -0.04 * np.array([0.5, 0.75, 1.0, 0.75, 0.5])
    production = loss_obj._compute_heat_production(predictions)
    np.testing.assert_allclose(
        production, np.broadcast_to(expected, (2, 5, 5, 5)), atol=1e-7
    )
    with tf.GradientTape() as tape:
        loss = loss_obj(tf.zeros_like(predictions), predictions)
    gradients = tape.gradient(loss, predictions)
    assert np.isclose(loss, np.mean(expected**2))
    assert gradients is not None
    assert np.all(np.isfinite(gradients))
    assert np.any(gradients.numpy()[..., ::2] != 0)
    assert np.any(gradients.numpy()[..., 1::2] != 0)
    negative_curvature = predictions.numpy()
    negative_curvature[..., ::2] *= -1
    assert np.isclose(loss_obj(negative_curvature, negative_curvature), 0)


@pytest.mark.parametrize(
    'depths',
    [
        [],
        [0],
        [0, 1000],
        [-1000, 0, 1000],
        [0, 0, 0],
        [0, 1000, 3000],
        [0, 1000, np.nan],
        [0, 1000, np.inf],
    ],
)
def test_geothermal_nonnegative_production_invalid_depths(depths):
    """Reject depth grids that cannot resolve the source divergence."""
    with pytest.raises(AssertionError):
        GeothermalNonnegativeHeatProductionLoss(dx=1, dy=1, depths=depths)


@pytest.mark.parametrize('spacing', [0, -1, np.nan, np.inf])
@pytest.mark.parametrize('axis', ['dx', 'dy'])
def test_geothermal_nonnegative_production_invalid_spacing(axis, spacing):
    """Require finite, positive horizontal grid spacing."""
    kwargs = {'dx': 1, 'dy': 1, axis: spacing}
    with pytest.raises(AssertionError, match='finite and positive'):
        GeothermalNonnegativeHeatProductionLoss(**kwargs)


@pytest.mark.parametrize('shape', [(2, 4, 6), (2, 4, 5, 1, 1, 6)])
def test_geothermal_nonnegative_production_invalid_rank(shape):
    """Require 4D or 5D predictions."""
    predictions = np.zeros(shape, dtype=np.float32)
    loss_obj = GeothermalNonnegativeHeatProductionLoss(dx=1, dy=1)
    with pytest.raises(ValueError, match='4D or 5D'):
        loss_obj(predictions, predictions)


@pytest.mark.parametrize('shape', [(1, 2, 4, 6), (1, 4, 2, 1, 6)])
def test_geothermal_nonnegative_production_invalid_spatial_shape(shape):
    """Reject spatial grids too small for temperature curvature."""
    predictions = np.zeros(shape, dtype=np.float32)
    loss_obj = GeothermalNonnegativeHeatProductionLoss(
        dx=1, dy=1, depths=[0, 1, 2]
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match='three spatial'):
        loss_obj(predictions, predictions)


def test_geothermal_temp_grad_loss_depth_intersection_and_errors():
    """Test depth behavior and expected validation errors"""

    with pytest.raises(AssertionError):
        GeothermalPositiveTemperatureGradientLoss(depths=[0])

    loss_obj = GeothermalPositiveTemperatureGradientLoss(depths=[0, 1, 2])
    with pytest.raises(ValueError):
        loss_obj(np.zeros((2, 4, 6)), np.zeros((2, 4, 6)))


def test_geothermal_temp_grad_loss():
    """Test temp grad loss on synthetic data."""

    depths = [1000, 2000, 3000]
    loss_obj = GeothermalPositiveTemperatureGradientLoss(depths=depths)

    batch = 2
    s1 = s2 = 8

    tensors = []
    for depth in depths:
        temp = depth / 10 + np.zeros((batch, s1, s2), dtype=np.float32)
        tensors.append(temp)

    x_gen = np.stack(tensors, axis=-1)
    x_true = np.zeros_like(x_gen)

    loss_ref = loss_obj(x_true, x_gen).numpy()
    assert np.isclose(loss_ref, 0.0, atol=1e-7)

    x_gen_perturbed = x_gen.copy()
    x_gen_perturbed[..., 1] += 500
    loss_perturbed = loss_obj(x_true, x_gen_perturbed).numpy()

    assert loss_perturbed > loss_ref


def test_geothermal_moho_bc_loss():
    """Test Moho boundary-condition loss on synthetic data."""

    loss_obj = GeothermalMohoBCLoss(
        heat_flow_features=['q_0m'],
        moho_gradient_layer='moho_temp_gradient',
        upper_mantle_thermal_conductivity=4.0,
    )

    batch = 2
    s1 = s2 = 8

    heat_flow = 200 + np.zeros((batch, s1, s2, 1), dtype=np.float32)
    moho_gradient = 50 + np.zeros((batch, s1, s2, 1), dtype=np.float32)

    loss_ref = loss_obj(moho_gradient, heat_flow).numpy()
    assert loss_ref < 1e-10

    heat_flow_perturbed = heat_flow.copy()
    heat_flow_perturbed[..., 0] -= 5
    loss_perturbed = loss_obj(moho_gradient, heat_flow_perturbed).numpy()

    assert loss_perturbed > loss_ref
