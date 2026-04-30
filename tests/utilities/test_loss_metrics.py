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
    GeothermalConductiveHeatTransferLoss,
    GeothermalMohoBCLoss,
    GeothermalPositiveTemperatureGradientLoss,
    LowResLoss,
    MaterialDerivativeLoss,
    MmdLoss,
    SpatialExtremesLoss,
    SpatiotemporalFftLoss,
    TemporalExtremesLoss,
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


def test_geothermal_heat_transfer_loss_depth_intersection_and_errors():
    """Test geothermal heat transfer depth validation and input errors."""

    with pytest.raises(AssertionError):
        GeothermalConductiveHeatTransferLoss(dx=1, dy=1, depths=[0])

    with pytest.raises(AssertionError):
        GeothermalConductiveHeatTransferLoss(dx=1, dy=1, depths=[1000, 2000])

    with pytest.raises(AssertionError):
        GeothermalConductiveHeatTransferLoss(
            dx=1, dy=1, depths=[0, 1000, 3000]
        )

    loss_obj = GeothermalConductiveHeatTransferLoss(
        dx=1, dy=1, depths=[0, 1000, 2000]
    )
    with pytest.raises(ValueError):
        loss_obj(np.zeros((2, 4, 6)), np.zeros((2, 4, 6)))


def test_geothermal_heat_transfer_loss():
    """Test heat transfer loss on synthetic data."""

    depths = [0, 1, 2]
    dx = dy = 1.0
    loss_obj = GeothermalConductiveHeatTransferLoss(
        dx=dx, dy=dy, depths=depths
    )

    batch = 2
    s1 = 8
    s2 = 8
    x = np.arange(s1, dtype=np.float32)[np.newaxis, :, np.newaxis]
    y = np.arange(s2, dtype=np.float32)[np.newaxis, np.newaxis, :]

    k_const = 2.0
    t_slope_z = 0.01
    t_slope_x = 0.1
    t_slope_y = 0.2
    conductive_flux = k_const * (t_slope_x + t_slope_y + t_slope_z)
    q_const = conductive_flux * 1000

    tensors = []
    for depth in depths:
        temp = (
            t_slope_x * x
            + t_slope_y * y
            + t_slope_z * depth
            + np.zeros((batch, s1, s2), dtype=np.float32)
        )
        tensors.append(temp)

        k = k_const + np.zeros((batch, s1, s2), dtype=np.float32)
        tensors.append(k)

    q = q_const + np.zeros((batch, s1, s2), dtype=np.float32)
    tensors.append(q)

    x_gen = np.stack(tensors, axis=-1)
    x_true = np.zeros_like(x_gen)

    loss_ref = loss_obj(x_true, x_gen).numpy()
    assert np.isclose(loss_ref, 0.0, atol=1e-7)

    x_gen_perturbed = x_gen.copy()
    q_offset_idx = 2 * len(depths)
    x_gen_perturbed[..., q_offset_idx] += 1000.0
    loss_perturbed = loss_obj(x_true, x_gen_perturbed).numpy()

    assert loss_perturbed > loss_ref


def test_geothermal_heat_transfer_loss_errors():
    """Test depth behavior and expected validation errors"""

    dx = dy = 1.0
    with pytest.raises(AssertionError):
        GeothermalConductiveHeatTransferLoss(dx=dx, dy=dy, depths=[0])

    with pytest.raises(AssertionError):
        GeothermalConductiveHeatTransferLoss(dx=dx, dy=dy, depths=[1, 2])

    with pytest.raises(AssertionError):
        GeothermalConductiveHeatTransferLoss(dx=dx, dy=dy, depths=[1, 2, 5])

    loss_obj = GeothermalConductiveHeatTransferLoss(
        dx=dx, dy=dy, depths=[0, 1, 2, 3]
    )
    with pytest.raises(ValueError):
        loss_obj(np.zeros((2, 4, 6)), np.zeros((2, 4, 6)))


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
