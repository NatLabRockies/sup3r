"""Validating PresRat correction procedures


Relevant resources used in the tests:
- fp_resource: Synthetic dataset representing observed historical (oh).
- fp_cc: Filename of standard biased dataset.
- fut_cc: Future dataset sample based on fp_cc + an offset + small noise
- fp_fut_cc: Filname to `fut_cc`.
- fut_cc_notrend: Future dataset identical to fp_cc, i.e. no trend.
- fp_fut_cc_notrend: Filename to fut_cc_notrend.
- presrat_params: Parameters of reference to test PresRat (using fp_fut_cc).
- presrat_notrend_params: Quantiles of future (mf) are identical to bias
  reference (mh). Thus, there is no trend in the model.
- presrat_identity_params: All distributions (oh & mf) are identical to mh,
    i.e. observations equal to model that doesn't change on time.
- presrat_nozeros_params: Same of presrat_params, but no zero_rate, i.e.
    there is a bias correction but all zero_rate values are equal to 0
    (percent), i.e. no value is modified to zero.
- presrat_nochanges_params: Like presrat_identity_params, but also all
    zero_rate are zero percent, i.e. no values should be forced to be zero.
"""

import math
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from rex import Outputs

from sup3r import CONFIG_DIR
from sup3r.bias import (
    PresRat,
    local_presrat_bc,
    local_qdm_bc,
)
from sup3r.bias.mixins import ZeroRateMixin
from sup3r.models import Sup3rGan
from sup3r.pipeline.forward_pass import ForwardPass, ForwardPassStrategy
from sup3r.preprocessing import DataHandler
from sup3r.preprocessing.utilities import get_date_range_kwargs
from sup3r.utilities.utilities import (
    RANDOM_GENERATOR,
    Timer,
    xr_open_mfdataset,
)

CC_LAT_LON = DataHandler(pytest.FP_RSDS, features='rsds').lat_lon
# A reference zero rate threshold that might not make sense physically but for
# testing purposes only. This might change in the future to force edge cases.
ZR_THRESHOLD = 0.01
TAU = 0.01
TARGET = (38.24552917480469, -105.46875)
SHAPE = (4, 4)
VAR_MIN = 0
# Fix this max
VAR_MAX = 1300

# Time duration in days of all sample dataset
# More than a year to check year transition situations
SAMPLE_TIME_DURATION = 2 * 365 + 1
# Temporal resolution in days of sample dataset
SAMPLE_TIME_RESOLUTION = 1
SAMPLE_ZERO_RATE = 0.01


@pytest.fixture(scope='module')
def fp_resource(tmpdir_factory):
    """Synthetic data, observed historical dataset

    Note
    ----
    Latitude MUST be descending and longitude ascending, otherwise
    `bias_transforms._get_factors()` does the wrong thing.
    """
    fn = tmpdir_factory.mktemp('data').join('precip_oh.h5')

    time = pd.date_range(
        '2018-01-01 00:00:00', '2019-01-01 00:00:00', freq='6h'
    )
    lat = np.arange(39.77, 39.00, -0.04)
    lon = np.arange(-105.14, -104.37, 0.04)
    ghi = RANDOM_GENERATOR.lognormal(0.0, 1.0, (time.size, lat.size, lon.size))

    ds = xr.Dataset(
        data_vars={'ghi': (['time', 'lat', 'lon'], ghi)},
        coords={
            'time': ('time', time),
            'lat': ('lat', lat),
            'lon': ('lon', lon),
        },
    )

    ds = ds.sortby('lat', ascending=False)
    lat_2d, lon_2d = xr.broadcast(ds['lat'], ds['lon'])
    meta = pd.DataFrame({
        'latitude': lat_2d.values.flatten(),
        'longitude': lon_2d.values.flatten(),
    })

    shapes = {'ghi': (len(ds.ghi.time), np.prod(ds.ghi.isel(time=0).shape))}
    attrs = {'ghi': None}
    chunks = {'ghi': None}
    dtypes = {'ghi': 'float32'}

    Outputs.init_h5(
        fn,
        ['ghi'],
        shapes,
        attrs,
        chunks,
        dtypes,
        meta=meta,
        time_index=pd.DatetimeIndex(ds.time),
    )
    with Outputs(fn, 'a') as out:
        out['ghi'] = ds.stack(flat=('lat', 'lon'))['ghi']

    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope='module')
def precip():
    """Synthetic historical modeled dataset"""
    lat = np.array([
        40.3507847105177,
        39.649032596592,
        38.9472804370071,
        38.2455282337738,
    ])
    lon = np.array([254.53125, 255.234375, 255.9375, 256.640625])

    time = pd.date_range(
        '2015-01-01T12:00:00', '2016-12-31T12:00:00', freq='D'
    )
    pr = RANDOM_GENERATOR.lognormal(0.0, 1.0, (time.size, lat.size, lon.size))

    # Transform the upper tail into negligible to guarantee some 'zero
    # precipiation days'.
    thr = np.sort(pr.flatten())[-math.ceil(0.001 * pr.size)]
    pr = np.where(pr < thr, pr, SAMPLE_ZERO_RATE / 2)

    # In case of playing with offset or other adjustments
    assert pr.min() >= 0

    ds = xr.DataArray(
        name='rsds',
        data=pr,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': ('time', pd.DatetimeIndex(time)),
            'lat': ('lat', lat),
            'lon': ('lon', lon),
        },
    )

    return ds


@pytest.fixture(scope='module')
def fp_cc(tmpdir_factory, precip):
    """Precipitation sample filename

    DataHandlerNCforCC requires a file to be opened
    """
    fn = tmpdir_factory.mktemp('data').join('precip_mh.nc')
    precip.to_netcdf(fn, format='NETCDF4', engine='h5netcdf')
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


# fut_cc
@pytest.fixture(scope='module')
def precip_fut(precip):
    """Synthetic data, modeled future (mf) dataset"""
    da = precip.copy(deep=True)

    time = da['time'] + np.timedelta64(18263, 'D')
    time.attrs = da['time'].attrs
    da['time'] = time
    # Adding an offset of 3 IQ
    offset = 3 * float(da.quantile(0.75) - da.quantile(0.25))
    da += offset
    # adding a small noise
    da += 1e-6 * RANDOM_GENERATOR.random(da.shape)

    return da


@pytest.fixture(scope='module')
def fp_fut_cc(tmpdir_factory, precip_fut):
    """Sample future CC dataset (precipitation) filename"""
    fn = tmpdir_factory.mktemp('data').join('precip_mf.nc')
    precip_fut.to_netcdf(fn, format='NETCDF4', engine='h5netcdf')
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope='module')
def fut_cc(fp_fut_cc):
    """Gives the dataset itself related to fp_fut_cc

    Giving an object in memory makes everything more efficient by avoiding I/O
    reading files and overhead for multiple uses.

    Note that ``Resources`` modifies the dataset so we cannot just load the
    NetCDF. Here we run a couple of checks to confirm that the output
    dataset is as expected by sup3r.

    To use time as expected by sup3r: time = pd.to_datetime(da.time)
    To use latlon as expected by sup3r:
      latlon = np.stack(xr.broadcast(da["lat"], da["lon"] - 360),
                        axis=-1).astype('float32')
    """
    ds = xr_open_mfdataset(fp_fut_cc)

    # Operating with numpy arrays impose a fixed dimensions order
    # This compute is required here.
    da = ds['rsds'].compute().transpose('lat', 'lon', 'time')

    # The _get_factors() assume latitude as descending and it will
    # silently return wrong values otherwise.
    da = da.sortby('lat', ascending=False)

    latlon = np.stack(xr.broadcast(da['lat'], da['lon'] - 360), axis=-1)
    # Confirm that dataset order is consistent
    # Somewhere in pipeline latlon are downgraded to f32
    assert np.allclose(latlon.astype('float32'), CC_LAT_LON)

    # Verify data alignment in comparison with expected for FP_RSDS
    for ii in range(ds.lat.size):
        for jj in range(ds.lon.size):
            assert np.allclose(
                da.sel(lat=latlon[ii, jj, 0]).sel(lon=latlon[ii, jj, 1] + 360),
                da.data[ii, jj],
            )

    return da


@pytest.fixture(scope='module')
def fp_fut_cc_notrend(tmpdir_factory, fp_cc):
    """Sample future CC (mf) dataset identical to historical CC (mh)

    This is currently a copy of fp_cc, thus no trend on time.
    """
    fn = tmpdir_factory.mktemp('data').join('test_mf_notrend.nc')
    shutil.copyfile(fp_cc, fn)
    # DataHandlerNCforCC requires a string
    fn = str(fn)
    return fn


@pytest.fixture(scope='module')
def fut_cc_notrend(fp_fut_cc_notrend):
    """Extract the dataset from fp_fut_cc_notrend

    The internal process to read such dataset is way more complex than just
    reading it and there are some transformations. This function must provide
    a dataset compatible with the one expected from the standard processing.
    """
    ds = xr_open_mfdataset(fp_fut_cc_notrend)

    # Although it is the same file, somewhere in the data reading process
    # the longitude is transformed to the standard [-180 to 180] and it is
    # expected to be like that everywhere.
    ds['lon'] = ds['lon'] - 360

    # Operating with numpy arrays impose a fixed dimensions order
    # This compute is required here.
    da = ds['rsds'].compute().transpose('lat', 'lon', 'time')

    # The _get_factors() assume latitude as descending and it will
    # silently return wrong values otherwise.
    da = da.sortby('lat', ascending=False)

    latlon = np.stack(xr.broadcast(da['lat'], da['lon']), axis=-1)
    # Confirm that dataset order is consistent
    # Somewhere in pipeline latlon are downgraded to f32
    assert np.allclose(latlon.astype('float32'), CC_LAT_LON)

    # Verify data alignment in comparison with expected for FP_RSDS
    for ii in range(ds.lat.size):
        for jj in range(ds.lon.size):
            np.allclose(
                da.sel(lat=latlon[ii, jj, 0]).sel(lon=latlon[ii, jj, 1]),
                da.data[ii, jj],
            )

    return da


@pytest.fixture(scope='module')
def presrat_params(tmpdir_factory, fp_resource, fp_cc, fp_fut_cc):
    """PresRat parameters for standard datasets

    Use the standard datasets to estimate the distributions and save
    in a temporary place to be re-used
    """
    calc = PresRat(
        fp_resource,
        fp_cc,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        distance_upper_bound=0.7,
        bias_handler='DataHandlerNCforCC',
    )
    fn = tmpdir_factory.mktemp('params').join('presrat.h5')
    # Physically non-sense threshold choosed to result in gridpoints with and
    # without zero rate correction for the given testing dataset.
    _ = calc.run(zero_rate_threshold=ZR_THRESHOLD, fp_out=fn, max_workers=1)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


@pytest.fixture(scope='module')
def presrat_notrend_params(
    tmpdir_factory, fp_resource, fp_cc, fp_fut_cc_notrend
):
    """No change in time

    The bias_fut distribution is equal to bias (modeled historical), so no
    change in time.

    We could save some overhead by copying fp_fut_cc and replacing some
    values there. That was done before but missed some variables resulting
    in errors.
    """
    calc = PresRat(
        fp_resource,
        fp_cc,
        fp_fut_cc_notrend,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        distance_upper_bound=0.7,
        bias_handler='DataHandlerNCforCC',
    )
    fn = tmpdir_factory.mktemp('params').join('presrat_notrend.h5')
    _ = calc.run(zero_rate_threshold=ZR_THRESHOLD, fp_out=fn)

    # DataHandlerNCforCC requires a string
    fn = str(fn)

    return fn


@pytest.fixture(scope='module')
def presrat_identity_params(tmpdir_factory, presrat_params):
    """Identical distribution"""

    fn = tmpdir_factory.mktemp('params').join('presrat_identity.h5')
    shutil.copyfile(presrat_params, fn)

    with h5py.File(fn, 'r+') as f:
        f['bias_rsds_params'][:] = f['bias_fut_rsds_params'][:]
        f['base_ghi_params'][:] = f['bias_fut_rsds_params'][:]
        f.flush()

    return str(fn)


@pytest.fixture(scope='module')
def presrat_nochanges_params(tmpdir_factory, presrat_params):
    """Identical distribution, no zero rate, and K=1

    All distributions are identical, zero rate changes are all zero, and the
    K factor is equal to 1, therefore, the PresRat correction should not change
    anything.

    Note that distributions are based on bias_fut, so it is assumed that the
    test cases will be datasets coherent with that bias_fut distribution,
    otherwise it could lead to differences if out of that scale.
    """
    fn = tmpdir_factory.mktemp('params').join('presrat_nochanges.h5')
    shutil.copyfile(presrat_params, fn)

    with h5py.File(fn, 'r+') as f:
        f['bias_fut_rsds_params'][:] = f['bias_rsds_params'][:]
        f['base_ghi_params'][:] = f['bias_rsds_params'][:]
        f['ghi_zero_rate'][:] *= 0
        f['rsds_tau_fut'][:] *= 0
        f['rsds_k_factor'][:] = 1
        f.attrs['zero_rate_threshold'] = 0
        f.flush()

    return str(fn)


@pytest.fixture(scope='module')
def presrat_nozeros_params(tmpdir_factory, presrat_params):
    """PresRat parameters without zero rate correction

    The same standard parameters but all zero_rate values are equal to zero,
    which means that zero percent, i.e. none, of the values should be forced
    to be zero.
    """
    fn = tmpdir_factory.mktemp('params').join('presrat_nozeros.h5')
    shutil.copyfile(presrat_params, fn)

    with h5py.File(fn, 'r+') as f:
        f['ghi_zero_rate'][:] *= 0
        f['rsds_tau_fut'][:] *= 0
        f.flush()

    return str(fn)


# ==== Zero rate estimate ====


def test_zero_precipitation_rate():
    """Zero rate estimate using median"""
    f = ZeroRateMixin().zero_precipitation_rate
    arr = RANDOM_GENERATOR.random(100)

    rate = f(arr, threshold=np.median(arr))
    assert rate == 0.5


def test_zero_precipitation_rate_extremes():
    """Zero rate estimate with extremme thresholds"""
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.arange(10)

    rate = f(arr, threshold=-1)
    assert rate == 0

    rate = f(arr, threshold=0)
    assert rate == 0.1

    # Remember, 9 is the last value, i.e. the 10th value
    rate = f(arr, threshold=9)
    assert rate == 1

    rate = f(arr, threshold=arr.max() + 1)
    assert rate == 1


def test_zero_precipitation_rate_nanonly():
    """Zero rate estimate with only NaNs gives NaN"""
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.nan * np.zeros(10)

    # All NaN gives NaN rate
    rate = f(arr)
    assert np.isnan(rate)


def test_zero_precipitation_rate_nan():
    """Zero rate estimate with NaNs

    NaN shouldn't be counted to find the rate. Thus an array with NaNs should
    give the same results if the NaN were removed before the calculation.
    """
    f = ZeroRateMixin().zero_precipitation_rate
    arr = np.arange(10)

    r1 = f(arr, threshold=5)
    r2 = f(np.concatenate([5 * [np.nan], arr]), threshold=5)
    assert r1 == r2


# ==== PresRat parameters estimate ====


def test_presrat_calc(fp_resource, fp_cc, fp_fut_cc):
    """Standard PresRat (pre) calculation

    Estimate the required parameters with a standard setup.
    """
    calc = PresRat(
        fp_resource,
        fp_cc,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out = calc.run(max_workers=2)

    expected_vars = [
        'bias_rsds_params',
        'bias_fut_rsds_params',
        'base_ghi_params',
        'ghi_zero_rate',
        'rsds_k_factor',
        'rsds_tau_fut',
    ]
    for v in expected_vars:
        assert v in out, f'Missing {v} in the calculated output'
        assert (
            out[v].shape[:2] == SHAPE
        ), "Doesn't match expected spatial shape"
        # This is only true because fill and extend are applied by default.
        assert np.all(np.isfinite(out[v])), f'Invalid value for {v}'

    for k, v in ((k, v) for k, v in out.items() if k.endswith('_zero_rate')):
        assert np.all((v >= 0) & (v <= 1)), f'Out of range [0, 1]: {k}'

    for k, v in ((k, v) for k, v in out.items() if k.endswith('_k_factor')):
        assert np.all(v > 0), f'K factor must be positive: {k}'


@pytest.mark.parametrize('threshold', [0, 1, 1e6])
def test_parallel(fp_resource, fp_cc, fp_fut_cc, threshold):
    """Running in parallel must not alter results

    Check with different thresholds, which will result in different zero rates.
    """
    s = PresRat(
        fp_resource,
        fp_cc,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out_s = s.run(max_workers=1, zero_rate_threshold=threshold)

    p = PresRat(
        fp_resource,
        fp_cc,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out_p = p.run(max_workers=2, zero_rate_threshold=threshold)

    for k in out_s:
        assert k in out_p, f'Missing {k} in parallel run'
        assert np.allclose(
            out_s[k], out_p[k], equal_nan=True
        ), f'Different results for {k}'


@pytest.mark.parametrize('threshold', [0, 1, 1e6])
def test_presrat_zero_rate(fp_resource, fp_cc, fp_fut_cc, threshold):
    """Estimate zero_rate within PresRat.run()

    Use thresholds that gives 0%, 100%, and something between.

    Notes
    -----
    - Rate should be zero if threshold is zero for this dataset.
    """
    calc = PresRat(
        fp_resource,
        fp_cc,
        fp_fut_cc,
        'ghi',
        'rsds',
        target=TARGET,
        shape=SHAPE,
        bias_handler='DataHandlerNCforCC',
    )

    out = calc.run(zero_rate_threshold=threshold)

    assert 'ghi_zero_rate' in out, 'Missing ghi_zero_rate in calc output'
    for k, v in ((k, v) for k, v in out.items() if k.endswith('_zero_rate')):
        # This is only true because fill and extend are applied by default.
        assert np.all(np.isfinite(v)), f'Invalid value for {v}'

        assert np.all((v >= 0) & (v <= 1)), f'Out of range [0, 1]: {k}'

        if threshold <= 0:
            assert np.all(v == 0), 'It should be rate 0 for threshold==0'
        elif threshold >= 1e4:
            assert np.all(v == 1), 'It should be rate 1 for threshold>=1e4'


# ==== PresRat Transform ====


def test_presrat_transform(presrat_params, precip_fut):
    """A standard run with local_presrat_bc

    Confirms that:
    - unbiased values is different than input biases data
    - unbiased zero rate is not smaller the input zero rate
    """
    # local_presrat_bc expects time in the last dimension.

    data = precip_fut.transpose('lat', 'lon', 'time')
    time = pd.to_datetime(precip_fut.time)
    latlon = np.stack(
        xr.broadcast(precip_fut['lat'], precip_fut['lon'] - 360), axis=-1
    ).astype('float32')

    unbiased = local_presrat_bc(
        data,
        latlon,
        'ghi',
        'rsds',
        bias_fp=presrat_params,
        date_range_kwargs=get_date_range_kwargs(time),
    )

    assert np.isfinite(unbiased).any(), "Can't compare if only NaN"
    # Confirm that there were changes, but at this point stop there.
    assert not np.allclose(data, unbiased, equal_nan=False)

    n_zero = (data == 0).astype('i').sum()
    unbiased_n_zero = (unbiased == 0).astype('i').sum()
    assert n_zero <= unbiased_n_zero


def test_presrat_transform_nochanges(presrat_nochanges_params, fut_cc_notrend):
    """The correction should result in no changes at all

    Note that there are a lot of implicit transformations, so we cannot expect
    to be able to esily compare all gridpoints.

    The unbiased output must be the same if:
    - The three CDFs are the same, so no change due to QDM. There is one
      caveat here. The data to be corrected is compared with the mf's CDF, and
      if out of distribution, it would lead to differences;
    - All zero rate set to zero percent, so no value is forced to zero;
    - The value to be corrected is the same used to estimate the means for the
      K factor;
    """
    data = fut_cc_notrend.values
    time = pd.to_datetime(fut_cc_notrend.time)
    latlon = np.stack(
        xr.broadcast(fut_cc_notrend['lat'], fut_cc_notrend['lon']),
        axis=-1,
    ).astype('float32')

    unbiased = local_presrat_bc(
        data,
        latlon,
        'ghi',
        'rsds',
        presrat_nochanges_params,
        get_date_range_kwargs(time),
    )

    assert np.isfinite(unbiased).any(), "Can't compare if only NaN"

    assert np.allclose(
        data, unbiased, equal_nan=False
    ), "This case shouldn't modify the data"


def test_presrat_transform_nozerochanges(presrat_nozeros_params, fut_cc):
    """No adjustment to zero

    Correction procedure results in some changes, but since the zero_rate
    correction is all set to zero percent, there are no values adjusted to
    zero.
    """
    data = fut_cc.values
    time = pd.to_datetime(fut_cc.time)
    latlon = np.stack(
        xr.broadcast(fut_cc['lat'], fut_cc['lon'] - 360), axis=-1
    ).astype('float32')

    corrected = local_presrat_bc(
        data,
        latlon,
        'ghi',
        'rsds',
        presrat_nozeros_params,
        get_date_range_kwargs(time),
    )

    assert np.isfinite(data).any(), "Can't compare if only NaN"
    assert not np.allclose(
        data, corrected, equal_nan=False
    ), 'Expected changes due to bias correction'
    assert not (
        (data != 0) & (corrected == 0)
    ).any(), 'Unexpected value corrected (zero_rate) to zero (dry day)'


def test_compare_qdm_vs_presrat(presrat_params, precip_fut):
    """Compare bias correction methods QDM vs PresRat"""

    # local_presrat_bc and local_qdm_bc expects time in the last dimension.
    data = precip_fut.transpose('lat', 'lon', 'time').values
    time = pd.to_datetime(precip_fut.time)
    latlon = np.stack(
        xr.broadcast(precip_fut['lat'], precip_fut['lon'] - 360), axis=-1
    ).astype('float32')

    unbiased_qdm = local_qdm_bc(
        data,
        latlon,
        'ghi',
        'rsds',
        presrat_params,
        get_date_range_kwargs(time),
    )
    unbiased_presrat = local_presrat_bc(
        data,
        latlon,
        'ghi',
        'rsds',
        presrat_params,
        get_date_range_kwargs(time),
    )

    assert (
        unbiased_qdm.shape == unbiased_presrat.shape
    ), 'QDM and PresRat output should have the same shape'

    n_zero_qdm = (unbiased_qdm < TAU).astype('i').sum()
    n_zero_presrat = (unbiased_presrat < TAU).astype('i').sum()
    assert (
        n_zero_qdm <= n_zero_presrat
    ), 'PresRat should guarantee greater or equal zero precipitation days'


def test_fwp_integration(tmp_path, presrat_params, fp_fut_cc):
    """Integration of the bias correction method into the forward pass

    Validate two aspects:
        (1) We should be able to run a forward pass with unbiased data.
        (2) The bias trend should be observed in the predicted output.

    TODO: This still needs to do (2)
    """
    fp_gen = os.path.join(CONFIG_DIR, 'spatiotemporal/gen_3x_4x_2f.json')
    fp_disc = os.path.join(CONFIG_DIR, 'spatiotemporal/disc.json')
    features = ['rsds']
    target = TARGET
    shape = SHAPE
    temporal_slice = slice(None, None, 1)
    fwp_chunk_shape = (4, 4, 150)
    input_files = [fp_fut_cc]

    Sup3rGan.seed()
    model = Sup3rGan(fp_gen, fp_disc, learning_rate=1e-4)
    _ = model.generate(np.ones((4, 10, 10, 6, len(features))))
    model.set_model_params(
        lr_features=features,
        hr_out_features=features,
        s_enhance=3,
        t_enhance=4,
        input_resolution={'spatial': '12km', 'temporal': '60min'},
    )

    out_dir = os.path.join(tmp_path, 'st_gan')
    model.save(out_dir)

    bias_correct_kwargs = {
        'rsds': {
            'feature_name': 'rsds',
            'base_dset': 'ghi',
            'bias_fp': presrat_params,
        }
    }

    strat = ForwardPassStrategy(
        input_files,
        model_kwargs={'model_dir': out_dir},
        fwp_chunk_shape=fwp_chunk_shape,
        spatial_pad=0,
        temporal_pad=0,
        input_handler_kwargs={
            'target': target,
            'shape': shape,
            'time_slice': temporal_slice,
        },
        out_pattern=os.path.join(tmp_path, 'out_{file_id}.nc'),
        input_handler_name='DataHandlerNCforCC',
    )
    bc_strat = ForwardPassStrategy(
        input_files,
        model_kwargs={'model_dir': out_dir},
        fwp_chunk_shape=fwp_chunk_shape,
        spatial_pad=0,
        temporal_pad=0,
        input_handler_kwargs={
            'target': target,
            'shape': shape,
            'time_slice': temporal_slice,
        },
        out_pattern=os.path.join(tmp_path, 'out_{file_id}.nc'),
        input_handler_name='DataHandlerNCforCC',
        bias_correct_method='local_presrat_bc',
        bias_correct_kwargs=bias_correct_kwargs,
    )

    timer = Timer()
    fwp = timer(ForwardPass, log=True)(strat)
    bc_fwp = timer(ForwardPass, log=True)(bc_strat)

    for ichunk in range(len(strat.node_chunks)):
        bc_chunk = bc_fwp.get_input_chunk(ichunk)
        chunk = fwp.get_input_chunk(ichunk)

        _delta = bc_chunk.input_data - chunk.input_data
        kwargs = {
            'model_kwargs': strat.model_kwargs,
            'model_class': strat.model_class,
            'allowed_const': strat.allowed_const,
            'output_workers': strat.output_workers,
        }
        _, data = fwp.run_chunk(chunk, meta=fwp.meta, **kwargs)
        _, bc_data = bc_fwp.run_chunk(bc_chunk, meta=bc_fwp.meta, **kwargs)

        _delta = bc_data - data
