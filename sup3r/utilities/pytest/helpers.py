"""Testing helpers."""

import os
from itertools import product

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.batch_handlers import BatchHandlerCC, BatchHandlerDC
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.samplers import DualSamplerCC, Sampler, SamplerDC
from sup3r.utilities.utilities import RANDOM_GENERATOR, pd_date_range
from sup3r.writers import OutputHandlerH5, OutputHandlerNC


def make_fake_tif(shape, out_file):
    """Make dummy data for tests."""

    y = np.linspace(70, -70, shape[0])
    x = np.linspace(-150, 150, shape[1])
    coords = {'band': [1], 'x': x, 'y': y}
    data_vars = {
        'band_data': (
            ('band', 'y', 'x'),
            RANDOM_GENERATOR.uniform(0, 1, (1, *shape)),
        )
    }
    nc = xr.Dataset(coords=coords, data_vars=data_vars)
    nc.to_netcdf(out_file, format='NETCDF4', engine='h5netcdf')


def make_fake_dset(
    shape, features, lat_range=None, lon_range=None, const=None
):
    """Make dummy data for tests."""

    if lat_range is None:
        lat_range = (70, -70)
    if lon_range is None:
        lon_range = (-150, 150)

    lats = np.linspace(*lat_range, shape[0])
    lons = np.linspace(*lon_range, shape[1])
    lons, lats = np.meshgrid(lons, lats)
    time = pd.date_range('2023-01-01', '2023-12-31', freq='60min')[: shape[2]]
    dims = (
        'time',
        Dimension.PRESSURE_LEVEL,
        Dimension.SOUTH_NORTH,
        Dimension.WEST_EAST,
    )
    coords = {}

    if len(shape) == 4:
        levels = np.linspace(1000, 0, shape[3])
        coords[Dimension.PRESSURE_LEVEL] = levels
    coords['time'] = time
    coords[Dimension.LATITUDE] = (
        (Dimension.SOUTH_NORTH, Dimension.WEST_EAST),
        lats,
    )
    coords[Dimension.LONGITUDE] = (
        (Dimension.SOUTH_NORTH, Dimension.WEST_EAST),
        lons,
    )

    dims = (
        'time',
        Dimension.PRESSURE_LEVEL,
        Dimension.SOUTH_NORTH,
        Dimension.WEST_EAST,
    )
    trans_axes = (2, 3, 0, 1)
    if len(shape) == 3:
        dims = ('time', *dims[2:])
        trans_axes = (2, 0, 1)
    data_vars = {}
    for f in features:
        if 'zg' in f:
            data = da.random.uniform(10, 1000, shape)
        elif 'orog' in f:
            data = da.random.uniform(0, 10, shape)
        elif 'pressure' in f:
            data = da.random.uniform(80000, 100000, shape)
        else:
            data = da.random.uniform(-1, 1, shape)
        data_vars[f] = (
            dims[: len(shape)],
            da.transpose(
                np.full(shape, const) if const is not None else data,
                axes=trans_axes,
            ),
        )
    nc = xr.Dataset(coords=coords, data_vars=data_vars)
    return nc.astype(np.float32)


def make_fake_nc_file(
    file_name, shape, features, lat_range=None, lon_range=None, const=None
):
    """Make nc file with dummy data for tests."""
    nc = make_fake_dset(shape, features, lat_range, lon_range, const)
    nc.to_netcdf(file_name, format='NETCDF4', engine='h5netcdf')


class DummyData(Container):
    """Dummy container with random data."""

    def __init__(self, data_shape, features):
        super().__init__()
        self.data = make_fake_dset(data_shape, features)


class DummySampler(Sampler):
    """Dummy container with random data."""

    def __init__(
        self,
        sample_shape,
        data_shape,
        features,
        batch_size,
        feature_sets=None,
        chunk_shape=None,
    ):
        data = make_fake_dset(data_shape, features=features)
        if chunk_shape is not None:
            data = data.chunk(chunk_shape)
        super().__init__(
            Sup3rDataset(high_res=data),
            sample_shape,
            batch_size=batch_size,
            feature_sets=feature_sets,
        )


def test_sampler_factory(SamplerClass):
    """Build test samplers which track indices."""

    class SamplerTester(SamplerClass):
        """Keep a record of sample indices for testing."""

        def __init__(self, *args, **kwargs):
            self.index_record = []
            super().__init__(*args, **kwargs)

        def get_sample_index(self, **kwargs):
            """Override get_sample_index to keep record of index accessible by
            batch handler. We store the index with the time entry divided by
            the batch size, since we have multiplied by the batch size to get
            a continuous time sample for multiple observations."""
            idx = super().get_sample_index(**kwargs)
            if len(idx) == 2:
                lr = list(idx[0])
                hr = list(idx[1])
                lr[2] = slice(
                    lr[2].start,
                    (lr[2].stop - lr[2].start) // self.batch_size
                    + lr[2].start,
                )
                hr[2] = slice(
                    hr[2].start,
                    (hr[2].stop - hr[2].start) // self.batch_size
                    + hr[2].start,
                )
                new_idx = (tuple(lr), tuple(hr))
            else:
                new_idx = list(idx)
                new_idx[2] = slice(
                    new_idx[2].start,
                    (new_idx[2].stop - new_idx[2].start) // self.batch_size
                    + new_idx[2].start,
                )
                new_idx = tuple(new_idx)
            self.index_record.append(new_idx)
            return idx

    return SamplerTester


DualSamplerTesterCC = test_sampler_factory(DualSamplerCC)
SamplerTesterDC = test_sampler_factory(SamplerDC)
SamplerTester = test_sampler_factory(Sampler)


class BatchHandlerTesterCC(BatchHandlerCC):
    """Batch handler with sampler with running index record."""

    SAMPLER = DualSamplerTesterCC


class BatchHandlerTesterDC(BatchHandlerDC):
    """Data-centric batch handler with record for testing"""

    SAMPLER = SamplerTesterDC

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temporal_weights_record = []
        self.spatial_weights_record = []
        self.s_index_record = []
        self.t_index_record = []
        self.space_bin_record = []
        self.time_bin_record = []
        self.space_bin_count = np.zeros(self.n_space_bins)
        self.time_bin_count = np.zeros(self.n_time_bins)
        self.max_rows = self.data[0].shape[0] - self.sample_shape[0] + 1
        self.max_cols = self.data[0].shape[1] - self.sample_shape[1] + 1
        self.max_tsteps = self.data[0].shape[2] - self.sample_shape[2] + 1
        self.spatial_bins = np.array_split(
            np.arange(0, self.max_rows * self.max_cols),
            self.n_space_bins,
        )
        self.spatial_bins = [b[-1] + 1 for b in self.spatial_bins]
        self.temporal_bins = np.array_split(
            np.arange(0, self.max_tsteps), self.n_time_bins
        )
        self.temporal_bins = [b[-1] + 1 for b in self.temporal_bins]

    def _update_bin_count(self, slices):
        s_idx = slices[0].start * self.max_cols + slices[1].start
        t_idx = slices[2].start
        self.s_index_record.append(s_idx)
        self.t_index_record.append(t_idx)
        self.space_bin_count[np.digitize(s_idx, self.spatial_bins)] += 1
        self.time_bin_count[np.digitize(t_idx, self.temporal_bins)] += 1

    def sample_batch(self):
        """Override get_samples to track sample indices."""
        out = super().sample_batch()
        if len(self.containers[0].index_record) > 0:
            self._update_bin_count(self.containers[0].index_record[-1])
        return out

    def __next__(self):
        out = super().__next__()
        if self._batch_count == self.n_batches:
            self.update_record()
        return out

    def update_record(self):
        """Reset records for a new epoch."""
        self.space_bin_record.append(self.space_bin_count)
        self.time_bin_record.append(self.time_bin_count)
        self.space_bin_count = np.zeros(self.n_space_bins)
        self.time_bin_count = np.zeros(self.n_time_bins)
        self.temporal_weights_record.append(self.temporal_weights)
        self.spatial_weights_record.append(self.spatial_weights)

    @staticmethod
    def _mean_record_normed(record):
        mean = np.array(record).mean(axis=0)
        return mean / mean.sum()


def BatchHandlerTesterFactory(BatchHandlerClass, SamplerClass):
    """Batch handler factory with sample counter and deterministic sampling for
    testing."""

    class BatchHandlerTester(BatchHandlerClass):
        """testing version of BatchHandler."""

        SAMPLER = SamplerClass

        def __init__(self, *args, **kwargs):
            self.sample_count = 0
            super().__init__(*args, **kwargs)

        def sample_batch(self):
            """Override get_samples to track sample count."""
            self.sample_count += 1
            return super().sample_batch()

    return BatchHandlerTester


def make_collect_chunks(td, ext='h5'):
    """Make fake chunked output files for collection tests.

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory
    ext : str
        File extension for output files. Either 'h5' or 'nc'. Default is 'h5'.

    Returns
    -------
    out_files : list
        List of filepaths to chunked files.
    data : ndarray
        (spatial_1, spatial_2, temporal, features)
        High resolution forward pass output
    ws_true : ndarray
        Windspeed between 0 and 20 in shape (spatial_1, spatial_2, temporal, 1)
    wd_true : ndarray
        Windir between 0 and 360 in shape (spatial_1, spatial_2, temporal, 1)
    features : list
        List of feature names corresponding to the last dimension of data
        ['windspeed_100m', 'winddirection_100m']
    hr_lat_lon : ndarray
        Array of lat/lon for hr data. (spatial_1, spatial_2, 2)
        Last dimension has ordering (lat, lon)
    hr_times : list
        List of np.datetime64 objects for hr data.
    """

    features = ['windspeed_100m', 'winddirection_100m']
    model_meta_data = {'foo': 'bar'}
    shape = (50, 50, 96, 1)
    ws_true = RANDOM_GENERATOR.uniform(0, 20, shape)
    wd_true = RANDOM_GENERATOR.uniform(0, 360, shape)
    data = np.concatenate((ws_true, wd_true), axis=3)
    lat = np.linspace(90, 0, 50)
    lon = np.linspace(-180, 0, 50)
    lon, lat = np.meshgrid(lon, lat)
    hr_lat_lon = np.dstack((lat, lon))

    gids = np.arange(np.prod(shape[:2]))
    gids = gids.reshape(shape[:2])

    hr_times = pd_date_range(
        '20220101', '20220103', freq='1800s', inclusive='left'
    )

    t_slices_hr = np.array_split(np.arange(len(hr_times)), 4)
    t_slices_hr = [slice(s[0], s[-1] + 1) for s in t_slices_hr]
    s_slices_hr = np.array_split(np.arange(shape[0]), 4)
    s_slices_hr = [slice(s[0], s[-1] + 1) for s in s_slices_hr]

    out_pattern = os.path.join(td, 'fp_out_{t}_{s}.' + ext)
    out_files = []

    Writer = OutputHandlerNC if ext == 'nc' else OutputHandlerH5
    for t, slice_hr in enumerate(t_slices_hr):
        for s, (s1_hr, s2_hr) in enumerate(product(s_slices_hr, s_slices_hr)):
            out_file = out_pattern.format(t=str(t).zfill(6), s=str(s).zfill(6))
            out_files.append(out_file)
            Writer._write_output(
                data[s1_hr, s2_hr, slice_hr, :],
                features,
                hr_lat_lon[s1_hr, s2_hr],
                hr_times[slice_hr],
                out_file,
                meta_data=model_meta_data,
                max_workers=1,
                row_inds=np.arange(shape[0])[s1_hr],
                col_inds=np.arange(shape[1])[s2_hr],
                gids=gids[s1_hr, s2_hr],
            )

    return (out_files, data, ws_true, wd_true, features, hr_lat_lon, hr_times)


def make_fake_h5_chunks(td):
    """Make fake h5 chunked output files for a 5x spatial 2x temporal
    multi-node forward pass output.

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory

    Returns
    -------
    out_files : list
        List of filepaths to chunked files.
    data : ndarray
        (spatial_1, spatial_2, temporal, features)
        High resolution forward pass output
    ws_true : ndarray
        Windspeed between 0 and 20 in shape (spatial_1, spatial_2, temporal, 1)
    wd_true : ndarray
        Windir between 0 and 360 in shape (spatial_1, spatial_2, temporal, 1)
    features : list
        List of feature names corresponding to the last dimension of data
        ['windspeed_100m', 'winddirection_100m']
    t_slices_lr : list
        List of low res temporal slices
    t_slices_hr : list
        List of high res temporal slices
    s_slices_lr : list
        List of low res spatial slices
    s_slices_hr : list
        List of high res spatial slices
    low_res_lat_lon : ndarray
        Array of lat/lon for input data. (spatial_1, spatial_2, 2)
        Last dimension has ordering (lat, lon)
    low_res_times : list
        List of np.datetime64 objects for coarse data.
    """

    features = ['windspeed_100m', 'winddirection_100m']
    model_meta_data = {'foo': 'bar'}
    shape = (50, 50, 96, 1)
    ws_true = RANDOM_GENERATOR.uniform(0, 20, shape)
    wd_true = RANDOM_GENERATOR.uniform(0, 360, shape)
    data = np.concatenate((ws_true, wd_true), axis=3)
    lat = np.linspace(90, 0, 10)
    lon = np.linspace(-180, 0, 10)
    lon, lat = np.meshgrid(lon, lat)
    low_res_lat_lon = np.dstack((lat, lon))

    gids = np.arange(np.prod(shape[:2]))
    gids = gids.reshape(shape[:2])

    low_res_times = pd_date_range(
        '20220101', '20220103', freq='3600s', inclusive='left'
    )

    t_slices_lr = [slice(0, 24), slice(24, None)]
    t_slices_hr = [slice(0, 48), slice(48, None)]

    s_slices_lr = [slice(0, 5), slice(5, 10)]
    s_slices_hr = [slice(0, 25), slice(25, 50)]

    out_pattern = os.path.join(td, 'fp_out_{t}_{i}{j}.h5')
    out_files = []
    for t, (slice_lr, slice_hr) in enumerate(zip(t_slices_lr, t_slices_hr)):
        for i, (s1_lr, s1_hr) in enumerate(zip(s_slices_lr, s_slices_hr)):
            for j, (s2_lr, s2_hr) in enumerate(zip(s_slices_lr, s_slices_hr)):
                out_file = out_pattern.format(
                    t=str(t).zfill(6),
                    i=str(i).zfill(3),
                    j=str(j).zfill(3),
                )
                out_files.append(out_file)
                OutputHandlerH5.write_output(
                    data[s1_hr, s2_hr, slice_hr, :],
                    features,
                    low_res_lat_lon[s1_lr, s2_lr],
                    low_res_times[slice_lr],
                    out_file,
                    meta_data=model_meta_data,
                    max_workers=1,
                    gids=gids[s1_hr, s2_hr],
                )

    return (
        out_files,
        data,
        ws_true,
        wd_true,
        features,
        t_slices_lr,
        t_slices_hr,
        s_slices_lr,
        s_slices_hr,
        low_res_lat_lon,
        low_res_times,
    )


def make_fake_cs_ratio_files(td, low_res_times, low_res_lat_lon, model_meta):
    """Make a set of dummy clearsky ratio files that match the GAN fwp outputs

    Parameters
    ----------
    td : tempfile.TemporaryDirectory
        Test TemporaryDirectory
    low_res_times :
        List of times for low res input data. If there is only a single low
        res timestep, it is assumed the data is daily.
    low_res_lat_lon
        Array of lat/lon for input data.
        (spatial_1, spatial_2, 2)
        Last dimension has ordering (lat, lon)
    model_meta : dict
        Meta data for model to write to file.

    Returns
    -------
    fps : list
        List of clearsky ratio .h5 chunked files.
    fp_pattern : str
        Glob pattern*string to find fps
    """
    fps = []
    chunk_dir = os.path.join(td, 'chunks/')
    fp_pattern = os.path.join(chunk_dir, 'sup3r_chunk_*.h5')
    os.makedirs(chunk_dir)

    for idt, timestamp in enumerate(low_res_times):
        fn = 'sup3r_chunk_{}_{}.h5'.format(str(idt).zfill(6), str(0).zfill(6))
        out_file = os.path.join(chunk_dir, fn)
        fps.append(out_file)

        cs_ratio = RANDOM_GENERATOR.uniform(0, 1, (20, 20, 1, 1))
        cs_ratio = np.repeat(cs_ratio, 24, axis=2)

        OutputHandlerH5.write_output(
            cs_ratio,
            ['clearsky_ratio'],
            low_res_lat_lon,
            pd.DatetimeIndex([timestamp]),
            out_file,
            max_workers=1,
            meta_data=model_meta,
        )
    return fps, fp_pattern
