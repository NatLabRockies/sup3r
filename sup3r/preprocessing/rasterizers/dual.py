"""Paired rasterizer class for matching separate low_res and high_res
datasets"""

import logging
from typing import Union
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from rex.utilities.regridder import Regridder

from sup3r.preprocessing.base import Container, Sup3rDataset
from sup3r.preprocessing.names import Dimension
from sup3r.preprocessing.utilities import log_args
from sup3r.utilities.utilities import spatial_coarsening
from sup3r.writers import Cacher

logger = logging.getLogger(__name__)


class DualRasterizer(Container):
    """Object containing xr.Dataset instances for low and high-res data.
    (Usually ERA5 and WTK, respectively). This essentially just regrids the
    low-res data to the coarsened high-res grid.  This is useful for caching
    prepping data which then can go directly to a
    :class:`~sup3r.preprocessing.samplers.dual.DualSampler`
    :class:`~sup3r.preprocessing.batch_queues.dual.DualBatchQueue`.

    Note
    ----
    When first extracting the low_res data make sure to extract a region that
    completely overlaps the high_res region. It is easiest to load the full
    low_res domain and let :class:`.DualRasterizer` select the appropriate
    region through regridding.
    """

    @log_args
    def __init__(
        self,
        data: Union[
            Sup3rDataset, tuple[xr.Dataset, xr.Dataset], dict[str, xr.Dataset]
        ],
        regrid_workers=1,
        regrid_lr=True,
        run_qa=True,
        s_enhance=1,
        t_enhance=1,
        lr_cache_kwargs=None,
        hr_cache_kwargs=None,
    ):
        """Initialize data container lr and hr :class:`Data` instances.
        Typically lr = ERA5 data and hr = WTK data.

        Parameters
        ----------
        data : Sup3rDataset | tuple[xr.Dataset, xr.Dataset] |
               dict[str, xr.Dataset]
            A tuple of xr.Dataset instances. The first must be low-res
            and the second must be high-res data
        regrid_workers : int | None
            Number of workers to use for regridding routine.
        regrid_lr : bool
            Flag to regrid the low-res data to the high-res grid. This will
            take care of any minor inconsistencies in different projections.
            Disable this if the grids are known to be the same.
        run_qa : bool
            Flag to run qa on the regridded low-res data. This will check for
            NaNs and fill them if there are not too many.
        s_enhance : int
            Spatial enhancement factor
        t_enhance : int
            Temporal enhancement factor
        lr_cache_kwargs : dict
            Cache kwargs for the call to lr_data.cache_data(cache_kwargs).
            Must include 'cache_pattern' key if not None, and can also include
            dictionary of chunk tuples with feature keys
        hr_cache_kwargs : dict
            Cache kwargs for the call to hr_data.cache_data(cache_kwargs).
            Must include 'cache_pattern' key if not None, and can also include
            dictionary of chunk tuples with feature keys
        """
        self.s_enhance = s_enhance
        self.t_enhance = t_enhance
        if isinstance(data, tuple):
            data = {'low_res': data[0], 'high_res': data[1]}
        if isinstance(data, dict):
            data = Sup3rDataset(**data)
        msg = (
            'The DualRasterizer requires a data tuple or dictionary with two '
            'members, low and high resolution in that order, or a '
            f'Sup3rDataset instance. Received {type(data)}.'
        )
        assert isinstance(data, Sup3rDataset), msg
        self.data = data
        self.regrid_workers = regrid_workers

        lr_step = self.data.low_res.time_step
        hr_step = self.data.high_res.time_step
        msg = (
            f'Time steps of high-res data ({hr_step} seconds) and low-res '
            f'data ({lr_step} seconds) are inconsistent with t_enhance = '
            f'{self.t_enhance}.'
        )
        assert np.allclose(lr_step, hr_step * self.t_enhance), msg

        self.lr_required_shape = (
            self.data.high_res.shape[0] // self.s_enhance,
            self.data.high_res.shape[1] // self.s_enhance,
            self.data.high_res.shape[2] // self.t_enhance,
        )
        self.hr_required_shape = (
            self.s_enhance * self.lr_required_shape[0],
            self.s_enhance * self.lr_required_shape[1],
            self.t_enhance * self.lr_required_shape[2],
        )

        msg = (
            f'The required low-res shape {self.lr_required_shape} is '
            'inconsistent with the shape of the raw data '
            f'{self.data.low_res.shape}'
        )
        assert all(
            req_s <= true_s
            for req_s, true_s in zip(
                self.lr_required_shape, self.data.low_res.shape
            )
        ), msg

        self.hr_lat_lon = self.data.high_res.lat_lon[
            slice(self.hr_required_shape[0]), slice(self.hr_required_shape[1])
        ]
        self.lr_lat_lon = spatial_coarsening(
            self.hr_lat_lon, s_enhance=self.s_enhance, obs_axis=False
        )
        self._regrid_lr = regrid_lr

        self.update_lr_data()
        self.update_hr_data()
        super().__init__(data=(self.data.low_res, self.data.high_res))

        if run_qa:
            self.check_regridded_lr_data()

        if lr_cache_kwargs is not None:
            Cacher(self.data.low_res, lr_cache_kwargs)

        if hr_cache_kwargs is not None:
            Cacher(self.data.high_res, hr_cache_kwargs)

    def update_hr_data(self):
        """Set the high resolution data attribute and check if
        hr_data.shape is divisible by s_enhance. If not, take the largest
        shape that can be."""
        msg = (
            f'hr_data.shape: {self.data.high_res.shape[:3]} is not '
            f'divisible by s_enhance: {self.s_enhance}. Using shape: '
            f'{self.hr_required_shape} instead.'
        )
        need_new_shape = (
            self.data.high_res.shape[:3] != self.hr_required_shape[:3]
        )
        if need_new_shape:
            logger.warning(msg)
            warn(msg)

            hr_data_new = {}
            for f in self.data.high_res.features:
                hr_slices = [slice(sh) for sh in self.hr_required_shape]
                hr = self.data.high_res.to_dataarray().sel(variable=f).data
                hr_data_new[f] = hr[tuple(hr_slices)]

            hr_coords_new = {
                Dimension.LATITUDE: self.hr_lat_lon[..., 0],
                Dimension.LONGITUDE: self.hr_lat_lon[..., 1],
                Dimension.TIME: self.data.high_res.indexes['time'][
                    : self.hr_required_shape[2]
                ],
            }
            logger.debug(
                'Updating self.data.high_res with new shape: '
                f'{self.hr_required_shape[:3]}'
            )
            self.data.high_res = self.data.high_res.update_ds({
                **hr_coords_new,
                **hr_data_new,
            })

    def get_regridder(self):
        """Get regridder object"""
        target_meta = pd.DataFrame(
            columns=[Dimension.LATITUDE, Dimension.LONGITUDE],
            data=self.lr_lat_lon.reshape((-1, 2)),
        )
        return Regridder(
            self.data.low_res.meta,
            target_meta,
            max_workers=self.regrid_workers,
        )

    def update_lr_data(self):
        """Regrid low_res data for all requested noncached features. Load
        cached features if available and overwrite=False"""

        if self._regrid_lr:
            logger.debug('Regridding low resolution feature data.')
            regridder = self.get_regridder()

            lr_data_new = {}
            for f in self.data.low_res.features:
                lr = self.data.low_res.to_dataarray().sel(variable=f).data
                lr = lr[..., : self.lr_required_shape[2]]
                lr_data_new[f] = regridder(lr).reshape(self.lr_required_shape)

            lr_coords_new = {
                Dimension.LATITUDE: self.lr_lat_lon[..., 0],
                Dimension.LONGITUDE: self.lr_lat_lon[..., 1],
                Dimension.TIME: self.data.low_res.indexes[Dimension.TIME][
                    : self.lr_required_shape[2]
                ],
            }
            logger.debug('Updating self.data.low_res with regridded data.')
            self.data.low_res = self.data.low_res.update_ds({
                **lr_coords_new,
                **lr_data_new,
            })

    def check_regridded_lr_data(self):
        """Check for NaNs after regridding and do NN fill if needed."""
        fill_feats = []
        logger.debug('Checking for NaNs after regridding')
        qa_info = self.data.low_res.qa(stats=['nan_perc'])
        for f in self.data.low_res.features:
            nan_perc = qa_info[f]['nan_perc']
            if nan_perc > 0:
                msg = f'{f} data has {nan_perc:.3f}% NaN values!'
                if nan_perc < 10:
                    fill_feats.append(f)
                    logger.warning(msg)
                    warn(msg)
                if nan_perc >= 10:
                    logger.error(msg)
                    raise ValueError(msg)

        if any(fill_feats):
            msg = (
                'Doing nearest neighbor nan fill on low_res data for '
                f'features = {fill_feats}'
            )
            logger.info(msg)
            self.data.low_res = self.data.low_res.interpolate_na(
                features=fill_feats, method='nearest'
            )
